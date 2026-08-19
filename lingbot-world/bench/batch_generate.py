#!/usr/bin/env python3
"""Batched generation: load the model ONCE per method, iterate over all clips.

Outputs:
  videos/<clip>_<method>.mp4
  stats/<clip>_<method>.json

Run under torchrun, e.g.:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 batch_generate.py \
      --ckpt_dir "$CKPT_DIR" \
      --clips_dir /path/to/clips --out_dir /path/to/out \
      --methods window,world_state_cr --ulysses_size 8
"""
import argparse, os, sys, json, time, logging, glob, gc, traceback
import torch
import torch.distributed as dist
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video

_SINK = int(os.getenv("MA_KV_SINK_SIZE", "1"))
MA = dict(enable_motion_adaptive_kv_eviction=True,
          ma_kv_recent_window=int(os.getenv("MA_KV_RECENT_WINDOW", "1")),
          ma_kv_keep_ratio=float(os.getenv("MA_KV_KEEP_RATIO", "0.5")),
          ma_kv_min_keep_chunks=int(os.getenv("MA_KV_MIN_KEEP", "2")),
          ma_kv_latent_rescue=True, ma_kv_latent_rescue_thr=0.08)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Default World-State CR = future-use selector + Memory Consolidation (v3).
_WS_CKPT = os.getenv(
    "SELECTOR_CKPT",
    os.path.join(_ROOT, "assets", "selectors", "selector_ws_future_v1.pt"),
)
_WS_V1_CKPT = os.getenv(
    "SELECTOR_WS_V1_CKPT",
    os.path.join(_ROOT, "assets", "selectors", "selector_ws_v1.pt"),
)
_LEARNED_CKPT = os.getenv(
    "SELECTOR_LEARNED_CKPT",
    os.path.join(_ROOT, "assets", "selectors", "selector_all4.pt"),
)
_WS = dict(selector="learned", selector_ckpt=_WS_CKPT)
_WS_V1 = dict(selector="learned", selector_ckpt=_WS_V1_CKPT)
_LEARNED = dict(selector="learned", selector_ckpt=_LEARNED_CKPT)
_HEURISTIC = dict(selector="heuristic")

_BASE_MOCE = dict(local_attn_size=-1, sink_size=_SINK, **MA)
_SWTP_KW = dict(enable_swtp=True, swtp_keep_ratio=0.5, swtp_num_summary=64,
                swtp_min_saliency_gini=0.20, swtp_energy_cover=0.9,
                archive_diversity_pool=4)
_CONSOL_EMA = dict(consolidation="ema", consol_beta=0.7, consol_patience=2,
                   consol_rank_alpha=0.0)
_CONSOL_FULL = dict(consolidation="full", consol_beta=0.7, consol_patience=2,
                    consol_gist_tokens=64, consol_gist_budget=512,
                    consol_rank_alpha=0.5, consol_l2_bottom_ratio=0.5)
# World-State CR versions:
#   v1 = attention-mass selector (frozen ablation)
#   v2 = future-use selector only (former default)
#   v3 = v2 + Memory Consolidation full + SWTP  ← default
_WS_V2 = dict(**_BASE_MOCE, **_WS)
_WS_V3 = dict(**_BASE_MOCE, **_WS, **_SWTP_KW, **_CONSOL_FULL)

def _v3(**kw):
    d = dict(**_WS_V3)
    d.update(kw)
    return d

METHODS = {
    "window": dict(local_attn_size=30, sink_size=_SINK),
    "heuristic_cr": dict(**_BASE_MOCE, **_HEURISTIC),
    "learned_cr": dict(**_BASE_MOCE, **_LEARNED),
    # Default World-State CR = v3 (future-use v2 selector + consolidation).
    # Tuned on default_loop: α=0.5, gist_tokens=64 (= ws_v3_a05_g64).
    "world_state_cr": dict(**_WS_V3),
    "world_state_cr_v3": dict(**_WS_V3),
    # Back-compat / sweep-name aliases of the default.
    "world_state_cr_future": dict(**_WS_V3),
    "world_state_cr_consol": dict(**_WS_V3),
    "ws_v3_a05_g64": dict(**_WS_V3),
    # Frozen ablations.
    "world_state_cr_v1": dict(**_BASE_MOCE, **_WS_V1),
    "world_state_cr_v2": dict(**_WS_V2),
    "world_state_cr_ema": dict(**_BASE_MOCE, **_WS, **_CONSOL_EMA),
    "swtp": dict(local_attn_size=-1, sink_size=_SINK, **_SWTP_KW),
    # Consolidation param sweep ablations (fixed L2 bottom-half trigger).
    "ws_v3_a0": _v3(consol_rank_alpha=0.0, consol_gist_tokens=96),
    "ws_v3_a05": _v3(consol_rank_alpha=0.5, consol_gist_tokens=96),
    "ws_v3_a1": _v3(consol_rank_alpha=1.0, consol_gist_tokens=96),
    "ws_v3_a05_g128": _v3(consol_rank_alpha=0.5, consol_gist_tokens=128),
}

_REALCAMVID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realcamvid")
_DEFAULT_SUBSETS = {
    "default_loop": "clips_default_loop",
    "default_random": "clips_default_random",
    "default_all": "clips_default_all",
}


def _resolve_clips_dir(clips_dir: str | None, subset: str | None) -> str:
    if subset:
        if subset not in _DEFAULT_SUBSETS:
            raise SystemExit(
                f"Unknown --subset {subset!r}; choose from {sorted(_DEFAULT_SUBSETS)}")
        resolved = os.path.join(_REALCAMVID, _DEFAULT_SUBSETS[subset])
        if not os.path.isdir(resolved):
            raise SystemExit(
                f"Subset dir missing: {resolved}\n"
                f"Run: bash {_REALCAMVID}/build_subsets.sh")
        return resolved
    if not clips_dir:
        raise SystemExit("Provide --clips_dir or --subset {default_loop,default_random,default_all}")
    return clips_dir


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-A14B")
    ap.add_argument("--size", default="480*832")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--clips_dir", default=None,
                    help="Directory of clip folders (image.jpg + poses.npy). "
                         "Omit when using --subset.")
    ap.add_argument("--subset", default=None,
                    choices=sorted(_DEFAULT_SUBSETS),
                    help="Official RealCam-Vid default test subset "
                         "(default_loop=24, default_random=40, default_all=64).")
    ap.add_argument("--out_dir", required=True, help="dir containing videos/ and stats/")
    ap.add_argument("--methods", default="window,world_state_cr")
    ap.add_argument("--frame_num", type=int, default=481)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--ulysses_size", type=int, default=8)
    ap.add_argument("--max_attention_size", type=int, default=47000)
    ap.add_argument("--shard", default="0/1", help="i/N: this worker handles clips[i::N]")
    ap.add_argument("--limit", type=int, default=0, help="if >0, only first N clips (smoke)")
    ap.add_argument("--timesteps_index", type=str, default=None,
                    help="Override Fast sampling schedule indices, e.g. 0,179,358,679.")
    return ap.parse_args()


def main():
    args = parse_args()
    args.clips_dir = _resolve_clips_dir(args.clips_dir, args.subset)
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.ERROR,
                        format="[%(asctime)s] %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(stream=sys.stdout)])

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size)
    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, "ulysses_size must equal world size"
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    vids = os.path.join(args.out_dir, "videos")
    stats_dir = os.path.join(args.out_dir, "stats")
    os.makedirs(vids, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    clips = sorted(d for d in glob.glob(os.path.join(args.clips_dir, "*"))
                   if os.path.isfile(os.path.join(d, "image.jpg"))
                   and os.path.isfile(os.path.join(d, "poses.npy")))
    si, sn = (int(x) for x in args.shard.split("/"))
    clips = clips[si::sn]
    if args.limit > 0:
        clips = clips[:args.limit]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if rank == 0:
        logging.info(f"shard={args.shard} clips={len(clips)} methods={methods} ulysses={args.ulysses_size}")
        if len(methods) > 1:
            # FSDP/NCCL teardown via `del pipe; empty_cache()` does not fully reclaim
            # CUDA state; subsequent methods' peak_memory_allocated_gb is inflated by
            # ~15–20GB (not an algorithm leak). Prefer one method per process
            # (run_dp2.sh loops METHODS that way).
            logging.warning(
                "Multiple methods in one process: peak_memory after the first method "
                "is NOT comparable (FSDP teardown residue). Use one --methods entry "
                "per torchrun, or run_dp2.sh which launches each method fresh."
            )

    for method in methods:
        if method not in METHODS:
            if rank == 0:
                logging.error(f"unknown method {method}; skip")
            continue
        ctor = dict(METHODS[method])
        t0 = time.perf_counter()
        if rank == 0:
            logging.info(f"==== building pipeline for method={method} ({ctor}) ====")
        use_fsdp = args.ulysses_size > 1
        pipe = wan.WanI2VFast(
            config=cfg, checkpoint_dir=args.ckpt_dir, device_id=local_rank, rank=rank,
            t5_fsdp=use_fsdp, dit_fsdp=use_fsdp, use_sp=use_fsdp,
            convert_model_dtype=True, **ctor)
        gen_kwargs = {}
        if args.timesteps_index:
            gen_kwargs["timesteps_index"] = [
                int(x.strip()) for x in args.timesteps_index.split(",") if x.strip()]
        if rank == 0:
            logging.info(f"[{method}] pipeline ready in {time.perf_counter()-t0:.1f}s")

        done = skipped = failed = 0
        for ci, cdir in enumerate(clips):
            clip = os.path.basename(cdir)
            vid = os.path.join(vids, f"{clip}_{method}.mp4")
            stt = os.path.join(stats_dir, f"{clip}_{method}.json")
            if os.path.isfile(vid) and os.path.isfile(stt):
                skipped += 1
                continue
            try:
                img = Image.open(os.path.join(cdir, "image.jpg")).convert("RGB")
                prompt_path = os.path.join(cdir, "prompt.txt")
                prompt = open(prompt_path).read().strip() if os.path.isfile(prompt_path) else ""
                video = pipe.generate(
                    prompt, img,
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=cfg.sample_shift,
                    seed=args.base_seed,
                    action_path=cdir,
                    max_attention_size=args.max_attention_size,
                    offload_model=False,
                    **gen_kwargs,
                )
                if rank == 0 and video is not None:
                    save_video(tensor=video[None], save_file=vid, fps=cfg.sample_fps,
                               nrow=1, normalize=True, value_range=(-1, 1))
                    payload = {
                        "config": {"method": method, "frame_num": args.frame_num},
                        "example": clip,
                        "stats": getattr(pipe, "last_generation_stats", {}) or {},
                    }
                    with open(stt, "w") as f:
                        json.dump(payload, f, indent=2)
                done += 1
            except Exception:
                failed += 1
                if rank == 0:
                    logging.error(f"[{method}] clip={clip} failed:\n{traceback.format_exc()}")
            if world_size > 1:
                dist.barrier()
            gc.collect()
            torch.cuda.empty_cache()
        if rank == 0:
            logging.info(f"[{method}] done={done} skipped={skipped} failed={failed}")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        logging.info("ALL_METHODS_DONE")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
