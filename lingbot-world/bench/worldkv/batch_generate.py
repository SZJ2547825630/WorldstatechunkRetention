#!/usr/bin/env python3
"""Batch RealCam-Vid generation with official WorldKV (training-free retrieval+compression).

Uses third_party/WorldKV as the wan implementation. Outputs match MoSaiC bench layout:
  videos/<clip>_worldkv.mp4
  stats/<clip>_worldkv.json

Example (2-GPU SP shard):
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 bench/worldkv/batch_generate.py \\
    --ckpt_dir /path/to/lingbot-world-base-cam \\
    --subset default_loop --out_dir output/realcamvid_worldkv \\
    --ulysses_size 2 --shard 0/4
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
from PIL import Image

_LINGBOT_WORLD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LINGBOT_ROOT = os.path.dirname(_LINGBOT_WORLD)
_WORLDKV_ROOT = os.environ.get(
    "WORLDKV_ROOT",
    os.path.join(_LINGBOT_ROOT, "third_party", "WorldKV"),
)
if not os.path.isdir(_WORLDKV_ROOT):
    raise SystemExit(f"WorldKV root not found: {_WORLDKV_ROOT}")
# Prefer official WorldKV package over local lingbot-world/wan
sys.path.insert(0, _WORLDKV_ROOT)

import wan  # noqa: E402  (WorldKV)
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.distributed.util import init_distributed_group  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402

_REALCAMVID = os.path.join(_LINGBOT_WORLD, "bench", "realcamvid")
_DEFAULT_SUBSETS = {
    "default_loop": "clips_default_loop",
    "default_random": "clips_default_random",
    "default_all": "clips_default_all",
}

METHOD = "worldkv"


def _resolve_clips_dir(clips_dir: str | None, subset: str | None) -> str:
    if subset:
        if subset not in _DEFAULT_SUBSETS:
            raise SystemExit(f"Unknown --subset {subset!r}; choose from {sorted(_DEFAULT_SUBSETS)}")
        resolved = os.path.join(_REALCAMVID, _DEFAULT_SUBSETS[subset])
        if not os.path.isdir(resolved):
            raise SystemExit(f"Subset dir missing: {resolved}\nRun: bash {_REALCAMVID}/build_subsets.sh")
        return resolved
    if not clips_dir:
        raise SystemExit("Provide --clips_dir or --subset")
    return clips_dir


def parse_args():
    ap = argparse.ArgumentParser(description="Official WorldKV batch generate on RealCam-Vid")
    ap.add_argument("--task", default="i2v-A14B")
    ap.add_argument("--size", default="480*832")
    ap.add_argument("--ckpt_dir", required=True,
                    help="Must contain 'cam' in path for WorldKV control_type detection")
    ap.add_argument("--clips_dir", default=None)
    ap.add_argument("--subset", default=None, choices=sorted(_DEFAULT_SUBSETS))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frame_num", type=int, default=481)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--ulysses_size", type=int, default=2)
    ap.add_argument("--max_attention_size", type=int, default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    # WorldKV official defaults (README)
    ap.add_argument("--retrieval_frames", type=int, default=18)
    ap.add_argument("--kv_compression_keep_ratio", type=float, default=0.5)
    ap.add_argument("--kv_bank_on_gpu", action="store_true", default=True)
    ap.add_argument("--no_kv_bank_on_gpu", action="store_true",
                    help="Store KV bank on CPU (slower, less VRAM)")
    ap.add_argument("--convert_model_dtype", action="store_true", default=True)
    return ap.parse_args()


def main():
    args = parse_args()
    args.clips_dir = _resolve_clips_dir(args.clips_dir, args.subset)
    kv_bank_on_gpu = bool(args.kv_bank_on_gpu) and not args.no_kv_bank_on_gpu

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    if "cam" not in os.path.abspath(args.ckpt_dir):
        logging.warning(
            "ckpt_dir has no 'cam' substring; WorldKV may mis-detect control_type. "
            "Prefer lingbot-world-base-cam."
        )

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

    clips = sorted(
        d for d in glob.glob(os.path.join(args.clips_dir, "*"))
        if os.path.isfile(os.path.join(d, "image.jpg"))
        and os.path.isfile(os.path.join(d, "poses.npy"))
    )
    si, sn = (int(x) for x in args.shard.split("/"))
    clips = clips[si::sn]
    if args.limit > 0:
        clips = clips[: args.limit]

    if rank == 0:
        logging.info(
            f"WorldKV batch: root={_WORLDKV_ROOT} shard={args.shard} clips={len(clips)} "
            f"ulysses={args.ulysses_size} retrieval_frames={args.retrieval_frames} "
            f"kv_bank_on_gpu={kv_bank_on_gpu}"
        )

    use_fsdp = args.ulysses_size > 1
    t_build = time.perf_counter()
    pipe = wan.WanI2VFast(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=use_fsdp,
        dit_fsdp=use_fsdp,
        use_sp=use_fsdp,
        convert_model_dtype=args.convert_model_dtype,
    )
    if rank == 0:
        logging.info(f"pipeline ready in {time.perf_counter() - t_build:.1f}s")

    done = skipped = failed = 0
    for ci, cdir in enumerate(clips):
        clip = os.path.basename(cdir)
        vid = os.path.join(vids, f"{clip}_{METHOD}.mp4")
        stt = os.path.join(stats_dir, f"{clip}_{METHOD}.json")
        if os.path.isfile(vid) and os.path.isfile(stt):
            skipped += 1
            continue
        try:
            img = Image.open(os.path.join(cdir, "image.jpg")).convert("RGB")
            prompt_path = os.path.join(cdir, "prompt.txt")
            prompt = open(prompt_path).read().strip() if os.path.isfile(prompt_path) else ""
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            video = pipe.generate(
                prompt,
                img,
                action_path=cdir,
                chunk_size=3,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=cfg.sample_shift,
                seed=args.base_seed,
                offload_model=False,
                max_attention_size=args.max_attention_size,
                use_retrieval=True,
                retrieval_frames=args.retrieval_frames,
                retrieval_rope_correction=False,
                full_kv=False,
                sliding_window=0,
                kv_compression_enable=True,
                kv_compression_keep_ratio=args.kv_compression_keep_ratio,
                kv_compression_anchor_rotate=False,
                kv_compression_at_store=True,
                kv_compression_pooled=True,
                kv_bank_on_gpu=kv_bank_on_gpu,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            peak = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
                if torch.cuda.is_available() else None
            )
            if rank == 0 and video is not None:
                save_video(
                    tensor=video[None],
                    save_file=vid,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                # WorldKV window ≈ sink + retrieval_frames + recent(6)
                approx_ctx_frames = 1 + args.retrieval_frames + 6
                payload = {
                    "config": {
                        "method": METHOD,
                        "frame_num": args.frame_num,
                        "retrieval_frames": args.retrieval_frames,
                        "kv_compression_keep_ratio": args.kv_compression_keep_ratio,
                        "kv_bank_on_gpu": kv_bank_on_gpu,
                        "worldkv_root": _WORLDKV_ROOT,
                    },
                    "example": clip,
                    "stats": {
                        "total_generation_time_s": elapsed,
                        "peak_memory_allocated_gb": peak,
                        # approx active attention frames (latent); token count filled post-hoc if known
                        "worldkv_active_latent_frames": approx_ctx_frames,
                        "retrieval_frames": args.retrieval_frames,
                    },
                }
                with open(stt, "w") as f:
                    json.dump(payload, f, indent=2)
                logging.info(
                    f"[{ci+1}/{len(clips)}] {clip} ok time={elapsed:.1f}s peak={peak:.2f}GB"
                )
            done += 1
        except Exception:
            failed += 1
            if rank == 0:
                logging.error(f"clip={clip} failed:\n{traceback.format_exc()}")
        if world_size > 1:
            dist.barrier()
        gc.collect()
        torch.cuda.empty_cache()

    if rank == 0:
        logging.info(f"[{METHOD}] done={done} skipped={skipped} failed={failed}")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
