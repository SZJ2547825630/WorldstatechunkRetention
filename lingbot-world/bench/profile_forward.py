#!/usr/bin/env python3
"""Break down where a steady-state DiT forward spends its time.

Aggregated stats show WS-CR cuts attention context 37% vs window but only 15%
of chunk time, implying ~70% of a chunk is context-independent. This measures
what that remainder actually is: NCCL (FSDP all-gather + Ulysses all-to-all),
attention, GEMM, or elementwise/norm traffic -- and how much of the wall clock
is GPU-idle, which is the only part CUDA graphs can recover.

Run under torchrun with the same topology as the eval, e.g.:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 bench/profile_forward.py \
      --ckpt_dir /DATA/YuanZhen/Lingbot/lingbot-world-base-cam \
      --out_dir output/profile_ws_cr --method world_state_cr
"""
import argparse
import json
import logging
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wan  # noqa: E402
from bench.batch_generate import METHODS, _resolve_clips_dir  # noqa: E402
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.distributed.util import init_distributed_group  # noqa: E402

# Kernel-name buckets. Order matters: first match wins.
BUCKETS = [
    ("comm", ("nccl", "allgather", "all_gather", "reducescatter", "reduce_scatter",
              "alltoall", "all_to_all", "allreduce", "all_reduce", "broadcast")),
    ("attention", ("flash", "fmha", "attn", "mha_fwd", "cudnn_generated")),
    ("gemm", ("gemm", "cutlass", "cublas", "nvjet", "sm80_", "sm90_", "s16816",
              "wgrad", "implicit", "matmul", "tensorop")),
    ("norm_elementwise", ("norm", "elementwise", "vectorized", "silu", "gelu",
                          "softmax", "index", "cat", "copy", "fill", "reduce",
                          "unrolled", "transpose", "permute", "slice")),
]


class _ProfilingDone(Exception):
    """Unwinds out of generate() once the profiling window closes."""


def bucket_of(name: str) -> str:
    low = name.lower()
    for label, keys in BUCKETS:
        if any(k in low for k in keys):
            return label
    return "other"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-A14B")
    ap.add_argument("--size", default="480*832")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--clips_dir", default=None)
    ap.add_argument("--subset", default="default_loop")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--method", default="world_state_cr", choices=sorted(METHODS))
    ap.add_argument("--frame_num", type=int, default=241)
    ap.add_argument("--ulysses_size", type=int, default=2)
    ap.add_argument("--max_attention_size", type=int, default=47000)
    ap.add_argument("--fsdp", type=int, default=1,
                    help="1 = dit_fsdp/t5_fsdp on (matches eval); 0 = replicate weights.")
    ap.add_argument("--warmup_calls", type=int, default=32,
                    help="Forwards to run before timing (8 chunks x 4 steps).")
    ap.add_argument("--timed_calls", type=int, default=8)
    ap.add_argument("--profiled_calls", type=int, default=8)
    ap.add_argument("--trace", type=int, default=0, help="1 = also export a chrome trace.")
    return ap.parse_args()


def main():
    args = parse_args()
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

    os.makedirs(args.out_dir, exist_ok=True)
    clips_dir = _resolve_clips_dir(args.clips_dir, args.subset)
    clips = sorted(
        d for d in (os.path.join(clips_dir, x) for x in os.listdir(clips_dir))
        if os.path.isfile(os.path.join(d, "image.jpg"))
        and os.path.isfile(os.path.join(d, "poses.npy")))
    if not clips:
        raise SystemExit(f"no clips under {clips_dir}")
    cdir = clips[0]

    cfg = WAN_CONFIGS[args.task]
    use_fsdp = bool(args.fsdp) and args.ulysses_size > 1
    if rank == 0:
        logging.info(f"method={args.method} fsdp={use_fsdp} ulysses={args.ulysses_size} "
                     f"frames={args.frame_num} clip={os.path.basename(cdir)}")
    t0 = time.perf_counter()
    pipe = wan.WanI2VFast(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=local_rank, rank=rank,
        t5_fsdp=use_fsdp, dit_fsdp=use_fsdp, use_sp=args.ulysses_size > 1,
        convert_model_dtype=True, **METHODS[args.method])
    if rank == 0:
        logging.info(f"pipeline ready in {time.perf_counter()-t0:.1f}s")

    device = torch.device(f"cuda:{local_rank}")
    prof_start = args.warmup_calls + args.timed_calls
    prof_end = prof_start + args.profiled_calls
    state = {"n": 0, "timed": [], "prof_wall": 0.0}
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, with_stack=False, profile_memory=False)
    original_forward = pipe.model.forward

    def instrumented_forward(*fargs, **fkwargs):
        i = state["n"]
        state["n"] = i + 1
        if args.warmup_calls <= i < prof_start:
            torch.cuda.synchronize(device)
            t = time.perf_counter()
            out = original_forward(*fargs, **fkwargs)
            torch.cuda.synchronize(device)
            state["timed"].append(time.perf_counter() - t)
            return out
        if i == prof_start:
            torch.cuda.synchronize(device)
            prof.start()
            state["prof_wall"] = time.perf_counter()
        out = original_forward(*fargs, **fkwargs)
        if i == prof_end - 1:
            torch.cuda.synchronize(device)
            state["prof_wall"] = time.perf_counter() - state["prof_wall"]
            prof.stop()
            raise _ProfilingDone
        return out

    pipe.model.forward = instrumented_forward

    img = Image.open(os.path.join(cdir, "image.jpg")).convert("RGB")
    prompt_path = os.path.join(cdir, "prompt.txt")
    prompt = open(prompt_path).read().strip() if os.path.isfile(prompt_path) else ""
    try:
        pipe.generate(prompt, img, max_area=MAX_AREA_CONFIGS[args.size],
                      frame_num=args.frame_num, shift=cfg.sample_shift, seed=42,
                      action_path=cdir, max_attention_size=args.max_attention_size,
                      offload_model=False)
    except _ProfilingDone:
        pass
    except Exception:
        if rank == 0:
            traceback.print_exc()
        raise

    if rank != 0:
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    # Only real device kernels. CPU-side ranges (`FullyShardedDataParallel.forward`,
    # `nccl:all_to_all`, ...) report device time too, but it is the time of the
    # kernels underneath them; counting both double-counts everything. device_type
    # does not separate them reliably here, so key off the launched kernel name.
    def is_kernel(key: str) -> bool:
        if key.startswith("Memcpy") or key.startswith("Memset"):
            return True
        if "::" in key and not key.startswith("void "):
            return False  # aten::/nccl:: style CPU op
        if key.startswith("nccl:") or "." in key.split("(")[0]:
            return False  # `nccl:all_gather`, `FullyShardedDataParallel.forward`
        return True

    totals = {}
    top = []
    for evt in prof.key_averages():
        if not is_kernel(evt.key):
            continue
        self_dev = getattr(evt, "self_device_time_total", None)
        if self_dev is None:
            self_dev = getattr(evt, "self_cuda_time_total", 0)
        if not self_dev:
            continue
        ms = self_dev / 1000.0
        totals[bucket_of(evt.key)] = totals.get(bucket_of(evt.key), 0.0) + ms
        top.append((ms, evt.key))

    gpu_ms = sum(totals.values())
    wall_ms = state["prof_wall"] * 1000.0
    n = args.profiled_calls
    timed = state["timed"]
    steady_ms = (sum(timed) / len(timed) * 1000.0) if timed else float("nan")

    from wan.modules.attention import ATTN_PATH_COUNTS

    lines = []
    lines.append(f"method={args.method} fsdp={use_fsdp} ulysses={args.ulysses_size}")
    lines.append(f"attention backend calls: {dict(ATTN_PATH_COUNTS)}")
    lines.append(f"steady-state forward (sync'd, n={len(timed)}): {steady_ms:.1f} ms")
    lines.append(f"profiled window: {n} forwards, wall {wall_ms:.1f} ms "
                 f"({wall_ms/n:.1f} ms/forward, includes profiler overhead)")
    lines.append(f"GPU kernel time: {gpu_ms:.1f} ms ({gpu_ms/n:.1f} ms/forward), "
                 f"busy = {100.0*gpu_ms/max(wall_ms,1e-9):.1f}% of wall "
                 "(>100% means kernels overlapped across streams)")
    lines.append(f"GPU-idle (launch/Python/sync gap): {max(wall_ms-gpu_ms,0.0)/n:.1f} ms/forward "
                 "<- upper bound on what CUDA graphs can recover")
    lines.append("")
    lines.append(f"{'bucket':>18}  {'ms/forward':>10}  {'% GPU':>7}")
    for k in sorted(totals, key=lambda x: -totals[x]):
        lines.append(f"{k:>18}  {totals[k]/n:10.2f}  {100.0*totals[k]/gpu_ms:6.1f}%")
    lines.append("")
    lines.append("top kernels by self GPU time:")
    for ms, key in sorted(top, reverse=True)[:25]:
        lines.append(f"  {ms/n:8.2f} ms/fwd  [{bucket_of(key):>16}]  {key[:96]}")

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    tag = f"{args.method}_fsdp{int(use_fsdp)}_u{args.ulysses_size}"
    with open(os.path.join(args.out_dir, f"profile_{tag}.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out_dir, f"profile_{tag}.json"), "w") as f:
        json.dump({
            "method": args.method, "fsdp": use_fsdp, "ulysses": args.ulysses_size,
            "frame_num": args.frame_num,
            "steady_forward_ms": steady_ms, "profiled_forwards": n,
            "wall_ms_per_forward": wall_ms / n, "gpu_ms_per_forward": gpu_ms / n,
            "gpu_busy_frac": gpu_ms / max(wall_ms, 1e-9),
            "bucket_ms_per_forward": {k: v / n for k, v in totals.items()},
        }, f, indent=2)
    if args.trace:
        prof.export_chrome_trace(os.path.join(args.out_dir, f"trace_{tag}.json"))

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
