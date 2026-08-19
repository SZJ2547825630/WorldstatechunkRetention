#!/usr/bin/env python3
"""Micro-benchmark the exact attention shape the DiT runs, across backends.

Profiling showed attention is ~40% of a steady-state forward (525 ms of 1318 ms)
and that it currently dispatches to FlashAttention 2 (`flash_fwd_kernel`) even
though the GPU is Hopper. This measures how much of that 525 ms a different
backend could recover, before committing to a from-source FA3 build.

Shapes come from i2v-A14B at 480x832: per Ulysses rank the query is one chunk
(3 latent frames x 1560 tokens) and the key is the retained KV cache.
"""
import argparse
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q_len", type=int, default=4680, help="3 latent frames x 1560")
    ap.add_argument("--kv_len", type=int, default=23400, help="retained ctx + current")
    ap.add_argument("--heads", type=int, default=20, help="40 heads / ulysses 2")
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=40, help="for the per-forward estimate")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    dev = torch.device("cuda")
    dt = torch.bfloat16
    q = torch.randn(1, args.q_len, args.heads, args.head_dim, device=dev, dtype=dt)
    k = torch.randn(1, args.kv_len, args.heads, args.head_dim, device=dev, dtype=dt)
    v = torch.randn_like(k)
    # [B, H, S, D] layout for SDPA.
    qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))

    flops = 2 * 2 * args.q_len * args.kv_len * args.heads * args.head_dim
    results = {}

    try:
        import flash_attn

        cu_q = torch.tensor([0, args.q_len], device=dev, dtype=torch.int32)
        cu_k = torch.tensor([0, args.kv_len], device=dev, dtype=torch.int32)
        qf, kf, vf = q.squeeze(0), k.squeeze(0), v.squeeze(0)
        results["fa2_varlen (current)"] = timeit(
            lambda: flash_attn.flash_attn_varlen_func(
                qf, kf, vf, cu_q, cu_k, args.q_len, args.kv_len, causal=False),
            args.iters)
        results["fa2_dense"] = timeit(
            lambda: flash_attn.flash_attn_func(q, k, v, causal=False), args.iters)
    except Exception as e:
        results["fa2_varlen (current)"] = float("nan")
        print("flash_attn failed:", e)

    try:
        import flash_attn_interface

        results["fa3"] = timeit(
            lambda: flash_attn_interface.flash_attn_func(q, k, v, causal=False),
            args.iters)
    except Exception as e:
        print("flash_attn_interface unavailable:", type(e).__name__)

    for label, backend in [("sdpa_cudnn", SDPBackend.CUDNN_ATTENTION),
                           ("sdpa_flash", SDPBackend.FLASH_ATTENTION),
                           ("sdpa_mem_efficient", SDPBackend.EFFICIENT_ATTENTION)]:
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(qt, kt, vt)
                results[label] = timeit(
                    lambda: F.scaled_dot_product_attention(qt, kt, vt), args.iters)
        except Exception as e:
            print(f"{label} unavailable: {type(e).__name__}: {str(e)[:120]}")

    base = results.get("fa2_varlen (current)", float("nan"))
    print(f"\nq={args.q_len} kv={args.kv_len} heads={args.heads} d={args.head_dim} bf16 "
          f"on {torch.cuda.get_device_name(0)}")
    print(f"{'backend':>24}  {'ms/layer':>9}  {'TFLOPS':>7}  {'vs FA2':>7}  {'est ms/forward':>14}")
    for name, ms in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"{name:>24}  {ms:9.3f}  {flops/(ms*1e-3)/1e12:7.1f}  "
              f"{base/ms:6.2f}x  {ms*args.layers:14.1f}")


if __name__ == "__main__":
    main()
