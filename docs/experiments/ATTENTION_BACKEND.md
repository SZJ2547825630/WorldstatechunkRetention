# Attention backend (cuDNN default, FA2 ablation)

Default is **cuDNN fused Hopper SDPA** (`WAN_ATTN_BACKEND=auto`) on unmasked self-attention. FlashAttention 2 remains available as an ablation / paper-table path (`WAN_ATTN_BACKEND=flash`).

## Why this exists

A steady-state DiT forward under World-State CR (Ulysses=2, FSDP on) spends roughly:

| bucket | ms / forward | share |
|--------|-------------:|------:|
| GEMM | ~590 | 45% |
| Attention | ~525 (FA2) → ~334 (cuDNN) | 40% → 26% |
| elementwise / norm | ~150 | 11% |
| NCCL | ~100 | 7% (mostly overlapped) |
| GPU idle | ~0 | 0% |

CUDA Graphs (FasterWAM-style `torch.compile(reduce-overhead)`) recover idle launch time; here the GPU is already saturated, so graphs do not help. Attention is the largest *single* exact-math lever that does not change the memory policy.

Micro-bench at the real shape (`q=4680`, `kv=23400`, 20 heads, `d=128`, bf16):

| backend | ms / layer | vs FA2 |
|---------|-----------:|-------:|
| FA2 varlen | 13.3 | 1.00× |
| cuDNN SDPA (default) | 8.4 | **1.57×** |

## How to switch

```bash
# default (no env needed): cuDNN on eligible self-attn
N=4 U=2 METHODS=world_state_cr OUT=... CLIPS=... bash bench/run_dp2.sh --frame_num 481

# restore FA2 for paper / memory-policy ablations
WAN_ATTN_BACKEND=flash N=4 U=2 METHODS=world_state_cr ...
```

Dispatch lives in `lingbot-world/wan/modules/attention.py` (`flash_attention`). Only unmasked, non-GQA, `head_dim≤128` calls take cuDNN; padded cross-attn and non-square causal stay on FA2.

## Measured on `default_loop` (n=24, World-State CR v2)

| | window | WS-CR FA2 | **WS-CR cuDNN (default)** |
|--|--:|--:|--:|
| FPS↑ | 1.18 | 1.47 | **1.66** (~1.40× vs window) |
| avg chunk time | 8.46 s | 6.43 s | **5.53 s** |
| peak GB | 54.7 | 39.2 | 39.8 |
| PSNR↑ | 9.830 | 10.026 | **9.854** (still **> window**) |
| SSIM↑ | 0.300 | 0.327 | 0.324 |
| LPIPS↓ | 0.726 | 0.719 | 0.720 |
| FID↓ | 67.00 | 66.20 | 61.24 |

cuDNN vs FA2 on the same WS-CR policy: PSNR −0.17 dB on this eval file (earlier paired FA2-future run was −0.28 dB); 17/24 clips slightly worse. Single-call parity is within one bf16 ULP; the gap accumulates over 40×4×40 attention calls.

## Profiling helpers

```bash
torchrun --nproc_per_node=2 bench/profile_forward.py \
  --ckpt_dir /DATA/YuanZhen/Lingbot/lingbot-world-base-cam \
  --out_dir output/profile_v2 --method world_state_cr --fsdp 1 --ulysses_size 2

python bench/bench_attention.py
```

## Takeaway

cuDNN is the default runtime for WS-CR v2: ~1.40× vs window throughput while keeping revisit PSNR above window on `default_loop`. Use `WAN_ATTN_BACKEND=flash` when you need bit-stable FA2 numbers. Further compute-axis work in this tree is TICH (`--enable_cond_hoist`), not CUDA Graphs.
