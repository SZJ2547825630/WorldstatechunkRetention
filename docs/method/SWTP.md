# SWTP — Saliency-Weighted Token Pruning

**SWTP** sparsifies the **token dimension inside archive chunks**. It is
orthogonal to Chunk Retention (which sparsifies the chunk dimension).

## Idea

An archive chunk is a long-range memory anchor. Most of its ~4680 tokens are
redundant once sink + recent already cover the scene. SWTP keeps:

1. Top-K tokens by latent residual saliency (K chosen by **energy coverage**,
   capped by `swtp_keep_ratio`)
2. A small set of **spatial-cell** summary tokens (`swtp_num_summary`), with
   key-norm compensation so summaries can compete in softmax

Applied **lazily at archive promotion** (when a chunk leaves the recent window
and enters the archive tier), so recent chunks stay dense for local coherence.
Sink chunks are never pruned.

## Differences vs the original SWTP

| | Original | Current |
|--|--|--|
| Summary pooling | Raster strips (often a full image row) | Compact spatial cells per frame |
| Summary key scale | Raw mean (≈0.3× kept ‖k‖) | Norm-compensated to kept mean |
| Low Gini | Skip compression forever | Uniform spatial lattice + summaries |
| Keep count | Fixed `keep_ratio·T` | Energy-cover ≤ `keep_ratio·T` |

## CLI

```bash
python generate_fast.py ... --enable_swtp \
  --swtp_keep_ratio 0.5 --swtp_num_summary 64 \
  --swtp_min_saliency_gini 0.20 --swtp_energy_cover 0.9
```

## Implementation

- `lingbot-world/wan/utils/swtp.py` — pooling / energy-cover / modes
  (`standard`, `uniform`, `gist`)
- `lingbot-world/wan/image2video_fast.py` — `_compute_token_saliency`,
  `_apply_archive_compression`, standalone `_append_swtp_kv_segments`
