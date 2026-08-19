# Memory Consolidation

In-cache **lifecycle management** for World-State CR. Distinct from WorldKV
(pose-top-k external bank + anchor/novelty prune): Consolidation never builds
an append-only retrieval bank; it re-ranks and demotes segments that already
live in the dynamic KV cache.

**Default World-State CR (v3)** enables Consolidation `full` (+ SWTP) with
default_loop-tuned knobs: `consol_rank_alpha=0.5`, `consol_gist_tokens=64`,
`consol_l2_bottom_ratio=0.5`. See [`CHUNK_RETENTION.md`](CHUNK_RETENTION.md).

## Mechanisms

| ID | Name | What it does |
|----|------|----------------|
| C1 | EMA utility + hysteresis | `u ← βu + (1-β)s`; rank mixes `α·s + (1-α)·u` |
| C2 | Tiered demotion | L0 full → L1 SWTP → L2 gist → L3 drop; bottom half of kept → L2 |
| Probe | Revisit coverage | Fraction of pose-revisits whose target chunk is still in cache |

C3 (cross-chunk merge) is deferred.

## CLI

```bash
# Default (v3 / ws_v3_a05_g64): already on via --memory_policy world_state_cr

# Explicit knobs (defaults shown)
--enable_motion_adaptive_kv_eviction --enable_swtp --consolidation full \
  --consol_beta 0.7 --consol_patience 2 \
  --consol_rank_alpha 0.5 --consol_l2_bottom_ratio 0.5 \
  --consol_gist_tokens 64 --consol_gist_budget 512
```

Batch methods: default `world_state_cr` (= `ws_v3_a05_g64`);
ablations `world_state_cr_v2`, `ws_v3_a0`, `ws_v3_a1`, …

## Peak memory note

When comparing methods, run **one method per process** (`run_dp2.sh` does this).
Reloading FSDP in the same process leaves GPU residue that falsely inflates
the next method's `peak_memory_allocated_gb` — not an algorithm leak.

## Files

- `lingbot-world/wan/utils/memory_consolidation.py` — EMA / tiers / probe
- `lingbot-world/wan/image2video_fast.py` — wired into `_evict_motion_adaptive_kv_cache`
- Stats: `consolidation`, `revisit_coverage`, `tier_counts`
