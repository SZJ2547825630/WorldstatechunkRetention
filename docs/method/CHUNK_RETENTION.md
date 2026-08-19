# Chunk Retention — Three CR Variants

LingBot-World-Fast generates video in latent chunks. Historical K/V is the only
long-range memory. **Chunk Retention** decides which past chunks stay in the
active attention context after each new chunk is produced.

## Shared mechanism

All CR variants use the same three-tier layout:

| Tier | Rule |
|------|------|
| Sink | First `sink_size` chunks, always kept |
| Recent | Last `recent_window` chunks, FIFO |
| Archive | Older candidates ranked by a utility score; keep top-K under token budget |

Default tier ratio (**1:1:2**): `sink_size=1`, `ma_kv_recent_window=1`, `ma_kv_min_keep_chunks=2`
(from RealCam-Vid ratio sweep; `ma_kv_keep_ratio=0.5`).

Ranking happens **after** the chunk is generated (post-generation retention).
Dropped archive chunks are gone for the rest of the rollout (unless demoted
into L1/L2 under Memory Consolidation — see below).

## Heuristic CR (`heuristic_cr`)

Archive score = camera/action motion peak-mean, fused with latent residual
rescue (`max(camera_motion, latent_motion)`). No learned weights.

## Learned CR (`learned_cr`)

5-D `ChunkSelector` MLP (`selector_all4.pt`, schema `learned.v0`):

`motion_norm, age_norm, cam_angular, kcent_affinity, value_norm`

Frozen baseline for ablations against World-State CR.

## World-State CR (`world_state_cr`, default = **v3**)

Default retention policy. **v3 = v2 selector + Memory Consolidation (full)**,
with default_loop-tuned knobs:

| Layer | What |
|-------|------|
| Selector (v2) | Future Coverage Oracle labels + `world_state.v2` features (`selector_ws_future_v1.pt`) |
| Ranking | `α·s + (1-α)·u_ema` with **α=0.5** |
| Consolidation C2 | L0→L1(SWTP)→L2(gist)→L3; demote bottom half of kept to L2 |
| L2 gist | **`consol_gist_tokens=64`** (sweep winner `ws_v3_a05_g64`) |
| SWTP | Required for L1/L2; spatial-cell summaries + energy-cover keep |

Aliases of the default: `world_state_cr_v3`, `world_state_cr_consol`,
`world_state_cr_future`, `ws_v3_a05_g64`.

```bash
python generate_fast.py ... --memory_policy world_state_cr
# equivalent aliases: world_state_cr_v3 / world_state_cr_consol / ws_v3_a05_g64
```

## Ablations

| Policy / method | Meaning |
|-----------------|--------|
| `world_state_cr_v2` | Former default: future-use selector only (no consol / no SWTP) |
| `world_state_cr_v1` | Attention-mass selector (`selector_ws_v1.pt`, schema `world_state.v1`) |
| `world_state_cr_ema` | v2 + EMA ranking only (no L1/L2 compression) |
| `swtp` | SWTP without CR |

See [`MEMORY_CONSOLIDATION.md`](MEMORY_CONSOLIDATION.md) and [`SWTP.md`](SWTP.md).

## CLI

```bash
python generate_fast.py ... --memory_policy heuristic_cr
python generate_fast.py ... --memory_policy learned_cr
python generate_fast.py ... --memory_policy world_state_cr      # v3 default
python generate_fast.py ... --memory_policy world_state_cr_v2   # selector-only
python generate_fast.py ... --memory_policy world_state_cr_v1   # attn-mass
```

Baseline without CR: `--memory_policy window` (sliding local attention).
