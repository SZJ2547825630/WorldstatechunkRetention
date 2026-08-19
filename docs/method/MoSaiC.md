# MoSaiC — World-State CR ⊕ SWTP

**MoSaiC** is the default cascade product name:

```
KV budget ≈ (# retained chunks) × (# tokens per archive chunk)
            └── Chunk Retention ─┘   └──── SWTP ────┘
```

Official default: **World-State CR** for chunk retention + **SWTP** for
token pruning (+ optional archive diversity pool).

## Run

```bash
source env.sh
./run_mosaic.sh
# equivalent:
#   --memory_policy world_state_cr --enable_swtp \
#   --archive_diversity_pool 4 ...
```

## Ablations

Keep factors orthogonal:

| Config | Chunk policy | SWTP |
|--------|--------------|------|
| Window | `window` | off |
| Heuristic CR | `heuristic_cr` | off |
| Learned CR | `learned_cr` | off |
| World-State CR | `world_state_cr` | off |
| SWTP-only | none / window | on |
| MoSaiC | `world_state_cr` | on |

See also [`CHUNK_RETENTION.md`](CHUNK_RETENTION.md) and [`SWTP.md`](SWTP.md).
