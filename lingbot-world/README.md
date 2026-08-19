# lingbot-world (sole source tree)

Inference runtime for **World-State Chunk Retention** (and optional MoSaiC)
on LingBot-World-Fast. Edit code **here only** — do not maintain a parallel
copy under `lingbot-world-mosaic/`.

Parent overview: [`../README.md`](../README.md).

## Methods

| CLI `--memory_policy` | Meaning |
|-----------------------|---------|
| `window` | Sliding-window baseline |
| `heuristic_cr` | Motion-score CR |
| `learned_cr` | 5-D ChunkSelector (`assets/selectors/selector_all4.pt`) |
| `world_state_cr` | **Default v3**: future-use selector + consolidation + SWTP |
| `world_state_cr_v2` | Selector only (former default) |
| `world_state_cr_v1` | Attention-mass ablation (`selector_ws_v1.pt`) |
| `world_state_cr_future` / `_v3` / `_consol` | Aliases of `world_state_cr` |

Orthogonal compute-axis: **TICH** (`--enable_cond_hoist`; default off).  
Attention default: cuDNN SDPA (`WAN_ATTN_BACKEND=auto`).  
**MoSaiC** naming still means CR ⊕ SWTP; v3 already enables SWTP + consolidation.

## Entry points

| Script | Method |
|--------|--------|
| `scripts/run_window.sh` | Sliding window |
| `scripts/run_heuristic_cr.sh` | Heuristic CR |
| `scripts/run_learned_cr.sh` | Learned CR |
| `scripts/run_world_state_cr.sh` | World-State CR v3 (default) |
| `scripts/run_ws_cr_stable.sh` | World-State CR (stable 4-GPU recipe) |
| `scripts/run_cond_hoist.sh` | TICH (exact condition hoisting) |
| `scripts/run_swtp.sh` | SWTP only |
| `scripts/run_mosaic.sh` | MoSaiC (CR ⊕ SWTP) |

Batch: `bench/batch_generate.py`  
Eval: `bench/eval_worldkv_memory.py` / `bench/realcamvid/run_worldkv_eval.sh`  
WorldKV baseline: `bench/worldkv/` + `bench/realcamvid/run_worldkv_official.sh`

## Layout

```
generate_fast.py          CLI + --memory_policy mapping
train_selector.py         ChunkSelector training
wan/image2video_fast.py   CR eviction + SWTP + consolidation + TICH
wan/modules/chunk_selector.py
wan/utils/selector_defaults.py
wan/utils/future_use_labels.py
wan/utils/memory_consolidation.py
wan/utils/swtp.py
assets/selectors/*.pt
examples/00–05
bench/realcamvid/         default_loop / default_random / default_all
tests/
```

## Assets

```
assets/selectors/selector_ws_future_v1.pt   # World-State CR (default)
assets/selectors/selector_ws_v1.pt          # v1 ablation
assets/selectors/selector_all4.pt           # Learned CR
```

## Tests

```bash
python -m pytest tests/ -q
```

## Environment

```bash
source ../env.sh   # from repo root
# requires CKPT_DIR → LingBot-World-Fast checkpoint
```
