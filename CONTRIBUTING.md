# Contributing

Thanks for collaborating on **World-State Chunk Retention** (and optional MoSaiC = CR ⊕ SWTP).

## Source of truth

| Path | Role |
|------|------|
| **`lingbot-world/`** | **Only** editable source tree (runtime, bench, tests, selectors) |
| `docs/` | Method + experiment docs |
| `MoSaiC/` | Paper LaTeX |
| `tools/pack_mosaic.sh` | Builds a portable zip from `lingbot-world/` |

Do **not** edit `lingbot-world-mosaic/` — it is a generated package (gitignored).

## Setup (dev)

```bash
git clone <this-repo>
cd <this-repo>
source env.sh

# Python env (example)
conda create -n lingbot python=3.11 -y
conda activate lingbot
pip install -r lingbot-world/requirements.txt
# flash-attn / sageattention: install per your CUDA stack

# Place LingBot-World-Fast checkpoint and point CKPT_DIR
export CKPT_DIR=/path/to/lingbot-world-base-cam
```

Weights layout: [`weights/README.md`](weights/README.md).

## Branch & PR hygiene

1. Create a feature branch from `main` (or the agreed default branch).
2. Keep PRs focused: one method change / one bench change / one docs change.
3. Do **not** commit:
   - model weights, videos, `output/`, `artifacts/`
   - RealCam-Vid raw clips or generated `clips_default_*` symlinks
   - `third_party/WorldKV` checkouts
4. Before opening a PR:
   ```bash
   cd lingbot-world
   python -m pytest tests/ -q
   ```
5. If you change a memory policy or selector default, update:
   - `docs/method/CHUNK_RETENTION.md`
   - `tests/test_memory_policy.py`
   - root `README.md` terminology table (if CLI names change)

## Coding conventions

- Prefer `--memory_policy {window,heuristic_cr,learned_cr,world_state_cr}` over raw flags.
- Default World-State CR = future-use selector (`selector_ws_future_v1.pt`, schema `world_state.v2`).
- Attention-mass ablation remains `world_state_cr_v1` / `selector_ws_v1.pt`.
- Match existing style in `wan/image2video_fast.py` and `generate_fast.py`.
- Keep CR post-generation semantics (no host demote/retrieve) unless the PR explicitly redesigns memory.

## Benchmarks

- Official battery: `default_loop` (24) / `default_random` (40) / `default_all` (64).
- Default quality protocol: WorldKV revisit eval (`bench/eval_worldkv_memory.py`).
- See [`docs/experiments/REALCAMVID.md`](docs/experiments/REALCAMVID.md).

## Selectors

Small checkpoints under `lingbot-world/assets/selectors/` **are** meant to be tracked:

| File | Role |
|------|------|
| `selector_ws_future_v1.pt` | Default World-State CR |
| `selector_ws_v1.pt` | World-State v1 ablation |
| `selector_all4.pt` | Learned CR (5-D) |

Retrain with `lingbot-world/train_selector.py` (see script docstring).

## License

Apache-2.0 (see [`LICENSE`](LICENSE)). Upstream LingBot-World is also Apache-2.0.
