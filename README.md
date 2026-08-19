# World-State Chunk Retention for Long-Horizon Video World Models

Sparse KV inference for camera-conditioned **LingBot-World-Fast**: keep a
bounded attention context without forgetting revisited viewpoints.

This repository is the research / collaboration tree around **Chunk Retention
(CR)** — especially **World-State CR** (default **v3** = future-use selector +
Memory Consolidation + SWTP). The only orthogonal compute-axis method kept here
is **TICH** (`--enable_cond_hoist`). The product name **MoSaiC** =
World-State CR ⊕ SWTP (secondary; not required for the main claim).

> **Sole source tree:** [`lingbot-world/`](lingbot-world/)  
> Paper LaTeX: [`MoSaiC/`](MoSaiC/) · Docs: [`docs/`](docs/) · Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## Why

Autoregressive world models store history in KV caches. A **sliding window**
is fast but drops past viewpoints; full history is faithful but too expensive.
**Chunk Retention** answers:

> *Under a fixed on-device KV budget, which historical chunks stay after each
> new chunk is generated?*

Default answer: tiered **sink / recent / archive**, with archive ranked by an
11-D **World-State** ChunkSelector trained for **future coverage**.

---

## Methods (CLI)

| Name | Role | Flag |
|------|------|------|
| Sliding Window | FIFO local attention baseline | `--memory_policy window` |
| Heuristic CR | Motion-score archive ranking (ablation) | `--memory_policy heuristic_cr` |
| Learned CR | 5-D ChunkSelector (`selector_all4.pt`) | `--memory_policy learned_cr` |
| **World-State CR (v3)** | **Default**: future-use selector + consolidation + SWTP | `--memory_policy world_state_cr` |
| World-State CR v2 | Selector only (former default) | `--memory_policy world_state_cr_v2` |
| World-State CR v1 | Attention-mass ablation (`selector_ws_v1.pt`) | `--memory_policy world_state_cr_v1` |
| SWTP | Token pruning inside archive chunks (part of CR v3) | `--enable_swtp` |
| MoSaiC | World-State CR ⊕ SWTP | `world_state_cr` (+ consolidation/SWTP from v3) |
| **TICH** | Timestep-invariant condition hoisting (exact, no approx) | `--enable_cond_hoist` |

All CR variants are **post-generation** retention: generate a chunk → score
history → keep a budgeted archive. Evicted chunks are not retrieved from host.

Default World-State CR is **v3** (aliases: `world_state_cr_v3` /
`world_state_cr_consol` / `world_state_cr_future`). Use `world_state_cr_v2`
for the former selector-only default. Attention default is cuDNN SDPA
(`WAN_ATTN_BACKEND=auto`; see `docs/experiments/ATTENTION_BACKEND.md`).

---

## Repository layout

```
.
├── README.md                 ← you are here
├── LICENSE                   ← Apache-2.0
├── CONTRIBUTING.md
├── env.sh                    ← paths: LINGBOT_WORLD, CKPT_DIR, WORLDKV_ROOT, …
├── docs/
│   ├── method/               ← CHUNK_RETENTION / SWTP / MEMORY_CONSOLIDATION / TICH
│   └── experiments/          ← RealCam-Vid protocol & notes (+ ATTENTION_BACKEND)
├── MoSaiC/                   ← ICLR paper LaTeX
├── lingbot-world/            ← ★ sole source (edit here)
│   ├── generate_fast.py      ← CLI entry
│   ├── train_selector.py     ← ChunkSelector training
│   ├── wan/                  ← CR / SWTP / consolidation (+ TICH) runtime
│   ├── assets/selectors/     ← small .pt checkpoints (tracked)
│   ├── examples/             ← demo clips 00–05
│   ├── scripts/              ← run_*.sh
│   ├── bench/                ← RealCam-Vid + WorldKV baseline
│   └── tests/
├── tools/pack_mosaic.sh      ← portable zip from lingbot-world/
├── weights/                  ← symlink stubs only (see weights/README.md)
├── third_party/              ← clone WorldKV here (not vendored)
└── artifacts/                ← local experiment dumps (gitignored)
```

**Not in git (and should stay local):** model weights, generated videos,
`lingbot-world/output/`, RealCam-Vid raw clips, `third_party/WorldKV` checkout,
and the generated `lingbot-world-mosaic/` package.

---

## Requirements

- Linux + NVIDIA GPU(s); multi-GPU recommended (scripts default to 8×)
- Python **3.10–3.11**, CUDA-compatible PyTorch **≥ 2.4**
- LingBot-World-Fast checkpoint ([HuggingFace `robbyant/lingbot-world-base-cam`](https://huggingface.co/collections/robbyant/lingbot-world))
- Optional: [WorldKV](https://github.com/cvlab-kaist/WorldKV) for the retrieval baseline

---

## Quick start

### 1. Clone & environment

```bash
git clone <this-repo-url>
cd <this-repo>
source env.sh                    # sets LINGBOT_WORLD, CKPT_DIR, …

conda create -n lingbot python=3.11 -y
conda activate lingbot
pip install -r lingbot-world/requirements.txt
# Install flash-attn / sageattention matching your CUDA if needed.
```

### 2. Place weights

```bash
# Option A: export path
export CKPT_DIR=/path/to/lingbot-world-base-cam

# Option B: follow weights/README.md and symlink under weights/
```

Checkpoint path for WorldKV must contain the substring `cam`
(control-type detection). Prefer `lingbot-world-base-cam`.

### 3. Smoke run (example01, World-State CR)

```bash
source env.sh
cd "$LINGBOT_WORLD"
./scripts/run_world_state_cr.sh
# outputs under lingbot-world/output/example01/videos/
```

Same scripts live under `lingbot-world/scripts/`:

```bash
./scripts/run_window.sh
./scripts/run_world_state_cr.sh
./scripts/run_cond_hoist.sh          # TICH (default MEMORY_POLICY=window)
```

### 4. Direct CLI

```bash
cd lingbot-world
torchrun --nproc_per_node=8 generate_fast.py \
  --task i2v-A14B --size 480*832 \
  --ckpt_dir "$CKPT_DIR" \
  --image examples/01/image.jpg \
  --action_path examples/01 \
  --dit_fsdp --t5_fsdp --ulysses_size 8 \
  --frame_num 481 --convert_model_dtype \
  --sink_size 1 --max_attention_size 47000 \
  --memory_policy world_state_cr \
  --ma_kv_recent_window 1 --ma_kv_keep_ratio 0.5 \
  --ma_kv_min_keep_chunks 2 \
  --ma_kv_latent_rescue --ma_kv_latent_rescue_thr 0.08 \
  --save_dir output/example01/videos \
  --save_file output/example01/videos/world_state_cr.mp4 \
  --prompt "..."
```

Default CR tier ratio: **sink : recent : archive_min = 1 : 1 : 2**
(`sink_size=1`, `ma_kv_recent_window=1`, `ma_kv_min_keep_chunks=2`,
`ma_kv_keep_ratio=0.5`).

---

## RealCam-Vid evaluation (default protocol)

Default quality protocol aligns with **WorldKV revisit memory**:
detect revisit frames from GT poses → compare revisit vs first-visit with
**PSNR / SSIM / LPIPS / FID**, plus **FPS / ctx / peakGB**.

### Official subsets

| Subset | Clips | Role |
|--------|------:|------|
| `default_loop` | 24 | Loop / revisit (primary) |
| `default_random` | 40 | Random forward (seed=42) |
| `default_all` | 64 | loop + random |

```bash
source env.sh
cd "$LINGBOT_WORLD"
bash bench/realcamvid/build_subsets.sh   # needs local RealCam-Vid archive

# Generate (8× GPU example)
torchrun --nproc_per_node=8 bench/batch_generate.py \
  --ckpt_dir "$CKPT_DIR" \
  --subset default_loop \
  --out_dir output/realcamvid_ws_vs_window \
  --methods window,world_state_cr \
  --frame_num 481 --ulysses_size 8

# Evaluate
REALCAMVID_OUT=output/realcamvid_ws_vs_window \
REALCAMVID_SUBSET=default_loop \
METHODS=window,world_state_cr \
  bash bench/realcamvid/run_worldkv_eval.sh
```

Optional WorldKV baseline:

```bash
git clone https://github.com/cvlab-kaist/WorldKV.git third_party/WorldKV
source env.sh
cd lingbot-world
bash bench/realcamvid/run_worldkv_official.sh
```

Details: [`docs/experiments/REALCAMVID.md`](docs/experiments/REALCAMVID.md),
[`lingbot-world/bench/realcamvid/README.md`](lingbot-world/bench/realcamvid/README.md).

### Example `default_loop` numbers (24 clips)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FPS↑ | ctx | peakGB |
|--------|------:|------:|-------:|-----:|-----:|----:|-------:|
| Window | 9.830 | 0.300 | 0.726 | 67.0 | 1.18 | 41471 | 54.7 |
| World-State v1 | 10.026 | 0.327 | 0.719 | 66.2 | 1.47 | 21489 | 39.2 |
| **World-State CR (default)** | **10.138** | **0.328** | **0.713** | **61.4** | **1.47** | **21489** | **39.2** |

---

## Train / ablate the ChunkSelector

```bash
cd lingbot-world
# Collect attention-mass oracles, then train future-use (default) weights:
python train_selector.py \
  --label_type future_use_v1 --feature_schema v2 \
  --oracle path/to/oracle_dense_*.pt \
  --out assets/selectors/selector_ws_future_v1.pt
```

Oracle helpers: `scripts/collect_oracle_ws.sh`.  
Feature schemas & formulas: `wan/modules/chunk_selector.py`,
`docs/method/CHUNK_RETENTION.md`.

---

## Tests

```bash
source env.sh
cd "$LINGBOT_WORLD"
python -m pytest tests/ -q
# or: python -m unittest discover -s tests -v
```

---

## Documentation map

| Topic | Path |
|-------|------|
| Chunk Retention (CR variants) | [`docs/method/CHUNK_RETENTION.md`](docs/method/CHUNK_RETENTION.md) |
| TICH (condition hoisting) | [`docs/method/TICH_Condition_Hoisting.md`](docs/method/TICH_Condition_Hoisting.md) |
| SWTP | [`docs/method/SWTP.md`](docs/method/SWTP.md) |
| MoSaiC cascade | [`docs/method/MoSaiC.md`](docs/method/MoSaiC.md) |
| RealCam-Vid protocol | [`docs/experiments/REALCAMVID.md`](docs/experiments/REALCAMVID.md) |
| Experiment index | [`docs/experiments/INDEX.md`](docs/experiments/INDEX.md) |
| Weights layout | [`weights/README.md`](weights/README.md) |
| WorldKV third-party | [`third_party/README.md`](third_party/README.md) |
| Source package README | [`lingbot-world/README.md`](lingbot-world/README.md) |
| Paper | [`MoSaiC/iclr2027_conference.tex`](MoSaiC/iclr2027_conference.tex) |
| Contributing | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

---

## Collaboration notes

1. **Edit only `lingbot-world/`** (+ `docs/`, `MoSaiC/` as needed).
2. Never commit weights, videos, or `output/` / `artifacts/`.
3. Keep PRs small; run `pytest` before review.
4. Portable release zip (optional):
   ```bash
   bash tools/pack_mosaic.sh /tmp/lingbot-world-mosaic.zip
   ```
5. Prefer discussing API changes (`--memory_policy` names, selector defaults)
   before merging.

---

## Upstream & citation

Built on [LingBot-World](https://github.com/robbyant/lingbot-world)
(Robbyant Team). WorldKV baseline: [Yi et al., WorldKV](https://arxiv.org/abs/2605.22718).

```bibtex
@misc{worldstatecr2026,
  title  = {World-State Chunk Retention for Long-Horizon Video World Models},
  author = {Anonymous},
  year   = {2026},
  note   = {ICLR submission; code in this repository}
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
