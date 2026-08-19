# RealCam-Vid default test subsets

Official **default evaluation battery** for Chunk Retention / MoSaiC.

| Subset | Clips | Role |
|--------|-------|------|
| `default_loop` | 24 | Revisit / loop-closure (World-State CR primary) |
| `default_random` | 40 | Random forward tours, seed=42 (generalization) |
| `default_all` | 64 | loop + random (recommended simple test) |

Clip IDs are listed under `subsets/*.txt`; metadata in `subsets/manifest.json`.

## Setup (once)

Large converted clips live in the archive. Link default subsets:

```bash
cd lingbot-world/bench/realcamvid
chmod +x build_subsets.sh
./build_subsets.sh
```

Override archive location:

```bash
REALCAMVID_ARCHIVE=/path/to/realcamvid ./build_subsets.sh
```

This creates:

- `clips_default_loop/`
- `clips_default_random/`
- `clips_default_all/`

## Run

```bash
source env.sh
cd "$LINGBOT_WORLD"

# Recommended battery (64 clips)
torchrun --nproc_per_node=8 bench/batch_generate.py \
  --ckpt_dir "$CKPT_DIR" \
  --clips_dir bench/realcamvid/clips_default_all \
  --out_dir output/realcamvid_default_all \
  --methods window,heuristic_cr,learned_cr,world_state_cr \
  --frame_num 481 --ulysses_size 8

# Loop-only (24 clips) — World-State CR focus
# --clips_dir bench/realcamvid/clips_default_loop
```

Or set `REALCAMVID_SUBSET=default_all` (see `batch_generate.py --subset`).

## Default eval = WorldKV protocol

After generation, score revisit fidelity (PSNR / SSIM / LPIPS / FID) + efficiency:

```bash
REALCAMVID_OUT=output/realcamvid_ws_vs_window_default_all_v2 \
REALCAMVID_SUBSET=default_loop \
METHODS=window,world_state_cr,worldkv \
  bash bench/realcamvid/run_worldkv_eval.sh
```

## Official WorldKV baseline (8×H20)

```bash
REALCAMVID_SUBSET=default_loop \
REALCAMVID_OUT=output/realcamvid_ws_vs_window_default_all_v2 \
  bash bench/realcamvid/run_worldkv_official.sh
```

See [`docs/experiments/REALCAMVID.md`](../../../docs/experiments/REALCAMVID.md) and [`bench/worldkv/README.md`](../worldkv/README.md).
