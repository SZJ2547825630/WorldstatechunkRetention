# Bench

Batch driver for official methods:

```bash
torchrun --nproc_per_node=8 batch_generate.py \
  --ckpt_dir "$CKPT_DIR" \
  --subset default_all \
  --out_dir output/realcamvid_default_all \
  --methods window,heuristic_cr,learned_cr,world_state_cr,swtp
```

**Default RealCam-Vid test battery** (24 loop + 40 random = 64 clips):
[`realcamvid/README.md`](realcamvid/README.md) · [`realcamvid/subsets/manifest.json`](realcamvid/subsets/manifest.json)

```bash
bash realcamvid/run_default.sh   # REALCAMVID_SUBSET=default_all|default_loop|default_random
```

Large historical dumps: `../../artifacts/archive_pre_paper/bench/`.
