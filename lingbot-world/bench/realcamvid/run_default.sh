#!/usr/bin/env bash
# Default RealCam-Vid battery: 64 clips (24 loop + 40 random), four CR methods + window.
source "$(dirname "$0")/../../scripts/_common.sh"

SUBSET="${REALCAMVID_SUBSET:-default_all}"
OUT="${REALCAMVID_OUT:-output/realcamvid_${SUBSET}}"

bash "$(dirname "$0")/build_subsets.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
torchrun --nproc_per_node=8 bench/batch_generate.py \
  --ckpt_dir "$CKPT_DIR" \
  --subset "$SUBSET" \
  --out_dir "$OUT" \
  --methods window,heuristic_cr,learned_cr,world_state_cr \
  --frame_num 481 \
  --ulysses_size 8 \
  --max_attention_size 47000 \
  --base_seed 42
