#!/usr/bin/env bash
# Standalone SWTP (token pruning, no MoCE chunk retention) @ example04.
source "$(dirname "$0")/_common.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
torchrun --nproc_per_node=8 generate_fast.py \
  --task i2v-A14B \
  --size 480*832 \
  --ckpt_dir "$CKPT_DIR" \
  --image examples/04/image.jpg \
  --action_path examples/04 \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --frame_num 481 \
  --convert_model_dtype \
  --enable_swtp \
  --swtp_keep_ratio 0.5 \
  --swtp_num_summary 64 \
  --swtp_min_saliency_gini 0.20 \
  --save_dir output/example04/videos \
  --save_file output/example04/videos/SWTP.mp4 \
  --prompt "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and timeless majesty of this ancient wonder."
