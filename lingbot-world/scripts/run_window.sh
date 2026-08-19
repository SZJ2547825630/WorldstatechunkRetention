#!/usr/bin/env bash
# Sliding-window KV baseline @ example01 (Stonehenge).
source "$(dirname "$0")/_common.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
torchrun --nproc_per_node=8 generate_fast.py \
  --task i2v-A14B \
  --size 480*832 \
  --ckpt_dir "$CKPT_DIR" \
  --image examples/01/image.jpg \
  --action_path examples/01 \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --frame_num 481 \
  --convert_model_dtype \
  --base_seed 42 \
  --sink_size 1 \
  --local_attn_size 30 \
  --max_attention_size 47000 \
  --memory_policy window \
  --save_dir output/example01/videos \
  --save_file output/example01/videos/window.mp4 \
  --prompt "A slow panoramic sweep around Stonehenge on a misty, overcast day, capturing the ancient standing stones in serene stillness, with soft ambient wind and distant bird calls enhancing the timeless atmosphere."
