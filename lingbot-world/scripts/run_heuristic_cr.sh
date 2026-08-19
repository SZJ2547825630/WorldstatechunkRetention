#!/usr/bin/env bash
# Heuristic CR (motion-score archive ranking) @ example01.
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
  --sink_size 1 \
  --max_attention_size 47000 \
  --memory_policy heuristic_cr \
  --ma_kv_recent_window 1 \
  --ma_kv_keep_ratio 0.5 \
  --ma_kv_min_keep_chunks 2 \
  --ma_kv_latent_rescue \
  --ma_kv_latent_rescue_thr 0.08 \
  --save_dir output/example01/videos \
  --save_file output/example01/videos/heuristic_cr.mp4 \
  --prompt "A slow panoramic sweep around Stonehenge on a misty, overcast day, capturing the ancient standing stones in serene stillness, with soft ambient wind and distant bird calls enhancing the timeless atmosphere."
