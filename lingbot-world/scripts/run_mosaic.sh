#!/usr/bin/env bash
# MoSaiC = World-State CR + SWTP @ example01.
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
  --memory_policy world_state_cr \
  --ma_kv_recent_window 1 \
  --ma_kv_keep_ratio 0.5 \
  --ma_kv_min_keep_chunks 2 \
  --ma_kv_latent_rescue \
  --ma_kv_latent_rescue_thr 0.08 \
  --enable_swtp \
  --swtp_keep_ratio 0.5 \
  --swtp_num_summary 64 \
  --swtp_min_saliency_gini 0.20 \
  --archive_diversity_pool 4 \
  --save_dir output/example01/videos \
  --save_file output/example01/videos/MoSaiC.mp4 \
  --prompt "A slow panoramic sweep around Stonehenge on a misty, overcast day, capturing the ancient standing stones in serene stillness, with soft ambient wind and distant bird calls enhancing the timeless atmosphere."
