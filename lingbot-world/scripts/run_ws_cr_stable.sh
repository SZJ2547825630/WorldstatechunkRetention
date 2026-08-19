#!/usr/bin/env bash
# Stable track: World-State CR memory axis (window/CR/TICH tree).
# Aligned with experiment settings (4×GPU, FRAME=361, example01).
#
#   EXAMPLE=01 FRAME=361 NPROC=4 ULYSSES=4 bash scripts/run_ws_cr_stable.sh
source "$(dirname "$0")/_common.sh"

EXAMPLE="${EXAMPLE:-01}"
FRAME="${FRAME:-361}"
NPROC="${NPROC:-4}"
ULYSSES="${ULYSSES:-4}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

SAVE_DIR="${SAVE_DIR:-output/example${EXAMPLE}/videos}"
SAVE_FILE="${SAVE_FILE:-$SAVE_DIR/world_state_cr_stable.mp4}"
mkdir -p "$SAVE_DIR"

PROMPT_FILE="examples/${EXAMPLE}/prompt.txt"
if [ -f "$PROMPT_FILE" ]; then
  PROMPT="$(cat "$PROMPT_FILE")"
else
  PROMPT="A cinematic camera move through the scene."
fi

CUDA_VISIBLE_DEVICES="$GPUS" \
torchrun --nproc_per_node="$NPROC" generate_fast.py \
  --task i2v-A14B \
  --size 480*832 \
  --ckpt_dir "$CKPT_DIR" \
  --image "examples/${EXAMPLE}/image.jpg" \
  --action_path "examples/${EXAMPLE}" \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size "$ULYSSES" \
  --frame_num "$FRAME" \
  --convert_model_dtype \
  --sink_size 1 \
  --max_attention_size 47000 \
  --memory_policy world_state_cr \
  --ma_kv_recent_window 1 \
  --ma_kv_keep_ratio 0.5 \
  --ma_kv_min_keep_chunks 2 \
  --ma_kv_latent_rescue \
  --ma_kv_latent_rescue_thr 0.08 \
  --save_dir "$SAVE_DIR" \
  --save_file "$SAVE_FILE" \
  --prompt "$PROMPT"
