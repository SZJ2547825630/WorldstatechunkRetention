#!/usr/bin/env bash
# Collect attention-mass oracles with world-state pose metadata for World-State CR selector training.
# Dense retention (keep_ratio=1) so archive candidates stay scorable.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${LINGBOT_ROOT:-$(cd "$ROOT/.." && pwd)}/env.sh" 2>/dev/null || true
cd "$ROOT"

CKPT_DIR="${CKPT_DIR:-/DATA/YuanZhen/Lingbot/weights/checkpoints/lingbot_world_fast}"
OUT_DIR="${OUT_DIR:-output/m1_ws}"
FRAME_NUM="${FRAME_NUM:-361}"
NPROC="${NPROC:-4}"
GPUS="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
CLIPS="${CLIPS:-00 01 02 04}"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/videos"
export CUDA_VISIBLE_DEVICES="$GPUS"

for id in $CLIPS; do
  ex="examples/$id"
  oracle="$OUT_DIR/oracle_dense_${id}.pt"
  if [[ -f "$oracle" ]]; then
    echo "[skip] $oracle already exists"
    continue
  fi
  prompt="$(tr '\n' ' ' < "$ex/prompt.txt" | sed 's/[[:space:]]\+/ /g')"
  echo "[collect] clip=$id frame_num=$FRAME_NUM -> $oracle"
  CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node="$NPROC" generate_fast.py \
    --task i2v-A14B \
    --size 480*832 \
    --ckpt_dir "$CKPT_DIR" \
    --image "$ex/image.jpg" \
    --action_path "$ex" \
    --prompt "$prompt" \
    --dit_fsdp --t5_fsdp \
    --ulysses_size "$NPROC" \
    --frame_num "$FRAME_NUM" \
    --convert_model_dtype \
    --base_seed 42 \
    --sink_size 1 \
    --local_attn_size -1 \
    --max_attention_size 160000 \
    --memory_policy heuristic_cr \
    --ma_kv_recent_window 2 \
    --ma_kv_keep_ratio 1.0 \
    --ma_kv_min_keep_chunks 40 \
    --ma_kv_latent_rescue \
    --ma_kv_latent_rescue_thr 0.08 \
    --collect_oracle \
    --oracle_out "$oracle" \
    --oracle_probe_every 4 \
    --save_dir "$OUT_DIR/videos" \
    --save_file "$OUT_DIR/videos/collect_dense_${id}.mp4" \
    2>&1 | tee "$OUT_DIR/logs/collect_dense_${id}.log"
done

echo "[done] oracles under $OUT_DIR"
