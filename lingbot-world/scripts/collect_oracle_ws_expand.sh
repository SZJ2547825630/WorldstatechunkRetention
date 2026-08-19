#!/usr/bin/env bash
# Expand World-State CR oracles: remaining examples + long RealCam-Vid clips.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${LINGBOT_ROOT:-$(cd "$ROOT/.." && pwd)}/env.sh" 2>/dev/null || true
cd "$ROOT"

CKPT_DIR="${CKPT_DIR:-/DATA/YuanZhen/Lingbot/weights/checkpoints/lingbot_world_fast}"
OUT_DIR="${OUT_DIR:-output/m1_ws}"
NPROC="${NPROC:-4}"
GPUS="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
# RealCam-Vid max pose length is ~279; use 241 for dense ranking groups.
RC_FRAME_NUM="${RC_FRAME_NUM:-241}"
EX_FRAME_NUM="${EX_FRAME_NUM:-361}"
RC_LIMIT="${RC_LIMIT:-8}"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/videos"
export CUDA_VISIBLE_DEVICES="$GPUS"

collect_one() {
  local tag="$1" image="$2" action="$3" prompt="$4" frame_num="$5"
  local oracle="$OUT_DIR/oracle_dense_${tag}.pt"
  if [[ -f "$oracle" ]]; then
    echo "[skip] $oracle"
    return 0
  fi
  echo "[collect] tag=$tag frames=$frame_num -> $oracle"
  CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node="$NPROC" generate_fast.py \
    --task i2v-A14B \
    --size 480*832 \
    --ckpt_dir "$CKPT_DIR" \
    --image "$image" \
    --action_path "$action" \
    --prompt "$prompt" \
    --dit_fsdp --t5_fsdp \
    --ulysses_size "$NPROC" \
    --frame_num "$frame_num" \
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
    --save_file "$OUT_DIR/videos/collect_dense_${tag}.mp4" \
    2>&1 | tee "$OUT_DIR/logs/collect_dense_${tag}.log"
}

# example 05 (03 has no prompt.txt — skip unless provided)
if [[ -f examples/05/prompt.txt ]]; then
  collect_one "05" "examples/05/image.jpg" "examples/05" \
    "$(tr '\n' ' ' < examples/05/prompt.txt | sed 's/[[:space:]]\+/ /g')" \
    "$EX_FRAME_NUM"
fi

# Longest RealCam-Vid clips with poses (deterministic sort by pose length).
mapfile -t RC_CLIPS < <(python - <<'PY'
import os, numpy as np
root = "bench/realcamvid/clips"
rows = []
for d in os.listdir(root):
    p = os.path.join(root, d)
    pose = os.path.join(p, "poses.npy")
    img = os.path.join(p, "image.jpg")
    pr = os.path.join(p, "prompt.txt")
    if not (os.path.isfile(pose) and os.path.isfile(img) and os.path.isfile(pr)):
        continue
    n = len(np.load(pose))
    if n >= 241:
        rows.append((-n, d))
rows.sort()
limit = int(os.environ.get("RC_LIMIT", "8"))
for _, d in rows[:limit]:
    print(d)
PY
)

i=0
for d in "${RC_CLIPS[@]}"; do
  tag=$(printf "rc%02d_%s" "$i" "$(echo "$d" | tr -c 'A-Za-z0-9' '_' | cut -c1-40)")
  collect_one "$tag" "bench/realcamvid/clips/$d/image.jpg" \
    "bench/realcamvid/clips/$d" \
    "$(tr '\n' ' ' < "bench/realcamvid/clips/$d/prompt.txt" | sed 's/[[:space:]]\+/ /g')" \
    "$RC_FRAME_NUM"
  i=$((i + 1))
done

echo "[done] oracles under $OUT_DIR"
ls -1 "$OUT_DIR"/oracle_dense_*.pt | wc -l
