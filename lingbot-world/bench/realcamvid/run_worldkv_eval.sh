#!/usr/bin/env bash
# WorldKV-style world-memory eval on RealCam-Vid generation outputs.
# Metrics: revisit PSNR / SSIM / LPIPS / FID + throughput / ctx / peak mem.
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
# Prefer already-active lingbot env; otherwise activate via root miniconda
if [[ "${CONDA_DEFAULT_ENV:-}" != "lingbot" ]]; then
  conda activate lingbot 2>/dev/null || true
fi
source /DATA/YuanZhen/Lingbot/env.sh
cd /DATA/YuanZhen/Lingbot/lingbot-world
# Broken local proxy breaks torch hub / curl; clear unless user set KEEP_PROXY=1
if [[ "${KEEP_PROXY:-0}" != "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
fi
export TORCH_HOME="${TORCH_HOME:-/DATA/YuanZhen/.cache/torch}"
export INCEPTION_WEIGHTS="${INCEPTION_WEIGHTS:-$TORCH_HOME/hub/checkpoints/inception_v3_google-0cc3c7bd.pth}"

OUT="${REALCAMVID_OUT:-output/realcamvid_ws_vs_window_default_all_v2}"
# Paper table default: 64-clip battery, three methods
SUBSET="${REALCAMVID_SUBSET:-default_all}"
METHODS="${METHODS:-window,world_state_cr,worldkv}"
RADIUS="${RADIUS:-0.15}"
MIN_GAP="${MIN_GAP:-30}"
MAX_PAIRS="${MAX_PAIRS:-64}"
DEVICE="${DEVICE:-cuda:0}"

case "$SUBSET" in
  default_loop)   CLIPS_DIR=bench/realcamvid/clips_default_loop ;;
  default_random) CLIPS_DIR=bench/realcamvid/clips_default_random ;;
  default_all)    CLIPS_DIR=bench/realcamvid/clips_default_all ;;
  *)              CLIPS_DIR="$SUBSET" ;;
esac

bash bench/realcamvid/build_subsets.sh >/dev/null

SAVE_NAME="worldkv_memory_eval_${SUBSET}.json"
echo "[worldkv-eval] out=$OUT clips=$CLIPS_DIR methods=$METHODS"

python bench/eval_worldkv_memory.py \
  --out_dir "$OUT" \
  --clips_dir "$CLIPS_DIR" \
  --methods "$METHODS" \
  --radius "$RADIUS" \
  --min_gap "$MIN_GAP" \
  --max_pairs_per_clip "$MAX_PAIRS" \
  --device "$DEVICE" \
  --save_name "$SAVE_NAME"

echo "[worldkv-eval] DONE -> $OUT/$SAVE_NAME"
python3 - "$OUT/$SAVE_NAME" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
print()
print("| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FPS↑ | ctx | peakGB | n |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for m, r in d.get("by_method", {}).items():
    def f(k, nd=3):
        v = r.get(k)
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"
    ctx = r.get("mean_ctx")
    ctx_s = f"{ctx:.0f}" if isinstance(ctx, (int, float)) else "—"
    print(
        f"| {m} | {f('psnr')} | {f('ssim',4)} | {f('lpips',4)} | {f('fid',2)} | "
        f"{f('throughput_fps')} | {ctx_s} | {f('mean_peak_gb',2)} | {r.get('n_clips',0)} |"
    )
print("source:", p)
PY
