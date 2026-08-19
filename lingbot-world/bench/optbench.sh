#!/bin/bash
# A/B micro-benchmark for MoSaiC CUDA optimizations.
# Runs MoSaiC on ONE long (>30s) clip, deterministic (seed 42), ulysses=2 on GPUs 0,1, under a
# reference config and an optimization config, then compares peak memory / chunk time and verifies
# output equivalence (pixel diff). Needs 2 free GPUs.
#
# Usage:  bench/optbench.sh <tag> "<ENV=VAL ...>"
#   e.g.  bench/optbench.sh kvbuffer "MOSAIC_KV_BUFFER=1"
#         bench/optbench.sh compile  "MOSAIC_COMPILE=1"
# The reference run (plain MoSaiC) is generated once and cached in optbench/ref/.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbot
cd /DATA/YuanZhen/Lingbot/lingbot-world
export TMPDIR=/DATA/.claude-tmp

TAG=${1:?usage: optbench.sh <tag> \"<ENV=VAL ...>\"}
ENVS=${2:-}
OUT=bench/realcamvid/optbench
CKPT=/DATA/YuanZhen/Lingbot/lingbot-world-base-cam
CLIPS=bench/realcamvid/clips_long   # 481-frame (>30s) clips stress the KV cache

run() {  # $1=subtag  $2=extra env assignments
  local sub=$1 envs=$2
  rm -rf "$OUT/$sub"
  env $envs CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29571 \
      bench/batch_generate.py --ckpt_dir "$CKPT" \
      --clips_dir "$CLIPS" --out_dir "$OUT/$sub" \
      --methods MoSaiC --ulysses_size 2 --shard 0/1 --limit 1 \
      > "$OUT/${sub}.log" 2>&1
  echo "[optbench] $sub done -> $OUT/$sub (log: $OUT/${sub}.log)"
}

mkdir -p "$OUT"
[ -f "$OUT/ref/stats/"*_MoSaiC.json ] 2>/dev/null || run ref ""
run "$TAG" "$ENVS"
echo "================ optbench: ref vs $TAG ================"
python bench/realcamvid/compare_optbench.py "$OUT/ref" "$OUT/$TAG"
