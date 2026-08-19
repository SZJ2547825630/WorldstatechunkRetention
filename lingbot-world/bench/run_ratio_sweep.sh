#!/bin/bash
# MoCE tier-ratio sweep: sink:recent:archive(min_keep). world_state_cr only, 12-clip clips_ratio subset.
# One config per batch_generate run (tier sizes via env). Outputs sweep_ratio/<tag>/.
set -uo pipefail
cd /DATA/YuanZhen/Lingbot/lingbot-world
SWEEP=bench/realcamvid/sweep_ratio
CLIPS=bench/realcamvid/clips_ratio
# vary one axis at a time around best-so-far 1:2:2
CONFIGS=${CONFIGS:-"1 2 2|1 1 2|1 3 2|1 2 1|1 2 3|2 2 2|0 2 2"}
mkdir -p "$SWEEP"

IFS='|' read -ra CFGS <<< "$CONFIGS"
for cfg in "${CFGS[@]}"; do
  read -r sink recent arch <<< "$cfg"
  tag="s${sink}_r${recent}_a${arch}"
  OUT="$SWEEP/$tag"
  if ls "$OUT"/stats/*_world_state_cr.json >/dev/null 2>&1; then
    echo "[ratio-sweep] $tag already done, skip"; continue
  fi
  echo "===== [ratio-sweep] $tag : sink=$sink recent=$recent archive(min_keep)=$arch ====="
  MA_KV_SINK_SIZE=$sink MA_KV_RECENT_WINDOW=$recent MA_KV_MIN_KEEP=$arch \
    N=4 U=2 GPU_BASE=0 METHODS=world_state_cr \
    CLIPS="$CLIPS" OUT="$OUT" bash bench/run_dp2.sh
done
echo "[ratio-sweep] ALL DONE -> $SWEEP"
