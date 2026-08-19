#!/bin/bash
# Parallel VBench scoring of run200_vbench long videos (baseline vs MoSaiC), custom_input mode.
# One dimension per GPU (7 dims -> GPU 0-6); each worker scores ALL baseline then ALL MoSaiC videos
# for its dimension (exact aggregate, no sharding approximation). VBench is slow on long videos
# (RAFT ~34s/video), so per-dim parallelism keeps wall-time ~= slowest single dim.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbot
cd /DATA/YuanZhen/Lingbot/lingbot-world
export HF_ENDPOINT=https://hf-mirror.com   # offline by now; mirror just in case

RUN=bench/realcamvid/run200_vbench
OUT=$RUN/vbench_out
mkdir -p "$OUT"

# Build per-method symlink dirs (full 81 videos each)
python -c "
import os, glob
RUN='$RUN'
for m in ['baseline','MoSaiC']:
    d=os.path.join(RUN,'vb_'+m.lower()); os.makedirs(d, exist_ok=True)  # lowercase to match worker --mosaic_dir
    for v in glob.glob(os.path.join(RUN,'videos','*_'+m+'.mp4')):
        l=os.path.join(d, os.path.basename(v))
        if not os.path.lexists(l): os.symlink(os.path.abspath(v), l)
    print(m, len(glob.glob(os.path.join(d,'*.mp4'))), 'videos')
"

DIMS=(subject_consistency background_consistency temporal_flickering motion_smoothness dynamic_degree imaging_quality aesthetic_quality)
pids=(); g=0
for d in "${DIMS[@]}"; do
  CUDA_VISIBLE_DEVICES=$g python bench/realcamvid/vbench_worker.py \
      --dim "$d" --baseline_dir "$RUN/vb_baseline" --mosaic_dir "$RUN/vb_mosaic" --out "$OUT" \
      > "$OUT/worker_${d}.log" 2>&1 &
  pids+=($!); echo "[launch] dim=$d -> GPU $g (pid $!)"; g=$((g+1))
done
echo "[run_vbench] ${#pids[@]} dim-workers running; waiting..."
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[run_vbench] DONE failed=$fail"
