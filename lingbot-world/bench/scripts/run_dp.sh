#!/bin/bash
# Data-parallel RealCam-Vid generation: N workers, 1 GPU each, clips sharded round-robin.
# Each worker loads the FULL model on its single GPU (ulysses_size=1, no FSDP) and processes
# clips[i::N]. This maximizes throughput for many independent short videos (DP scales ~linearly,
# unlike Ulysses sequence-parallel which pays all-to-all comm per layer).
#
# Usage:  bash bench/run_dp.sh [N_GPUS] [METHODS] [extra batch_generate args...]
#   bash bench/run_dp.sh 8 baseline,MoSaiC
#   bash bench/run_dp.sh 1 baseline --limit 2          # single-worker memory test
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbot
cd /DATA/YuanZhen/Lingbot/lingbot-world

N=${1:-8}
METHODS=${2:-baseline,MoSaiC}
shift 2 2>/dev/null || true
RV=bench/realcamvid
mkdir -p "$RV/logs"

pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i python bench/batch_generate.py \
      --ckpt_dir /DATA/YuanZhen/Lingbot/lingbot-world-base-cam \
      --clips_dir "$RV/clips" --out_dir "$RV" \
      --methods "$METHODS" --ulysses_size 1 --shard "$i/$N" "$@" \
      > "$RV/logs/dp_worker_${i}.log" 2>&1 &
  pids+=($!)
  echo "[launch] worker $i -> GPU $i (pid $!)  log=$RV/logs/dp_worker_${i}.log"
  sleep 8   # stagger model loads to smooth the disk/RAM peak
done

echo "[run_dp] ${#pids[@]} workers running on $N GPUs; methods=$METHODS; waiting..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "[run_dp] DONE  failed_workers=$fail"
