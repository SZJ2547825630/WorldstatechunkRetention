#!/bin/bash
# Hybrid data-parallel x sequence-parallel launcher.
# Starts N worker groups, each a torchrun group of U GPUs (ulysses_size=U). Clips are sharded
# round-robin across the N groups. N*U must be <= visible GPUs.
#   N=4 U=2 -> 4 groups of 2 GPUs (pairs 0-1,2-3,4-5,6-7), 4-way data parallelism.
# The 14B model is too large for ulysses=1 (peak ~93GB/96GB), so U>=2 to shard model+KV.
#
# Config via env vars:
#   N (groups, default 4)  U (gpus/group, default 2)  METHODS (default baseline,MoSaiC)
#   OUT (out dir, default bench/realcamvid/run200)  CLIPS (default bench/realcamvid/clips)
# Extra args after `--` are forwarded to batch_generate.py (e.g. --limit 1).
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbot
cd /DATA/YuanZhen/Lingbot/lingbot-world
# Avoid allocator fragmentation OOM when reloading the 14B model per method in a
# multi-method run (the 4th method can OOM on a tiny alloc despite del pipe +
# empty_cache). Caller may override.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

N=${N:-4}
U=${U:-2}
GPU_BASE=${GPU_BASE:-0}   # first physical GPU to use (offset; lets a run share the box)
METHODS=${METHODS:-baseline,MoSaiC}
OUT=${OUT:-bench/realcamvid/run200}
CLIPS=${CLIPS:-bench/realcamvid/clips}
mkdir -p "$OUT/logs"

pids=()
for i in $(seq 0 $((N - 1))); do
  lo=$((GPU_BASE + i * U))
  gpus=$(seq -s, $lo $((lo + U - 1)))
  CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=$U --master_port=$((29550 + i)) \
      bench/batch_generate.py \
      --ckpt_dir /DATA/YuanZhen/Lingbot/lingbot-world-base-cam \
      --clips_dir "$CLIPS" --out_dir "$OUT" \
      --methods "$METHODS" --ulysses_size $U --shard "$i/$N" "$@" \
      > "$OUT/logs/g${i}.log" 2>&1 &
  pids+=($!)
  echo "[launch] group $i -> GPUs $gpus  (ulysses=$U, shard $i/$N)  pid $!  log=$OUT/logs/g${i}.log"
  sleep 10   # stagger model loads
done

echo "[run_dp2] N=$N U=$U methods=$METHODS out=$OUT ; waiting for ${#pids[@]} groups..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "[run_dp2] DONE  failed_groups=$fail"
