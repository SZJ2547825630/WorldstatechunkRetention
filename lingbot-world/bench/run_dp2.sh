#!/bin/bash
# Hybrid data-parallel x sequence-parallel launcher.
# Starts N worker groups, each a torchrun group of U GPUs (ulysses_size=U). Clips are sharded
# round-robin across the N groups. N*U must be <= visible GPUs.
#   N=4 U=2 -> 4 groups of 2 GPUs (pairs 0-1,2-3,4-5,6-7), 4-way data parallelism.
# The 14B model is too large for ulysses=1 (peak ~93GB/96GB), so U>=2 to shard model+KV.
#
# Each METHOD is launched as its own wave of N torchrun groups (fresh processes).
# Do not pack multiple methods into one batch_generate process — FSDP teardown
# leaves GPU residue that falsely inflates the next method's peak_memory.
#
# Config via env vars:
#   N (groups, default 4)  U (gpus/group, default 2)  METHODS (default window,world_state_cr)
#   OUT (out dir, default bench/realcamvid/run200)  CLIPS (default bench/realcamvid/clips)
# Extra args after `--` are forwarded to batch_generate.py (e.g. --limit 1).
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbot
cd /DATA/YuanZhen/Lingbot/lingbot-world
# Expandable segments still helps within a long single-method run.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

N=${N:-4}
U=${U:-2}
GPU_BASE=${GPU_BASE:-0}   # first physical GPU to use (offset; lets a run share the box)
METHODS=${METHODS:-window,world_state_cr}
OUT=${OUT:-bench/realcamvid/run200}
CLIPS=${CLIPS:-bench/realcamvid/clips}
mkdir -p "$OUT/logs"

# One method per torchrun wave. Reloading FSDP/NCCL in the *same* process leaves
# ~15–20GB of unreclaimed CUDA state, which inflates the next method's
# peak_memory_allocated_gb (false "leak"). Fresh processes keep peaks comparable.
IFS=',' read -ra METHOD_ARR <<< "$METHODS"
fail=0
mi=0
for METHOD in "${METHOD_ARR[@]}"; do
  METHOD=$(echo "$METHOD" | xargs)
  [[ -z "$METHOD" ]] && continue
  echo "[run_dp2] ==== method=$METHOD ($((mi + 1))/${#METHOD_ARR[@]}) N=$N U=$U out=$OUT ===="
  pids=()
  for i in $(seq 0 $((N - 1))); do
    lo=$((GPU_BASE + i * U))
    gpus=$(seq -s, $lo $((lo + U - 1)))
    # Unique master_port per (method, group) to avoid TIME_WAIT collisions.
    port=$((29550 + mi * N + i))
    CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=$U --master_port=$port \
        bench/batch_generate.py \
        --ckpt_dir /DATA/YuanZhen/Lingbot/lingbot-world-base-cam \
        --clips_dir "$CLIPS" --out_dir "$OUT" \
        --methods "$METHOD" --ulysses_size $U --shard "$i/$N" "$@" \
        > "$OUT/logs/g${i}_${METHOD}.log" 2>&1 &
    pids+=($!)
    echo "[launch] method=$METHOD group $i -> GPUs $gpus  (ulysses=$U, shard $i/$N, port=$port)  pid $!  log=$OUT/logs/g${i}_${METHOD}.log"
    sleep 10   # stagger model loads
  done
  echo "[run_dp2] method=$METHOD waiting for ${#pids[@]} groups..."
  for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
  mi=$((mi + 1))
done
echo "[run_dp2] DONE  methods=$METHODS  failed_groups=$fail"
