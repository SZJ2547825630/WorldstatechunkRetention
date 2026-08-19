#!/usr/bin/env bash
# Generate World-State CR (default = former world_state_cr_future, P0+P1).
# METHOD defaults to world_state_cr; alias world_state_cr_future still works.
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
source /DATA/miniconda3_corl/etc/profile.d/conda.sh 2>/dev/null || true
conda activate lingbot
source /DATA/YuanZhen/Lingbot/env.sh
cd /DATA/YuanZhen/Lingbot/lingbot-world

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SUBSET="${REALCAMVID_SUBSET:-default_all}"
OUT="${REALCAMVID_OUT:-output/realcamvid_ws_vs_window_default_all_v2}"
N="${N:-4}"
U="${U:-2}"
FRAME_NUM="${FRAME_NUM:-481}"
METHOD="${METHOD:-world_state_cr}"

test -f assets/selectors/selector_ws_future_v1.pt || {
  echo "missing assets/selectors/selector_ws_future_v1.pt — train first" >&2
  exit 1
}

bash bench/realcamvid/build_subsets.sh
mkdir -p "$OUT/logs"

echo "[$(date +%H:%M:%S)] START method=$METHOD N=$N U=$U subset=$SUBSET out=$OUT"
pids=()
port_base=29810
for i in $(seq 0 $((N - 1))); do
  lo=$((i * U))
  gpus=$(seq -s, "$lo" $((lo + U - 1)))
  CUDA_VISIBLE_DEVICES="$gpus" torchrun --nproc_per_node="$U" --master_port=$((port_base + i)) \
    bench/batch_generate.py \
    --ckpt_dir "$CKPT_DIR" \
    --subset "$SUBSET" \
    --out_dir "$OUT" \
    --methods "$METHOD" \
    --ulysses_size "$U" \
    --shard "$i/$N" \
    --frame_num "$FRAME_NUM" \
    --max_attention_size 47000 \
    --base_seed 42 \
    > "$OUT/logs/${METHOD}_g${i}.log" 2>&1 &
  pids+=($!)
  echo "[launch] $METHOD group $i -> GPUs $gpus shard=$i/$N pid=$!"
  sleep 8
done

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=$((fail + 1))
done
echo "[$(date +%H:%M:%S)] GEN DONE method=$METHOD failed_groups=$fail"

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  METHODS="${EVAL_METHODS:-window,world_state_cr}" \
  REALCAMVID_OUT="$OUT" REALCAMVID_SUBSET="$SUBSET" \
    bash bench/realcamvid/run_worldkv_eval.sh || echo "[warn] eval failed"
fi

echo "FUTURE_CR_DONE fail=$fail out=$OUT"
exit "$fail"
