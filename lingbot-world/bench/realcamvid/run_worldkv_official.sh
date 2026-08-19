#!/usr/bin/env bash
# Official WorldKV on RealCam-Vid default subsets (8×H20).
# Layout: N=2 data-parallel groups × U=4 sequence-parallel GPUs
# (WorldKV README uses 4 GPUs + kv_bank_on_gpu; U=2 OOMs on H20 at long horizon).
# Writes <clip>_worldkv.{mp4,json} into OUT (can share dir with window / world_state_cr).
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
if [[ "${CONDA_DEFAULT_ENV:-}" != "lingbot" ]]; then
  conda activate lingbot 2>/dev/null || true
fi
source /DATA/YuanZhen/Lingbot/env.sh
cd /DATA/YuanZhen/Lingbot/lingbot-world

# Avoid broken local proxy for torch hub / downloads
if [[ "${KEEP_PROXY:-0}" != "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WORLDKV_ROOT="${WORLDKV_ROOT:-/DATA/YuanZhen/Lingbot/third_party/WorldKV}"
# WorldKV detects control_type via 'cam' in path
CKPT_CAM="${CKPT_CAM:-/DATA/YuanZhen/Lingbot/lingbot-world-base-cam}"

# Default battery: all 64 clips (24 loop + 40 random); skip already-finished videos
SUBSET="${REALCAMVID_SUBSET:-default_all}"
OUT="${REALCAMVID_OUT:-output/realcamvid_ws_vs_window_default_all_v2}"
N="${N:-2}"
U="${U:-4}"
# Set NO_KV_BANK_ON_GPU=1 to store retrieval bank on CPU (slower, less VRAM)
FRAME_NUM="${FRAME_NUM:-481}"
RETRIEVAL_FRAMES="${RETRIEVAL_FRAMES:-18}"
LIMIT="${LIMIT:-0}"
EVAL_METHODS="${EVAL_METHODS:-window,world_state_cr,worldkv}"

bash bench/realcamvid/build_subsets.sh
mkdir -p "$OUT/logs"

echo "[$(date +%H:%M:%S)] WorldKV official: subset=$SUBSET N=$N U=$U out=$OUT ckpt=$CKPT_CAM"
echo "  WORLDKV_ROOT=$WORLDKV_ROOT retrieval_frames=$RETRIEVAL_FRAMES"

pids=()
port_base=29710
for i in $(seq 0 $((N - 1))); do
  lo=$((i * U))
  gpus=$(seq -s, "$lo" $((lo + U - 1)))
  extra=()
  if [[ "${NO_KV_BANK_ON_GPU:-0}" == "1" ]]; then
    extra+=(--no_kv_bank_on_gpu)
  fi
  # Only pass --limit when >0 (bash ${LIMIT:+} treats 0 as set)
  if [[ "${LIMIT}" -gt 0 ]]; then
    extra+=(--limit "$LIMIT")
  fi
  CUDA_VISIBLE_DEVICES="$gpus" torchrun --nproc_per_node="$U" --master_port=$((port_base + i)) \
    bench/worldkv/batch_generate.py \
    --ckpt_dir "$CKPT_CAM" \
    --subset "$SUBSET" \
    --out_dir "$OUT" \
    --ulysses_size "$U" \
    --shard "$i/$N" \
    --frame_num "$FRAME_NUM" \
    --retrieval_frames "$RETRIEVAL_FRAMES" \
    --base_seed 42 \
    "${extra[@]}" \
    > "$OUT/logs/worldkv_g${i}.log" 2>&1 &
  pids+=($!)
  echo "[launch] worldkv group $i -> GPUs $gpus shard=$i/$N pid=$! log=$OUT/logs/worldkv_g${i}.log"
  sleep 8
done

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=$((fail + 1))
done
echo "[$(date +%H:%M:%S)] WorldKV generate DONE failed_groups=$fail"

# Aggregate runtime
python3 - <<PY
import json, glob, os
out = os.path.abspath("$OUT")
rows = []
for p in sorted(glob.glob(out + "/stats/*_worldkv.json")):
    d = json.load(open(p))
    s = d.get("stats", {})
    rows.append({
        "clip": d.get("example"),
        "t": s.get("total_generation_time_s"),
        "peak": s.get("peak_memory_allocated_gb"),
        "retr": s.get("retrieval_frames"),
    })
def mean(key):
    vs = [r[key] for r in rows if r.get(key) is not None]
    return (sum(vs)/len(vs)) if vs else None
summary = {
    "out": out, "n": len(rows),
    "mean_time_s": mean("t"),
    "mean_peak_gb": mean("peak"),
    "retrieval_frames": $RETRIEVAL_FRAMES,
}
path = os.path.join(out, "runtime_summary_worldkv.json")
json.dump(summary, open(path, "w"), indent=2)
print(json.dumps(summary, indent=2))
print("WROTE", path)
PY

# Default protocol: WorldKV memory eval (window + world_state_cr + worldkv if present)
METHODS="${EVAL_METHODS:-window,world_state_cr,worldkv}"
if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  echo "[$(date +%H:%M:%S)] Running default WorldKV-protocol eval methods=$METHODS"
  REALCAMVID_OUT="$OUT" REALCAMVID_SUBSET="$SUBSET" METHODS="$METHODS" \
    bash bench/realcamvid/run_worldkv_eval.sh || echo "[warn] eval failed (videos may still be incomplete)"
fi

echo "WORLDKV_OFFICIAL_DONE fail=$fail out=$OUT"
exit "$fail"
