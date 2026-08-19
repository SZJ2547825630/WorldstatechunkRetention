#!/usr/bin/env bash
# World-State CR vs Window on official RealCam-Vid default subsets (8×H20).
# Layout: N=4 data-parallel groups × U=2 sequence-parallel GPUs.
# Methods run in separate launches so peak memory is not cross-contaminated.
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
source /DATA/miniconda3_corl/etc/profile.d/conda.sh 2>/dev/null || true
conda activate lingbot
source /DATA/YuanZhen/Lingbot/env.sh
cd /DATA/YuanZhen/Lingbot/lingbot-world

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SUBSET="${REALCAMVID_SUBSET:-default_all}"
OUT="${REALCAMVID_OUT:-output/realcamvid_ws_vs_window_${SUBSET}}"
N="${N:-4}"
U="${U:-2}"
FRAME_NUM="${FRAME_NUM:-481}"
METHODS_SEQ="${METHODS_SEQ:-window world_state_cr}"

bash bench/realcamvid/build_subsets.sh
mkdir -p "$OUT/logs"

run_method() {
  local method="$1"
  local port_base="$2"
  echo "[$(date +%H:%M:%S)] ==== START method=$method N=$N U=$U subset=$SUBSET ===="
  local pids=()
  for i in $(seq 0 $((N - 1))); do
    local lo=$((i * U))
    local gpus
    gpus=$(seq -s, "$lo" $((lo + U - 1)))
    CUDA_VISIBLE_DEVICES="$gpus" torchrun --nproc_per_node="$U" --master_port=$((port_base + i)) \
      bench/batch_generate.py \
      --ckpt_dir "$CKPT_DIR" \
      --subset "$SUBSET" \
      --out_dir "$OUT" \
      --methods "$method" \
      --ulysses_size "$U" \
      --shard "$i/$N" \
      --frame_num "$FRAME_NUM" \
      --max_attention_size 47000 \
      --base_seed 42 \
      > "$OUT/logs/${method}_g${i}.log" 2>&1 &
    pids+=($!)
    echo "[launch] $method group $i -> GPUs $gpus shard=$i/$N pid=$!"
    sleep 8
  done
  local fail=0
  for p in "${pids[@]}"; do
    wait "$p" || fail=$((fail + 1))
  done
  echo "[$(date +%H:%M:%S)] ==== DONE method=$method failed_groups=$fail ===="
  return "$fail"
}

port=29610
overall_fail=0
for m in $METHODS_SEQ; do
  run_method "$m" "$port" || overall_fail=$((overall_fail + 1))
  port=$((port + 20))
done

# Aggregate quick summary
python3 - <<PY
import json, glob, os, statistics as st
out = os.path.abspath("$OUT")
rows = []
for p in sorted(glob.glob(out + "/stats/*.json")):
    d = json.load(open(p))
    s = d.get("stats", {})
    name = os.path.basename(p)[:-5]  # drop .json
    if name.endswith("_world_state_cr"):
        method, clip = "world_state_cr", name[: -len("_world_state_cr")]
    elif name.endswith("_window"):
        method, clip = "window", name[: -len("_window")]
    else:
        continue
    rows.append({
        "clip": clip, "method": method,
        "t": s.get("total_generation_time_s"),
        "ctx": s.get("avg_attention_context_tokens"),
        "peak": s.get("peak_memory_allocated_gb"),
        "ret": s.get("retained_chunks"),
        "tot": s.get("total_chunks"),
        "schema": s.get("feature_schema"),
    })
summary = {"out": out, "n_stats": len(rows), "by_method": {}}
for m in ("window", "world_state_cr"):
    xs = [r for r in rows if r["method"] == m]
    def mean(key):
        vs = [r[key] for r in xs if r.get(key) is not None]
        return (sum(vs)/len(vs)) if vs else None
    summary["by_method"][m] = {
        "n": len(xs),
        "mean_time_s": mean("t"),
        "mean_ctx": mean("ctx"),
        "mean_peak_gb": mean("peak"),
        "mean_retained": mean("ret"),
    }
path = os.path.join(out, "runtime_summary.json")
json.dump(summary, open(path, "w"), indent=2)
print(json.dumps(summary, indent=2))
print("WROTE", path)
PY

echo "WS_VS_WINDOW_DONE fail=$overall_fail out=$OUT"
exit "$overall_fail"
