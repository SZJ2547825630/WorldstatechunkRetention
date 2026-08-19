#!/bin/bash
# Smoke test: validate the full benchmark pipeline (generate baseline + MoCE + eval)
# on example 00, on this 2x H20 box. Non-destructive: writes only under bench/.
#
# Config mirrors MoSaiC.md documented setup: 2 GPUs, ulysses_size=2, max_attention_size=47000,
# sink_size=1. Baseline uses local_attn_size=30 (sliding window, ~equal token budget).

set -uo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbot

cd /DATA/YuanZhen/Lingbot/lingbot-world
BENCH=/DATA/YuanZhen/Lingbot/lingbot-world/bench
mkdir -p "$BENCH/videos" "$BENCH/logs" "$BENCH/stats" "$BENCH/eval"

export CUDA_VISIBLE_DEVICES=0,1
EX=00
PROMPT="$(cat examples/$EX/prompt.txt)"

COMMON=(--task i2v-A14B --size 480*832
        --ckpt_dir /DATA/YuanZhen/Lingbot/lingbot-world-base-cam
        --image examples/$EX/image.jpg --action_path examples/$EX
        --dit_fsdp --t5_fsdp --ulysses_size 2 --frame_num 481
        --convert_model_dtype --base_seed 42
        --sink_size 1 --max_attention_size 47000
        --save_dir "$BENCH/videos")

run_gen () {
  local tag="$1"; shift
  local log="$BENCH/logs/example${EX}_${tag}.log"
  echo "===== [$(date +%H:%M:%S)] GEN START $tag =====" | tee "$log"
  local t0=$SECONDS
  torchrun --nproc_per_node=2 --master_port=29531 generate_fast.py \
      "${COMMON[@]}" "$@" \
      --save_file "$BENCH/videos/example${EX}_${tag}.mp4" \
      --prompt "$PROMPT" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  echo "===== [$(date +%H:%M:%S)] GEN END $tag rc=$rc wall=$((SECONDS-t0))s =====" | tee -a "$log"
  return $rc
}

echo "######## 1/4 baseline generation ########"
run_gen baseline --local_attn_size 30
RC_BASE=$?

echo "######## 2/4 MoCE generation ########"
run_gen MoCE --enable_motion_adaptive_kv_eviction \
        --ma_kv_recent_window 4 --ma_kv_keep_ratio 0.5 --ma_kv_min_keep_chunks 2 \
        --ma_kv_latent_rescue --ma_kv_latent_rescue_thr 0.08
RC_MOCE=$?

echo "######## 3/4 parse timing stats ########"
python "$BENCH/parse_summary.py" --log "$BENCH/logs/example${EX}_baseline.log" \
       --config baseline --example $EX --out "$BENCH/stats/example${EX}_baseline.json" || true
python "$BENCH/parse_summary.py" --log "$BENCH/logs/example${EX}_MoCE.log" \
       --config MoCE --example $EX --out "$BENCH/stats/example${EX}_MoCE.json" || true

echo "######## 4/4 semantic quality eval (MoCE vs baseline) ########"
EVAL_LOG="$BENCH/logs/example${EX}_eval.log"
if [[ -f "$BENCH/videos/example${EX}_MoCE.mp4" && -f "$BENCH/videos/example${EX}_baseline.mp4" ]]; then
  python eval_semantics.py \
      --video "$BENCH/videos/example${EX}_MoCE.mp4" "$BENCH/videos/example${EX}_baseline.mp4" \
      --prompt_file examples/$EX/prompt.txt \
      --roi subject \
      --dino_model /DATA/YuanZhen/Lingbot/dinov2-base \
      --clip_model /DATA/YuanZhen/Lingbot/clip-vit-base-patch32 \
      --output_dir "$BENCH/eval/example${EX}" 2>&1 | tee "$EVAL_LOG"
  RC_EVAL=${PIPESTATUS[0]}
else
  echo "SKIP eval: one or both videos missing" | tee "$EVAL_LOG"
  RC_EVAL=1
fi

echo "================ SMOKE STATUS ================"
echo "baseline_gen_rc=$RC_BASE  moce_gen_rc=$RC_MOCE  eval_rc=$RC_EVAL"
ls -la "$BENCH/videos/" "$BENCH/stats/" "$BENCH/eval/example${EX}/" 2>/dev/null
echo "DONE_SMOKE"
