#!/bin/bash
# Batch MoCE-vs-baseline on converted RealCam-Vid clips. Args: clip_id [clip_id ...]
# Per clip: generate baseline + MoCE (2x H20, ulysses=2, equal KV budget), parse stdout
# timing, run semantic-quality comparison. Skips clips already done. Non-destructive.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbot
cd /DATA/YuanZhen/Lingbot/lingbot-world
RV=/DATA/YuanZhen/Lingbot/lingbot-world/bench/realcamvid
mkdir -p "$RV/videos" "$RV/logs" "$RV/stats" "$RV/eval"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ROI=center

COMMON=(--task i2v-A14B --size 480*832
        --ckpt_dir /DATA/YuanZhen/Lingbot/lingbot-world-base-cam
        --dit_fsdp --t5_fsdp --ulysses_size 8 --frame_num 481
        --convert_model_dtype --base_seed 42
        --sink_size 1 --max_attention_size 47000
        --save_dir "$RV/videos")

gen () { # clip_id tag extra_args...
  local clip="$1" tag="$2"; shift 2
  local cdir="$RV/clips/$clip"
  local vid="$RV/videos/${clip}_${tag}.mp4"
  local log="$RV/logs/${clip}_${tag}.log"
  if [[ -f "$vid" && -f "$RV/stats/${clip}_${tag}.json" ]]; then echo "[skip gen] $clip $tag"; return 0; fi
  echo "===== [$(date +%H:%M:%S)] GEN $clip $tag =====" | tee "$log"
  torchrun --nproc_per_node=8 --master_port=29533 generate_fast.py \
      "${COMMON[@]}" "$@" \
      --image "$cdir/image.jpg" --action_path "$cdir" \
      --save_file "$vid" \
      --prompt "$(cat "$cdir/prompt.txt")" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  python "$RV/../parse_summary.py" --log "$log" --config "$tag" --example "$clip" \
         --out "$RV/stats/${clip}_${tag}.json" >/dev/null 2>&1 || true
  echo "===== [$(date +%H:%M:%S)] DONE $clip $tag rc=$rc =====" | tee -a "$log"
  return $rc
}

for clip in "$@"; do
  echo "######## CLIP $clip ########"
  gen "$clip" baseline --local_attn_size 30
  gen "$clip" MoCE --enable_motion_adaptive_kv_eviction \
      --ma_kv_recent_window 4 --ma_kv_keep_ratio 0.5 --ma_kv_min_keep_chunks 2 \
      --ma_kv_latent_rescue --ma_kv_latent_rescue_thr 0.08
  # quality eval (MoCE vs baseline)
  if [[ -f "$RV/videos/${clip}_MoCE.mp4" && -f "$RV/videos/${clip}_baseline.mp4" && ! -f "$RV/eval/$clip/semantics_comparison.json" ]]; then
    echo "---- eval $clip ----"
    python eval_semantics.py \
        --video "$RV/videos/${clip}_MoCE.mp4" "$RV/videos/${clip}_baseline.mp4" \
        --prompt_file "$RV/clips/$clip/prompt.txt" --roi $ROI \
        --dino_model /DATA/YuanZhen/Lingbot/dinov2-base \
        --clip_model /DATA/YuanZhen/Lingbot/clip-vit-base-patch32 \
        --output_dir "$RV/eval/$clip" 2>&1 | tee "$RV/logs/${clip}_eval.log"
  fi
done
echo "DONE_BATCH $@"
