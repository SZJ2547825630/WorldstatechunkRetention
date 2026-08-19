#!/bin/bash
# MoSaiC (= MoCE + SWTP + trajectory-diversity) on the same RealCam-Vid clips that already
# have baseline + MoCE results. Generates MoSaiC video, parses timing, single-video quality eval.
# Skips done clips. Non-destructive.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbot
cd /DATA/YuanZhen/Lingbot/lingbot-world
RV=/DATA/YuanZhen/Lingbot/lingbot-world/bench/realcamvid
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ROI=center

COMMON=(--task i2v-A14B --size 480*832
        --ckpt_dir /DATA/YuanZhen/Lingbot/lingbot-world-base-cam
        --dit_fsdp --t5_fsdp --ulysses_size 8 --frame_num 481
        --convert_model_dtype --base_seed 42
        --sink_size 1 --max_attention_size 47000
        --save_dir "$RV/videos")

# clips = those with an existing baseline result (parity with MoCE-vs-baseline run)
clips=$(ls "$RV"/stats/*_baseline.json 2>/dev/null | xargs -n1 basename | sed 's/_baseline.json//')

for clip in $clips; do
  echo "######## MoSaiC CLIP $clip ########"
  cdir="$RV/clips/$clip"
  vid="$RV/videos/${clip}_MoSaiC.mp4"
  log="$RV/logs/${clip}_MoSaiC.log"
  if [[ -f "$vid" && -f "$RV/stats/${clip}_MoSaiC.json" ]]; then
    echo "[skip gen] $clip MoSaiC"
  else
    echo "===== [$(date +%H:%M:%S)] GEN $clip MoSaiC =====" | tee "$log"
    torchrun --nproc_per_node=8 --master_port=29535 generate_fast.py \
        "${COMMON[@]}" \
        --enable_motion_adaptive_kv_eviction \
        --ma_kv_recent_window 4 --ma_kv_keep_ratio 0.5 --ma_kv_min_keep_chunks 2 \
        --ma_kv_latent_rescue --ma_kv_latent_rescue_thr 0.08 \
        --enable_swtp --swtp_keep_ratio 0.5 --swtp_num_summary 64 --swtp_min_saliency_gini 0.20 \
        --archive_diversity_pool 4 \
        --image "$cdir/image.jpg" --action_path "$cdir" \
        --save_file "$vid" \
        --prompt "$(cat "$cdir/prompt.txt")" 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    python "$RV/../parse_summary.py" --log "$log" --config MoSaiC --example "$clip" \
           --out "$RV/stats/${clip}_MoSaiC.json" >/dev/null 2>&1 || true
    echo "===== [$(date +%H:%M:%S)] DONE $clip MoSaiC rc=$rc =====" | tee -a "$log"
  fi
  # single-video quality eval (absolute metrics; aggregator compares vs baseline & MoCE)
  if [[ -f "$vid" && ! -f "$RV/eval/$clip/mosaic_eval/semantics_eval.json" ]]; then
    echo "---- eval MoSaiC $clip ----"
    python eval_semantics.py \
        --video "$vid" \
        --prompt_file "$cdir/prompt.txt" --roi $ROI \
        --dino_model /DATA/YuanZhen/Lingbot/dinov2-base \
        --clip_model /DATA/YuanZhen/Lingbot/clip-vit-base-patch32 \
        --output_dir "$RV/eval/$clip/mosaic_eval" 2>&1 | tee "$RV/logs/${clip}_mosaic_eval.log"
  fi
done
echo "DONE_MOSAIC $(date +%H:%M:%S)"
