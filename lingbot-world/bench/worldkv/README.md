# Official WorldKV integration

Checkout: `third_party/WorldKV` → `/DATA/YuanZhen/WorldKV` (cvlab-kaist/WorldKV).

## Generate (8×H20)

```bash
source env.sh
cd "$LINGBOT_WORLD"

# default_loop (24 clips), write into existing comparison out dir
REALCAMVID_SUBSET=default_loop \
REALCAMVID_OUT=output/realcamvid_ws_vs_window_default_all_v2 \
  bash bench/realcamvid/run_worldkv_official.sh
```

Official flags (README defaults): `--use_retrieval --retrieval_frames 18 --kv_bank_on_gpu --kv_compression_enable --kv_compression_keep_ratio 0.5 --kv_compression_at_store --kv_compression_pooled`.

Checkpoint must contain `cam` in the path (`CKPT_CAM`, default `lingbot-world-base-cam`).

## Evaluate (default protocol)

WorldKV revisit protocol is the **default** quality metric:

```bash
REALCAMVID_OUT=output/realcamvid_ws_vs_window_default_all_v2 \
REALCAMVID_SUBSET=default_loop \
METHODS=window,world_state_cr,worldkv \
  bash bench/realcamvid/run_worldkv_eval.sh
```
