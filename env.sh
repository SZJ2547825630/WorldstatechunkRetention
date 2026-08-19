#!/usr/bin/env bash
# Source from repo root:  source env.sh
#
# Canonical paths. Sole editable source tree = $LINGBOT_WORLD (lingbot-world/).
# Override any variable before sourcing if your layout differs.

if [[ -z "${LINGBOT_ROOT:-}" ]]; then
  if [[ -n "${BASH_SOURCE[0]:-}" && "${BASH_SOURCE[0]}" != "${0:-}" ]]; then
    _env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  else
    _env_dir="$(pwd)"
  fi
  export LINGBOT_ROOT="$_env_dir"
  unset _env_dir
fi

# ★ Primary source tree (edit here)
export LINGBOT_WORLD="${LINGBOT_WORLD:-$LINGBOT_ROOT/lingbot-world}"

# Optional generated package (tools/pack_mosaic.sh); not the source of truth
export LINGBOT_MOSAIC="${LINGBOT_MOSAIC:-$LINGBOT_ROOT/lingbot-world-mosaic}"

# Optional WorldKV baseline — clone into third_party/WorldKV (see third_party/README.md)
export WORLDKV_ROOT="${WORLDKV_ROOT:-$LINGBOT_ROOT/third_party/WorldKV}"

# WorldKV control_type detection requires 'cam' in the checkpoint path
export CKPT_CAM="${CKPT_CAM:-$LINGBOT_ROOT/lingbot-world-base-cam}"

# Model & eval weights (see weights/README.md). Prefer exporting CKPT_DIR yourself.
export CKPT_DIR="${CKPT_DIR:-$LINGBOT_ROOT/weights/checkpoints/lingbot_world_fast}"
export DINO_DIR="${DINO_DIR:-$LINGBOT_ROOT/weights/eval/dinov2-base}"
export CLIP_DIR="${CLIP_DIR:-$LINGBOT_ROOT/weights/eval/clip-vit-base-patch32}"

# Local experiment dumps (gitignored)
export ARTIFACTS_DIR="${ARTIFACTS_DIR:-$LINGBOT_ROOT/artifacts}"
