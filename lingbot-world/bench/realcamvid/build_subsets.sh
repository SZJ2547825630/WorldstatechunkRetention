#!/usr/bin/env bash
# Materialize default RealCam-Vid test subsets as symlink clip dirs.
# Source clips live in the archive tree (large); this script only links them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ARCH="${REALCAMVID_ARCHIVE:-$ROOT/artifacts/archive_pre_paper/bench/realcamvid}"
SUBSETS="$SCRIPT_DIR/subsets"

if [[ ! -d "$ARCH/clips" ]]; then
  echo "Archive not found: $ARCH/clips" >&2
  echo "Set REALCAMVID_ARCHIVE to the converted RealCam-Vid tree." >&2
  exit 1
fi

link_subset() {
  local name="$1" list="$2" src_pool="$3"
  local dest="$SCRIPT_DIR/clips_${name}"
  rm -rf "$dest"
  mkdir -p "$dest"
  local n=0
  while IFS= read -r clip || [[ -n "$clip" ]]; do
    [[ -z "$clip" || "$clip" =~ ^# ]] && continue
    local src=""
    # Prefer longer loop-closure poses when available.
    for pool in clips_loop "$src_pool" clips clips_long; do
      if [[ -d "$ARCH/$pool/$clip" ]]; then
        src="$ARCH/$pool/$clip"
        break
      fi
    done
    if [[ -z "$src" || ! -f "$src/image.jpg" || ! -f "$src/poses.npy" ]]; then
      echo "[warn] skip missing clip: $clip" >&2
      continue
    fi
    ln -s "$src" "$dest/$clip"
    n=$((n + 1))
  done < "$list"
  echo "[$name] linked $n clips -> $dest"
}

link_subset "default_loop" "$SUBSETS/default_loop.txt" "clips_loop"
link_subset "default_random" "$SUBSETS/default_random.txt" "clips"
link_subset "default_all" "$SUBSETS/default_all.txt" "clips"

echo "Done. Use clips_dir=$SCRIPT_DIR/clips_default_loop (or _random / _all)."
