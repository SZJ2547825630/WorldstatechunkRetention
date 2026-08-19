#!/usr/bin/env bash
# Shared preamble for bench batch scripts.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_BENCH_DIR="$(cd "$_SCRIPT_DIR/.." && pwd)"
# shellcheck source=../../../env.sh
source "$_BENCH_DIR/../../env.sh"

export BENCH_DIR="$_BENCH_DIR"
export PROJECT_DIR="$LINGBOT_WORLD"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate lingbot
fi

cd "$PROJECT_DIR"
