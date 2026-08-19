#!/usr/bin/env bash
# Shared preamble for lingbot-world inference scripts.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../env.sh
source "$_SCRIPT_DIR/../../env.sh"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate lingbot
fi

cd "$LINGBOT_WORLD"
