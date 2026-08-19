#!/usr/bin/env bash
# Build a portable MoSaiC release zip from the main source tree.
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$TOOLS_DIR/.." && pwd)"
DEST="${1:-$ROOT/artifacts/archives/lingbot-world-mosaic.zip}"
STAGING="$(mktemp -d)"

trap 'rm -rf "$STAGING"' EXIT

PKG="$STAGING/lingbot-world-mosaic"
mkdir -p "$PKG"/{scripts,assets/selectors,examples,docs}

rsync -a \
  "$ROOT/lingbot-world/generate_fast.py" \
  "$ROOT/lingbot-world/train_selector.py" \
  "$ROOT/lingbot-world/requirements.txt" \
  "$ROOT/lingbot-world/pyproject.toml" \
  "$PKG/" 2>/dev/null || true

rsync -a --exclude '__pycache__' "$ROOT/lingbot-world/wan/" "$PKG/wan/"
rsync -a "$ROOT/lingbot-world/scripts/" "$PKG/scripts/"
rsync -a "$ROOT/lingbot-world/examples/" "$PKG/examples/" 2>/dev/null || true
cp "$ROOT/lingbot-world/assets/selectors/selector_ws_future_v1.pt" "$PKG/assets/selectors/" 2>/dev/null || true
cp "$ROOT/lingbot-world/assets/selectors/selector_ws_v1.pt" "$PKG/assets/selectors/" 2>/dev/null || true
cp "$ROOT/lingbot-world/assets/selectors/selector_all4.pt" "$PKG/assets/selectors/" 2>/dev/null || true
cp "$ROOT/lingbot-world/README.md" "$PKG/"
cp "$ROOT/docs/method/"*.md "$PKG/docs/" 2>/dev/null || true

cat > "$PKG/env.sh" <<'ENVEOF'
#!/usr/bin/env bash
export CKPT_DIR="${CKPT_DIR:-/path/to/lingbot-world-base-cam}"
export LINGBOT_WORLD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVEOF

mkdir -p "$(dirname "$DEST")"
( cd "$STAGING" && zip -rq "$DEST" lingbot-world-mosaic )
echo "Packed: $DEST"
