"""Default paths for Chunk Retention selector checkpoints."""

from __future__ import annotations

import os

# Default World-State CR = future-use selector (P0+P1 / world_state.v2).
# Frozen attention-mass ablation remains selector_ws_v1.pt (schema world_state).
WORLD_STATE_FUTURE_SELECTOR_CKPT_NAME = "selector_ws_future_v1.pt"
WORLD_STATE_SELECTOR_CKPT_NAME = WORLD_STATE_FUTURE_SELECTOR_CKPT_NAME
WORLD_STATE_V1_SELECTOR_CKPT_NAME = "selector_ws_v1.pt"
LEARNED_SELECTOR_CKPT_NAME = "selector_all4.pt"

# Back-compat aliases
SELECTOR_CKPT_NAME = WORLD_STATE_SELECTOR_CKPT_NAME
LEGACY_SELECTOR_CKPT_NAME = LEARNED_SELECTOR_CKPT_NAME


def _package_root() -> str:
    # wan/utils/selector_defaults.py -> package root (lingbot-world)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def selector_ckpt_candidates(schema: str = "world_state_future") -> list[str]:
    root = _package_root()
    if schema == "learned":
        names = [
            os.path.join(root, "assets", "selectors", LEARNED_SELECTOR_CKPT_NAME),
            os.path.join(root, "..", "lingbot-world", "assets", "selectors", LEARNED_SELECTOR_CKPT_NAME),
        ]
    elif schema in ("world_state", "world_state_v1", "attention_mass"):
        # Frozen v1 ablation (attention-mass oracle / world_state.v1).
        names = [
            os.path.join(root, "assets", "selectors", WORLD_STATE_V1_SELECTOR_CKPT_NAME),
            os.path.join(root, "..", "lingbot-world", "assets", "selectors",
                         WORLD_STATE_V1_SELECTOR_CKPT_NAME),
        ]
    else:
        # Default: future-use (also accepts schema aliases).
        names = [
            os.path.join(root, "assets", "selectors", WORLD_STATE_FUTURE_SELECTOR_CKPT_NAME),
            os.path.join(root, "..", "lingbot-world", "assets", "selectors",
                         WORLD_STATE_FUTURE_SELECTOR_CKPT_NAME),
        ]
    return [os.path.normpath(p) for p in names]


def default_selector_ckpt(schema: str = "world_state_future") -> str:
    for path in selector_ckpt_candidates(schema=schema):
        if os.path.isfile(path):
            return path
    return selector_ckpt_candidates(schema=schema)[0]


def resolve_selector_ckpt(explicit: str | None, schema: str = "world_state_future") -> str | None:
    if explicit:
        return explicit
    return default_selector_ckpt(schema=schema)
