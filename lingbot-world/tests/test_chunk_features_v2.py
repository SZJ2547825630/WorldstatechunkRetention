"""CPU tests for world_state.v2 (P1) feature formulas."""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wan.modules.chunk_selector import (
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_V2,
    ChunkSelector,
    build_chunk_features,
    build_chunk_features_v2,
    features_for_schema,
    load_selector,
    save_selector,
    FEATURE_NAMES,
)


def _pose(t, yaw_deg=0.0):
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return {
        "translation": list(t),
        "rotation": [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        "intrinsics": [500.0, 500.0, 320.0, 240.0],
        "camera_forward": [-s, 0.0, c],
    }


def _cand(cid, t, last_obs, yaw=0.0):
    return {
        "chunk_id": cid,
        "motion_score": 0.4,
        "camera_forward": _pose(t, yaw)["camera_forward"],
        "k_centroid": torch.ones(8),
        "value_norm": 1.0,
        "pose": _pose(t, yaw),
        "revisit_count": 0,
        "last_observed": last_obs,
    }


class TestFeaturesV2(unittest.TestCase):
    def test_v1_unchanged_by_v2_helper(self):
        cur = _cand(20, (0.0, 0.0, 0.0), 20)
        cand = _cand(2, (2.0, 0.0, 0.0), 2)
        r1 = build_chunk_features(cand, cur, 20, 0.5, 1.0, 1.0)
        r1b = features_for_schema(
            FEATURE_NAMES, cand, cur, 20, 0.5, 1.0, 1.0,
            schema_version=FEATURE_SCHEMA_VERSION)
        self.assertEqual(r1, r1b)

    def test_v2_age_stable_vs_rollout_length(self):
        # Same age gap=20: at t=40 and t=400, v1 /now differs a lot; v2 tanh does not.
        cand_early = _cand(20, (0.0, 0.0, 0.0), 20)
        cand_late = _cand(380, (0.0, 0.0, 0.0), 380)
        cur40 = _cand(40, (0.0, 0.0, 0.0), 40)
        cur400 = _cand(400, (0.0, 0.0, 0.0), 400)
        v1_early = build_chunk_features(cand_early, cur40, 40, 0.5, 1.0, 1.0)[-1]
        v1_late = build_chunk_features(cand_late, cur400, 400, 0.5, 1.0, 1.0)[-1]
        v2_early = build_chunk_features_v2(cand_early, cur40, 40, 0.5, 1.0, 1.0)[-1]
        v2_late = build_chunk_features_v2(cand_late, cur400, 400, 0.5, 1.0, 1.0)[-1]
        self.assertGreater(abs(v1_early - v1_late), 0.2)
        self.assertAlmostEqual(v2_early, v2_late, places=5)

    def test_v2_reachability_grows_with_time_budget(self):
        cur = _cand(10, (0.0, 0.0, 0.0), 10)
        near_recent = _cand(8, (1.0, 0.0, 0.0), 8)   # Δt=2
        near_old = _cand(1, (1.0, 0.0, 0.0), 1)       # Δt=9, same d
        # Index 6 == reachability
        r_recent = build_chunk_features_v2(near_recent, cur, 10, 0.5, 1.0, 1.0)[6]
        r_old = build_chunk_features_v2(near_old, cur, 10, 0.5, 1.0, 1.0)[6]
        self.assertGreater(r_old, r_recent)

    def test_save_load_v2_roundtrip(self):
        model = ChunkSelector(in_dim=11, hidden=16)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sel.pt")
            save_selector(
                model, path,
                meta={"label_type": "future_use_v1"},
                feature_names=FEATURE_NAMES,
                schema_version=FEATURE_SCHEMA_VERSION_V2,
            )
            loaded = load_selector(path)
            self.assertEqual(loaded._schema_version, FEATURE_SCHEMA_VERSION_V2)


if __name__ == "__main__":
    unittest.main()
