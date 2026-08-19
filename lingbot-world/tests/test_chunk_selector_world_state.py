"""CPU tests for World-State CR ChunkSelector feature schema."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wan.modules.chunk_selector import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    ChunkSelector,
    build_chunk_features,
    frustum_overlap_proxy,
    load_selector,
    rotation_geodesic,
    save_selector,
    se3_distance,
)


def _pose(t, yaw_deg=0.0, fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    import math
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    R = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    return {
        "translation": list(t),
        "rotation": R,
        "intrinsics": [fx, fy, cx, cy],
        "camera_forward": [-s, 0.0, c],
    }


class TestWorldStateFeatures(unittest.TestCase):
    def test_schema(self):
        self.assertEqual(FEATURE_SCHEMA_VERSION, "world_state.v1")
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIM)
        self.assertEqual(FEATURE_NAMES[0], "motion_norm")
        self.assertIn("frustum_overlap_proxy", FEATURE_NAMES)
        self.assertIn("reachability", FEATURE_NAMES)

    def test_build_row_length_and_pose_sensitivity(self):
        cur = {
            "chunk_id": 5,
            "motion_score": 0.4,
            "camera_forward": [0.0, 0.0, 1.0],
            "k_centroid": torch.ones(8),
            "value_norm": 1.0,
            "pose": _pose((0.0, 0.0, 0.0)),
            "revisit_count": 0,
            "last_observed": 5,
        }
        near = {
            "chunk_id": 1,
            "motion_score": 0.5,
            "camera_forward": [0.0, 0.0, 1.0],
            "k_centroid": torch.ones(8),
            "value_norm": 1.2,
            "pose": _pose((0.1, 0.0, 0.0)),
            "revisit_count": 2,
            "last_observed": 4,
        }
        far = {
            **near,
            "chunk_id": 0,
            "pose": _pose((5.0, 0.0, 0.0), yaw_deg=90.0),
            "camera_forward": [-1.0, 0.0, 0.0],
            "last_observed": 0,
            "revisit_count": 0,
        }
        row_near = build_chunk_features(near, cur, 5, motion_ref=0.5, vnorm_ref=1.0, translation_scale=1.0)
        row_far = build_chunk_features(far, cur, 5, motion_ref=0.5, vnorm_ref=1.0, translation_scale=1.0)
        self.assertEqual(len(row_near), FEATURE_DIM)
        self.assertEqual(len(row_far), FEATURE_DIM)
        ti = FEATURE_NAMES.index("translation_distance")
        fi = FEATURE_NAMES.index("frustum_overlap_proxy")
        self.assertLess(row_near[ti], row_far[ti])
        self.assertGreaterEqual(row_near[fi], row_far[fi])

    def test_geometry_helpers(self):
        a = _pose((0.0, 0.0, 0.0))
        b = _pose((1.0, 0.0, 0.0), yaw_deg=30.0)
        self.assertGreater(se3_distance(a, b, translation_scale=1.0), 0.0)
        Ra = torch.tensor(a["rotation"], dtype=torch.float32)
        Rb = torch.tensor(b["rotation"], dtype=torch.float32)
        self.assertGreater(rotation_geodesic(Ra, Rb), 0.0)
        self.assertGreater(frustum_overlap_proxy(a, a, 1.0), 0.5)

    def test_save_load_roundtrip_and_schema_compat(self):
        from wan.modules.chunk_selector import LEGACY_FEATURE_SCHEMA_VERSION
        model = ChunkSelector()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sel.pt")
            save_selector(model, path, meta={"motion_ref": 0.5, "vnorm_ref": 1.0})
            loaded = load_selector(path)
            self.assertEqual(loaded.in_dim, FEATURE_DIM)
            self.assertEqual(loaded._schema_version, FEATURE_SCHEMA_VERSION)

            # Learned CR 5-D checkpoint (and old m1.legacy.v0 alias) still loads.
            legacy = {
                "state_dict": ChunkSelector(in_dim=5).state_dict(),
                "in_dim": 5,
                "feature_names": [
                    "motion_norm", "age_norm", "cam_angular", "kcent_affinity", "value_norm",
                ],
                "schema_version": "m1.legacy.v0",
                "meta": {},
            }
            legacy_path = os.path.join(td, "legacy.pt")
            torch.save(legacy, legacy_path)
            loaded_legacy = load_selector(legacy_path)
            self.assertEqual(loaded_legacy.in_dim, 5)
            self.assertEqual(loaded_legacy._schema_version, LEGACY_FEATURE_SCHEMA_VERSION)

            # Unknown feature names are rejected.
            bad = {
                "state_dict": ChunkSelector(in_dim=2).state_dict(),
                "in_dim": 2,
                "feature_names": ["foo", "bar"],
                "meta": {},
            }
            bad_path = os.path.join(td, "bad.pt")
            torch.save(bad, bad_path)
            with self.assertRaises(AssertionError):
                load_selector(bad_path)


if __name__ == "__main__":
    unittest.main()
