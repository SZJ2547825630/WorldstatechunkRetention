"""CPU tests for Future Coverage Oracle (P0)."""

from __future__ import annotations

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wan.utils.future_use_labels import (
    LABEL_TYPE_FUTURE_USE,
    aggregate_future_use_records,
    convert_oracle_payload,
    future_use_utility,
    pose_reuse,
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


class TestFutureUseLabels(unittest.TestCase):
    def test_pose_reuse_higher_for_near(self):
        near = pose_reuse(_pose((0.0, 0.0, 0.0)), _pose((0.1, 0.0, 0.0)), translation_scale=1.0)
        far = pose_reuse(_pose((0.0, 0.0, 0.0)), _pose((8.0, 0.0, 0.0), yaw_deg=90.0),
                         translation_scale=1.0)
        self.assertGreater(near, far)

    def test_future_use_prefers_later_revisit(self):
        # Candidate 0 is unused now but heavily attended + revisited in the future.
        # Candidate 1 is hot at t but unused later.
        mass = {
            (5, 0): 0.05, (5, 1): 0.40,
            (6, 0): 0.10, (6, 1): 0.05,
            (7, 0): 0.50, (7, 1): 0.02,
            (8, 0): 0.55, (8, 1): 0.01,
        }
        meta = {
            0: {"chunk_id": 0, "pose": _pose((0.0, 0.0, 0.0)), "camera_forward": [0, 0, 1]},
            1: {"chunk_id": 1, "pose": _pose((3.0, 0.0, 0.0), 90), "camera_forward": [-1, 0, 0]},
            5: {"chunk_id": 5, "pose": _pose((3.0, 0.0, 0.0), 90), "camera_forward": [-1, 0, 0]},
            6: {"chunk_id": 6, "pose": _pose((1.5, 0.0, 0.0), 45), "camera_forward": [-0.7, 0, 0.7]},
            7: {"chunk_id": 7, "pose": _pose((0.2, 0.0, 0.0)), "camera_forward": [0, 0, 1]},
            8: {"chunk_id": 8, "pose": _pose((0.0, 0.0, 0.0)), "camera_forward": [0, 0, 1]},
        }
        y0 = future_use_utility(mass, meta, t=5, seg_id=0, horizon=3, gamma=0.9, alpha=0.5)
        y1 = future_use_utility(mass, meta, t=5, seg_id=1, horizon=3, gamma=0.9, alpha=0.5)
        self.assertGreater(y0, y1)

    def test_convert_payload_sets_label_type(self):
        records = [
            {"gen_chunk_id": h, "seg_id": s, "mass": 0.2 + 0.01 * s, "n_layers": 1}
            for h in range(1, 10) for s in range(0, h)
        ]
        meta = {
            i: {
                "chunk_id": i,
                "pose": _pose((float(i) * 0.1, 0.0, 0.0)),
                "camera_forward": [0.0, 0.0, 1.0],
                "motion_score": 0.3,
            }
            for i in range(10)
        }
        payload = {
            "records": records,
            "chunk_meta": meta,
            "config": {
                "sink_chunk_count": 1,
                "recent_window": 1,
                "translation_scale": 0.1,
                "label_type": "attention_mass",
            },
        }
        out = convert_oracle_payload(payload, horizon=4, gamma=0.9, alpha=0.5)
        self.assertEqual(out["config"]["label_type"], LABEL_TYPE_FUTURE_USE)
        self.assertGreater(len(out["records"]), 0)
        # Decision times should exclude sink/recent-only rows.
        for r in out["records"]:
            self.assertGreaterEqual(r["gen_chunk_id"] - r["seg_id"], 2)
            self.assertGreaterEqual(r["seg_id"], 1)

    def test_aggregate_empty_safe(self):
        self.assertEqual(aggregate_future_use_records([], {}), [])


if __name__ == "__main__":
    unittest.main()
