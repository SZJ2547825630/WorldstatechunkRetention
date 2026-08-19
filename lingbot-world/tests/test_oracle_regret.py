"""CPU smoke for exclude_chunk_ids + attention-output regret oracle helpers."""

from __future__ import annotations

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wan.modules.model_fast import (
    build_dynamic_kv_tensors,
    oracle_reset,
    oracle_set,
    oracle_state,
    _record_oracle_mass,
    _attention_output_from_kv,
)


class TestExcludeChunkIds(unittest.TestCase):
    def test_exclude_drops_segment(self):
        segs = [
            {"chunk_id": 0, "token_count": 4, "is_sink": True,
             "k": torch.randn(1, 4, 2, 8), "v": torch.randn(1, 4, 2, 8)},
            {"chunk_id": 1, "token_count": 4, "is_sink": False,
             "k": torch.randn(1, 4, 2, 8), "v": torch.randn(1, 4, 2, 8)},
            {"chunk_id": 2, "token_count": 4, "is_sink": False,
             "k": torch.randn(1, 4, 2, 8), "v": torch.randn(1, 4, 2, 8)},
        ]
        kv = {"segments": segs, "layer_idx": 0, "current_chunk_id": 3}
        cur_k = torch.randn(1, 2, 2, 8)
        cur_v = torch.randn(1, 2, 2, 8)
        k_full, v_full = build_dynamic_kv_tensors(kv, cur_k, cur_v, max_attention_size=None)
        k_drop, v_drop = build_dynamic_kv_tensors(
            kv, cur_k, cur_v, max_attention_size=None, exclude_chunk_ids={1})
        self.assertEqual(k_full.shape[1], 4 + 4 + 4 + 2)
        self.assertEqual(k_drop.shape[1], 4 + 4 + 2)
        self.assertEqual(v_drop.shape, k_drop.shape)


class TestAttnOutputRegret(unittest.TestCase):
    def test_regret_nonconstant_and_records(self):
        torch.manual_seed(0)
        segs = []
        for cid in range(3):
            # Make segment 1 uniquely aligned with the query so dropping it hurts more.
            scale = 10.0 if cid == 1 else 0.1
            k = torch.randn(1, 4, 2, 8) * scale
            v = torch.randn(1, 4, 2, 8)
            segs.append({
                "chunk_id": cid, "token_count": 4,
                "is_sink": cid == 0, "k": k, "v": v,
            })
        kv = {"segments": segs, "layer_idx": 0, "current_chunk_id": 3}
        # Query near segment-1 keys.
        q = segs[1]["k"][:, :2].clone()
        cur_k = torch.randn(1, 2, 2, 8) * 0.01
        cur_v = torch.randn(1, 2, 2, 8) * 0.01
        k_cache, v_cache = build_dynamic_kv_tensors(kv, cur_k, cur_v, None)

        full = _attention_output_from_kv(q, k_cache, v_cache, head_dim=8)
        k_drop, v_drop = build_dynamic_kv_tensors(
            kv, cur_k, cur_v, None, exclude_chunk_ids={1})
        drop = _attention_output_from_kv(q, k_drop, v_drop, head_dim=8)
        regret = (full - drop).flatten().norm() / full.flatten().norm().clamp_min(1e-8)
        self.assertGreater(float(regret.item()), 1e-4)

        oracle_reset()
        oracle_set(True, probe_every=1, mode="attn_output_regret", regret_max_cands=4)
        _record_oracle_mass(kv, q, k_cache, head_dim=8, v_cache=v_cache)
        recs = oracle_state()["records"]
        self.assertEqual(len(recs), 1)
        self.assertIn("seg_regret", recs[0])
        regrets = [r for r in recs[0]["seg_regret"] if r is not None]
        self.assertTrue(len(regrets) >= 1)
        # Not all finite regrets identical (drop-one actually changes output).
        if len(regrets) >= 2:
            self.assertGreater(max(regrets) - min(regrets), 1e-8)


if __name__ == "__main__":
    unittest.main()
