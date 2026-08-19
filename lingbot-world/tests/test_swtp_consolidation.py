"""Unit tests for SWTP spatial pooling / energy cover / Memory Consolidation."""
from __future__ import annotations

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wan.utils.swtp import (  # noqa: E402
    apply_swtp_to_kv,
    choose_keep_count,
    gini_coefficient,
    infer_token_grid,
)
from wan.utils.memory_consolidation import (  # noqa: E402
    ConsolidationConfig,
    assign_archive_tiers,
    enforce_gist_budget,
    rank_score,
    update_revisit_coverage,
    update_utility_ema,
)


class TestSWTP(unittest.TestCase):
    def test_infer_grid_prefers_explicit(self):
        self.assertEqual(infer_token_grid(4680, prefer=(3, 30, 52)), (3, 30, 52))

    def test_energy_cover_keeps_less_when_concentrated(self):
        s = torch.zeros(100)
        s[:5] = 1.0
        k = choose_keep_count(s, keep_ratio=0.5, energy_cover=0.9)
        self.assertLessEqual(k, 10)
        self.assertGreaterEqual(k, 1)

    def test_energy_cover_hits_cap_when_diffuse(self):
        s = torch.ones(100)
        k = choose_keep_count(s, keep_ratio=0.5, energy_cover=0.9)
        self.assertEqual(k, 50)

    def test_spatial_summary_narrower_than_raster_span(self):
        # Build a toy [B,T,H,D] with known grid 1x8x8.
        b, f, ht, wt, h, d = 1, 1, 8, 8, 2, 4
        t = f * ht * wt
        k = torch.randn(b, t, h, d)
        v = torch.randn_like(k)
        sal = torch.rand(t)
        k_red, v_red, kept = apply_swtp_to_kv(
            k, v, sal, keep_ratio=0.5, num_summary=8,
            token_grid=(f, ht, wt), energy_cover=0.0, mode="standard",
        )
        self.assertEqual(k_red.shape[0], 1)
        self.assertEqual(k_red.shape[1], v_red.shape[1])
        # Kept + some summaries; should be well below full T.
        self.assertLess(k_red.shape[1], t)
        self.assertGreater(k_red.shape[1], kept.numel())

    def test_gist_mode_has_no_kept_tokens(self):
        t, h, d = 64, 2, 4
        k = torch.randn(1, t, h, d)
        v = torch.randn_like(k)
        sal = torch.ones(t)
        k_red, v_red, kept = apply_swtp_to_kv(
            k, v, sal, keep_ratio=0.0, num_summary=8,
            token_grid=(1, 8, 8), mode="gist",
        )
        self.assertEqual(kept.numel(), 0)
        self.assertGreater(k_red.shape[1], 0)
        self.assertLessEqual(k_red.shape[1], 8)

    def test_uniform_mode_does_not_skip(self):
        t = 64
        k = torch.randn(1, t, 2, 4)
        v = torch.randn_like(k)
        sal = torch.ones(t)  # gini ~ 0
        self.assertLess(gini_coefficient(sal), 0.05)
        k_red, _, _ = apply_swtp_to_kv(
            k, v, sal, keep_ratio=0.5, num_summary=4,
            token_grid=(1, 8, 8), mode="uniform",
        )
        self.assertLess(k_red.shape[1], t)

    def test_summary_norm_compensation(self):
        t = 64
        k = torch.randn(1, t, 2, 8)
        v = torch.randn_like(k)
        sal = torch.rand(t)
        k_red, _, kept = apply_swtp_to_kv(
            k, v, sal, keep_ratio=0.5, num_summary=4,
            token_grid=(1, 8, 8), energy_cover=0.0, compensate_summary_norm=True,
        )
        if k_red.shape[1] > kept.numel() and kept.numel() > 0:
            kept_n = k.index_select(1, kept.to(k.device)).norm(dim=-1).mean()
            # Approximate: last tokens are summaries
            sum_n = k_red[:, kept.numel():].norm(dim=-1).mean()
            self.assertGreater(float(sum_n / kept_n), 0.5)


class TestConsolidation(unittest.TestCase):
    def test_ema_converges_toward_new_score(self):
        meta = {}
        cfg = ConsolidationConfig(mode="ema", beta=0.5)
        update_utility_ema(meta, 1.0, beta=cfg.beta, patience=2, stabilize_thr=0.9)
        update_utility_ema(meta, 0.0, beta=cfg.beta, patience=2, stabilize_thr=0.9)
        self.assertAlmostEqual(meta["u_ema"], 0.5, places=5)

    def test_hysteresis_and_stabilize(self):
        meta = {}
        for _ in range(3):
            update_utility_ema(meta, 1.0, beta=0.0, patience=2, stabilize_thr=0.5)
        self.assertTrue(meta.get("stabilized"))

    def test_assign_tiers_ema_is_binary(self):
        cfg = ConsolidationConfig(mode="ema")
        segs = [
            {"chunk_id": i, "_sel_score": 1.0 - 0.1 * i, "motion_score": 0.0}
            for i in range(6)
        ]
        meta = {i: {"u_ema": 1.0 - 0.1 * i} for i in range(6)}
        tiers = assign_archive_tiers(segs, keep_count=3, chunk_meta=meta, cfg=cfg)
        self.assertEqual(sum(1 for t in tiers.values() if t == "L1"), 3)
        self.assertEqual(sum(1 for t in tiers.values() if t == "L3"), 3)
        self.assertNotIn("L2", tiers.values())

    def test_assign_tiers_full_demotes_bottom_half(self):
        cfg = ConsolidationConfig(mode="full", patience=99, l2_bottom_ratio=0.5)
        segs = [{"chunk_id": 0, "_sel_score": 0.9, "motion_score": 0.0},
                {"chunk_id": 1, "_sel_score": 0.1, "motion_score": 0.0}]
        meta = {
            0: {"u_ema": 0.9, "low_streak": 0, "stabilized": False},
            1: {"u_ema": 0.1, "low_streak": 0, "stabilized": False},
        }
        tiers = assign_archive_tiers(segs, keep_count=2, chunk_meta=meta, cfg=cfg)
        self.assertEqual(tiers[0], "L1")
        self.assertEqual(tiers[1], "L2")

    def test_assign_tiers_stabilized_skips_l2(self):
        cfg = ConsolidationConfig(mode="full", l2_bottom_ratio=1.0)
        segs = [{"chunk_id": 0, "_sel_score": 0.5, "motion_score": 0.0},
                {"chunk_id": 1, "_sel_score": 0.4, "motion_score": 0.0}]
        meta = {
            0: {"u_ema": 0.5, "stabilized": True},
            1: {"u_ema": 0.4, "stabilized": True},
        }
        tiers = assign_archive_tiers(segs, keep_count=2, chunk_meta=meta, cfg=cfg)
        self.assertEqual(tiers[0], "L1")
        self.assertEqual(tiers[1], "L1")

    def test_rank_score_alpha_mix(self):
        cfg = ConsolidationConfig(mode="ema", rank_alpha=0.5, stabilize_bonus=0.0)
        seg = {"_sel_score": 1.0, "motion_score": 0.0}
        meta = {"u_ema": 0.0, "stabilized": False}
        self.assertAlmostEqual(rank_score(seg, meta, cfg), 0.5)

    def test_rank_score_bonus_for_stabilized(self):
        cfg = ConsolidationConfig(mode="ema", stabilize_bonus=0.05, rank_alpha=0.0)
        seg = {"_sel_score": 0.5, "motion_score": 0.0}
        meta = {"u_ema": 0.5, "stabilized": True}
        self.assertAlmostEqual(rank_score(seg, meta, cfg), 0.55)

    def test_gist_budget_demotes_weakest(self):
        cfg = ConsolidationConfig(mode="full", gist_tokens=100, gist_budget_tokens=150)
        segs = [{"chunk_id": i, "_sel_score": float(i), "motion_score": 0.0} for i in range(3)]
        meta = {i: {"u_ema": float(i)} for i in range(3)}
        tiers = {0: "L2", 1: "L2", 2: "L2"}
        out = enforce_gist_budget(tiers, segs, meta, cfg)
        self.assertEqual(sum(1 for t in out.values() if t == "L2"), 1)
        self.assertEqual(out[2], "L2")

    def test_revisit_coverage_probe(self):
        stats = {}
        update_revisit_coverage(stats, [1, 2, 3], retained_ids={1, 3})
        self.assertEqual(stats["revisit_events"], 3)
        self.assertAlmostEqual(stats["revisit_coverage"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
