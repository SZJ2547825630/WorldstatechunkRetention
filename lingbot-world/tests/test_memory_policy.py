"""Regression tests for explicit --memory_policy mapping and flag gating."""

from __future__ import annotations

import argparse
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from generate_fast import _apply_memory_policy


def _args(**kwargs):
    defaults = dict(
        memory_policy=None,
        enable_motion_adaptive_kv_eviction=False,
        enable_swtp=False,
        consolidation="off",
        archive_diversity_pool=0,
        consol_rank_alpha=None,
        consol_gist_tokens=None,
        consol_l2_bottom_ratio=None,
        selector="learned",
        selector_ckpt=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestMemoryPolicy(unittest.TestCase):
    def test_legacy_preserves_flags(self):
        a = _args(memory_policy="legacy", enable_motion_adaptive_kv_eviction=True, selector="heuristic")
        _apply_memory_policy(a)
        self.assertTrue(a.enable_motion_adaptive_kv_eviction)
        self.assertEqual(a.selector, "heuristic")

    def test_window(self):
        a = _args(memory_policy="window", enable_motion_adaptive_kv_eviction=True, selector="learned")
        _apply_memory_policy(a)
        self.assertFalse(a.enable_motion_adaptive_kv_eviction)
        self.assertEqual(a.selector, "heuristic")

    def test_heuristic_cr(self):
        a = _args(memory_policy="heuristic_cr")
        _apply_memory_policy(a)
        self.assertTrue(a.enable_motion_adaptive_kv_eviction)
        self.assertEqual(a.selector, "heuristic")

    def test_learned_cr(self):
        a = _args(memory_policy="learned_cr", selector="heuristic")
        _apply_memory_policy(a)
        self.assertTrue(a.enable_motion_adaptive_kv_eviction)
        self.assertEqual(a.selector, "learned")
        self.assertTrue(a.selector_ckpt.endswith("selector_all4.pt"))

    def test_world_state_cr_defaults_to_v3(self):
        a = _args(memory_policy="world_state_cr", selector="heuristic")
        _apply_memory_policy(a)
        self.assertTrue(a.enable_motion_adaptive_kv_eviction)
        self.assertEqual(a.selector, "learned")
        self.assertTrue(a.selector_ckpt.endswith("selector_ws_future_v1.pt"))
        self.assertTrue(a.enable_swtp)
        self.assertEqual(a.consolidation, "full")
        self.assertEqual(a.archive_diversity_pool, 4)
        self.assertEqual(a.consol_rank_alpha, 0.5)
        self.assertEqual(a.consol_gist_tokens, 64)
        self.assertEqual(a.consol_l2_bottom_ratio, 0.5)

    def test_world_state_cr_aliases_map_to_v3(self):
        for alias in ("world_state_cr_future", "world_state_cr_v3", "world_state_cr_consol"):
            a = _args(memory_policy=alias, selector="heuristic")
            _apply_memory_policy(a)
            self.assertEqual(a.memory_policy, "world_state_cr")
            self.assertEqual(a.consolidation, "full")
            self.assertTrue(a.enable_swtp)

    def test_world_state_cr_v2_ablation(self):
        a = _args(memory_policy="world_state_cr_v2", selector="heuristic",
                  enable_swtp=True, consolidation="full")
        _apply_memory_policy(a)
        self.assertTrue(a.enable_motion_adaptive_kv_eviction)
        self.assertTrue(a.selector_ckpt.endswith("selector_ws_future_v1.pt"))
        self.assertEqual(a.consolidation, "off")
        self.assertFalse(a.enable_swtp)

    def test_world_state_cr_v1_ablation(self):
        a = _args(memory_policy="world_state_cr_v1", selector="heuristic")
        _apply_memory_policy(a)
        self.assertTrue(a.enable_motion_adaptive_kv_eviction)
        self.assertEqual(a.selector, "learned")
        self.assertTrue(a.selector_ckpt.endswith("selector_ws_v1.pt"))
        self.assertEqual(a.consolidation, "off")

    def test_old_names_rejected(self):
        for policy in ("moce", "m1", "stategraph", "mosaic"):
            a = _args(memory_policy=policy)
            with self.assertRaises(AssertionError):
                _apply_memory_policy(a)

    def test_unknown_raises(self):
        a = _args(memory_policy="not_a_policy")
        with self.assertRaises(AssertionError):
            _apply_memory_policy(a)


class TestBatchGenerateMethodNames(unittest.TestCase):
    def test_official_methods_only(self):
        import importlib.util
        path = os.path.join(ROOT, "bench", "batch_generate.py")
        spec = importlib.util.spec_from_file_location("batch_generate_mod", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        expected = {
            "window", "heuristic_cr", "learned_cr",
            "world_state_cr", "world_state_cr_v3",
            "world_state_cr_future", "world_state_cr_consol", "ws_v3_a05_g64",
            "world_state_cr_v1", "world_state_cr_v2", "world_state_cr_ema",
            "swtp",
            "ws_v3_a0", "ws_v3_a05", "ws_v3_a1", "ws_v3_a05_g128",
        }
        self.assertEqual(set(mod.METHODS), expected)
        # Default world_state_cr is v3 = future-use ckpt + consol full + SWTP
        # with default_loop winner knobs (α=0.5, gist=64).
        default = mod.METHODS["world_state_cr"]
        self.assertTrue(default["selector_ckpt"].endswith("selector_ws_future_v1.pt"))
        self.assertEqual(default.get("consolidation"), "full")
        self.assertTrue(default.get("enable_swtp"))
        self.assertEqual(default.get("consol_rank_alpha"), 0.5)
        self.assertEqual(default.get("consol_gist_tokens"), 64)
        self.assertEqual(default.get("consol_l2_bottom_ratio"), 0.5)
        for alias in ("world_state_cr_v3", "world_state_cr_future",
                      "world_state_cr_consol", "ws_v3_a05_g64"):
            self.assertEqual(mod.METHODS[alias].get("consolidation"), "full")
            self.assertEqual(mod.METHODS[alias]["selector_ckpt"], default["selector_ckpt"])
            self.assertEqual(mod.METHODS[alias].get("consol_rank_alpha"), 0.5)
            self.assertEqual(mod.METHODS[alias].get("consol_gist_tokens"), 64)
        v2 = mod.METHODS["world_state_cr_v2"]
        self.assertTrue(v2["selector_ckpt"].endswith("selector_ws_future_v1.pt"))
        self.assertNotIn("consolidation", v2)
        self.assertFalse(v2.get("enable_swtp", False))
        self.assertTrue(
            mod.METHODS["world_state_cr_v1"]["selector_ckpt"].endswith(
                "selector_ws_v1.pt"))
        for dead in (
            "m1", "m1_ws", "m1_legacy", "moce", "MoCE", "stategraph",
            "MoSaiC_2step", "mosaic", "mosaic_consol", "MoSaiC",
        ):
            self.assertNotIn(dead, mod.METHODS)


if __name__ == "__main__":
    unittest.main()
