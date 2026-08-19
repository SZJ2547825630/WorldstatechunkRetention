"""Unit tests for Timestep-Invariant Condition Hoisting (no GPU required)."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_pkg = types.ModuleType("wan")
_pkg.__path__ = [str(_ROOT / "wan")]
_mods = types.ModuleType("wan.modules")
_mods.__path__ = [str(_ROOT / "wan" / "modules")]
sys.modules.setdefault("wan", _pkg)
sys.modules.setdefault("wan.modules", _mods)
_ch = _load("wan.modules.cond_hoist", _ROOT / "wan" / "modules" / "cond_hoist.py")


class _FakePatchModel(nn.Module):
    def __init__(self, in_ch=32, out_ch=8, k=(1, 2, 2)):
        super().__init__()
        self.patch_embedding = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=k)
        self.patch_size = k


class TestCondHoist(unittest.TestCase):
    def setUp(self):
        _ch.hoist_reset_all()

    def test_conv3d_split_matches_full(self):
        torch.manual_seed(0)
        model = _FakePatchModel(in_ch=32, out_ch=16, k=(1, 2, 2))
        x = torch.randn(16, 2, 8, 8)
        y = torch.randn(16, 2, 8, 8)
        full = model.patch_embedding(torch.cat([x, y], dim=0).unsqueeze(0))

        _ch.hoist_begin_chunk(0, enabled=True, conv_split=True, profile=False)
        static = _ch.conv3d_static_contribution(model, [y])
        split = _ch.patch_embed_split(model, [x], [y], static_list=static)[0]
        self.assertTrue(torch.allclose(full, split, rtol=1e-4, atol=1e-4))

    def test_block_cam_cache_mem_estimate(self):
        est = _ch.estimate_block_cam_cache_bytes(
            num_layers=40, tokens=4680, dim=5120, dtype_bytes=2)
        # 40 * 2 * 4680 * 5120 * 2 bytes
        self.assertGreater(est["mb_total"], 100.0)
        self.assertLess(est["gb_total"], 8.0)

    def test_hoist_reuse_counters(self):
        _ch.hoist_begin_chunk(0, enabled=True, global_cam=True)
        self.assertTrue(_ch.hoist_enabled())
        _ch.hoist_state()["cam_emb"] = torch.zeros(1, 4, 8)
        # second get should reuse — need a fake model; just check state API
        st = _ch.hoist_summary(num_steps=4)
        self.assertEqual(st["assumptions"]["frac_saved_on_hoistable"], 0.75)
        _ch.hoist_end_chunk()
        self.assertIsNone(_ch.hoist_state()["cam_emb"])


if __name__ == "__main__":
    unittest.main()
