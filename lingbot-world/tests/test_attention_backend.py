"""Regression tests for the attention backend dispatch.

Self-attention in the DiT runs without a padding mask, which lets it use cuDNN's
fused Hopper kernel instead of FlashAttention 2 (1.6x on H20 at the shape this
model runs, ~15% off a forward). These tests pin the two things that make that
safe: masked calls must stay on the flash path, and the cuDNN result must agree
with flash to bf16 rounding.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wan.modules import attention as attn_mod  # noqa: E402


def _qkv(b, lq, lk, n, d, device):
    torch.manual_seed(0)
    q = torch.randn(b, lq, n, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(b, lk, n, d, device=device, dtype=torch.bfloat16)
    return q, k, torch.randn_like(k)


class TestEligibility(unittest.TestCase):
    """Runs on CPU: eligibility is pure shape/argument inspection."""

    def setUp(self):
        self.q = torch.zeros(1, 8, 4, 64)
        self.k = torch.zeros(1, 16, 4, 64)
        self.kw = dict(q_lens=None, k_lens=None, dropout_p=0.,
                       window_size=(-1, -1), deterministic=False)
        self._saved_backend = attn_mod._SDPA_BACKEND
        attn_mod._SDPA_BACKEND = "auto"

    def tearDown(self):
        attn_mod._SDPA_BACKEND = self._saved_backend

    def test_unmasked_is_eligible(self):
        self.assertTrue(attn_mod._sdpa_eligible(self.q, self.k, **self.kw))

    def test_padding_mask_falls_back(self):
        for key in ("q_lens", "k_lens"):
            kw = dict(self.kw, **{key: torch.tensor([4], dtype=torch.int32)})
            self.assertFalse(attn_mod._sdpa_eligible(self.q, self.k, **kw), key)

    def test_sliding_window_falls_back(self):
        kw = dict(self.kw, window_size=(128, 0))
        self.assertFalse(attn_mod._sdpa_eligible(self.q, self.k, **kw))

    def test_dropout_and_deterministic_fall_back(self):
        self.assertFalse(attn_mod._sdpa_eligible(
            self.q, self.k, **dict(self.kw, dropout_p=0.1)))
        self.assertFalse(attn_mod._sdpa_eligible(
            self.q, self.k, **dict(self.kw, deterministic=True)))

    def test_gqa_and_large_head_dim_fall_back(self):
        k_gqa = torch.zeros(1, 16, 2, 64)
        self.assertFalse(attn_mod._sdpa_eligible(self.q, k_gqa, **self.kw))
        q_big, k_big = torch.zeros(1, 8, 4, 256), torch.zeros(1, 16, 4, 256)
        self.assertFalse(attn_mod._sdpa_eligible(q_big, k_big, **self.kw))

    def test_causal_falls_back_unless_square(self):
        # Flash anchors the causal mask bottom-right, `is_causal` top-left.
        self.assertFalse(attn_mod._sdpa_eligible(
            self.q, self.k, causal=True, **self.kw))
        square = torch.zeros(1, 16, 4, 64)
        self.assertTrue(attn_mod._sdpa_eligible(
            square, self.k, causal=True, **self.kw))

    def test_env_override_forces_flash(self):
        saved = attn_mod._SDPA_BACKEND
        try:
            attn_mod._SDPA_BACKEND = "flash"
            self.assertFalse(attn_mod._sdpa_eligible(self.q, self.k, **self.kw))
        finally:
            attn_mod._SDPA_BACKEND = saved


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestParity(unittest.TestCase):
    def setUp(self):
        self._saved_backend = attn_mod._SDPA_BACKEND
        attn_mod._CUDNN_SDPA_USABLE[0] = True

    def tearDown(self):
        attn_mod._SDPA_BACKEND = self._saved_backend

    def _compare(self, lq=512, lk=900, **kw):
        q, k, v = _qkv(1, lq, lk, 8, 64, "cuda")
        attn_mod._SDPA_BACKEND = "flash"
        ref = attn_mod.flash_attention(q, k, v, **kw).float()
        attn_mod._SDPA_BACKEND = "auto"
        got = attn_mod.flash_attention(q, k, v, **kw).float()
        self.assertEqual(ref.shape, got.shape)
        rel = (ref - got).abs().sum().item() / ref.abs().sum().item()
        return rel

    def test_matches_flash_within_bf16_rounding(self):
        for kw in ({}, {"q_scale": 0.7}, {"softmax_scale": 0.05}):
            self.assertLess(self._compare(**kw), 5e-3, kw)

    def test_square_causal_matches_flash(self):
        self.assertLess(self._compare(lq=900, lk=900, causal=True), 5e-3)

    def test_nonsquare_causal_is_bitwise_identical(self):
        # Ineligible, so both backends run flash and the masks cannot diverge.
        self.assertEqual(self._compare(causal=True), 0.0)

    def test_masked_call_is_bitwise_identical(self):
        # k_lens must never reach cuDNN, so both backends run the same kernel.
        self.assertEqual(
            self._compare(k_lens=torch.tensor([900], dtype=torch.int32)), 0.0)

    def test_dispatch_counters_track_the_path(self):
        q, k, v = _qkv(1, 128, 256, 4, 64, "cuda")
        before = dict(attn_mod.ATTN_PATH_COUNTS)
        attn_mod._SDPA_BACKEND = "auto"
        attn_mod.flash_attention(q, k, v)
        attn_mod.flash_attention(q, k, v, k_lens=torch.tensor([256], dtype=torch.int32))
        self.assertEqual(attn_mod.ATTN_PATH_COUNTS["cudnn"], before["cudnn"] + 1)
        self.assertEqual(attn_mod.ATTN_PATH_COUNTS["flash"], before["flash"] + 1)

    def test_default_backend_is_auto(self):
        # Production default prefers cuDNN; FA2 is opt-out via WAN_ATTN_BACKEND=flash.
        if "WAN_ATTN_BACKEND" in os.environ:
            self.skipTest("WAN_ATTN_BACKEND overridden in environment")
        self.assertEqual(os.getenv("WAN_ATTN_BACKEND", "auto").lower(), "auto")
        self.assertNotEqual(attn_mod._SDPA_BACKEND, "flash")


if __name__ == "__main__":
    unittest.main()
