"""Timestep-Invariant Condition Hoisting (TICH).

Exact elimination of chunk-invariant conditioning work inside the S-step
denoising loop. Not activation reuse — mathematically equivalent graph rewrite.

Hoistable subgraphs (same chunk, all timesteps):
  1. Global camera embedding (patch_embedding_wancamctrl + 2 Linear)
  2. Per-block camera Linear×4 → cached cam_scale / cam_shift
  3. I2V patch Conv3d static channel split: Conv([x,c]) = Conv_x(x)+Conv_c(c)+b

Elementwise ``x = (1+scale)*x + shift`` still runs every step.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_HOIST: Dict[str, Any] = {
    "enabled": False,
    "global_cam": True,
    "block_cam": True,
    "conv_split": True,
    "profile": False,
    "verify": False,
    # per-chunk caches
    "cam_emb": None,            # post-global-MLP token emb (pre SP pad/chunk)
    "cam_emb_sp": None,         # optional post-SP-chunk emb for this rank
    "block_scale": None,        # list[Tensor] len=num_layers
    "block_shift": None,
    "conv_static": None,        # list of static patch-emb contributions [B,C,F',H',W']
    "chunk_id": -1,
    "n_global_compute": 0,
    "n_global_reuse": 0,
    "n_block_compute": 0,
    "n_block_reuse": 0,
    "n_conv_split": 0,
    "n_conv_full": 0,
    # profile ms
    "time_ms": {
        "global_cam": [],
        "block_linears": [],
        "elemwise": [],
        "conv_static": [],
        "conv_dynamic": [],
        "conv_full": [],
    },
    # verify
    "verify_records": [],
}


def hoist_reset_all():
    hoist_end_chunk()
    _HOIST["enabled"] = False
    _HOIST["verify_records"] = []
    for k in _HOIST["time_ms"]:
        _HOIST["time_ms"][k] = []
    for k in (
        "n_global_compute", "n_global_reuse", "n_block_compute",
        "n_block_reuse", "n_conv_split", "n_conv_full",
    ):
        _HOIST[k] = 0


def hoist_begin_chunk(
    chunk_id: int = -1,
    *,
    enabled: bool = True,
    global_cam: bool = True,
    block_cam: bool = True,
    conv_split: bool = True,
    profile: bool = False,
    verify: bool = False,
):
    """Clear per-chunk caches; configure hoist flags."""
    _HOIST["enabled"] = bool(enabled)
    _HOIST["global_cam"] = bool(global_cam)
    _HOIST["block_cam"] = bool(block_cam)
    _HOIST["conv_split"] = bool(conv_split)
    _HOIST["profile"] = bool(profile)
    _HOIST["verify"] = bool(verify)
    _HOIST["chunk_id"] = int(chunk_id)
    _HOIST["cam_emb"] = None
    _HOIST["cam_emb_sp"] = None
    _HOIST["block_scale"] = None
    _HOIST["block_shift"] = None
    _HOIST["conv_static"] = None


def hoist_end_chunk():
    _HOIST["cam_emb"] = None
    _HOIST["cam_emb_sp"] = None
    _HOIST["block_scale"] = None
    _HOIST["block_shift"] = None
    _HOIST["conv_static"] = None
    _HOIST["chunk_id"] = -1


def hoist_state():
    return _HOIST


def hoist_enabled() -> bool:
    return bool(_HOIST.get("enabled"))


def _cuda_ms(fn, bucket: Optional[str] = None):
    if (not _HOIST.get("profile")) or (not torch.cuda.is_available()):
        return fn(), 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    ms = float(start.elapsed_time(end))
    if bucket:
        _HOIST["time_ms"].setdefault(bucket, []).append(ms)
    return out, ms


def record_verify(tag: str, a: torch.Tensor, b: torch.Tensor):
    if not _HOIST.get("verify"):
        return
    diff = (a.float() - b.float()).abs()
    _HOIST["verify_records"].append({
        "tag": str(tag),
        "chunk_id": int(_HOIST.get("chunk_id", -1)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rel": float(diff.norm() / (b.float().norm() + 1e-6)),
    })


def _mean(xs: List[float]) -> Optional[float]:
    xs = [float(v) for v in xs if math.isfinite(float(v))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def hoist_summary(*, num_steps: int = 4) -> Dict[str, Any]:
    """Aggregate hoist counters + profile; estimate theoretical wall save."""
    s = max(1, int(num_steps))
    frac_saved = float(s - 1) / float(s)  # hoist once, reuse s-1 times
    tm = _HOIST.get("time_ms") or {}
    profile = {
        k: {"n": len(v), "mean_ms": _mean(v), "sum_ms": (sum(v) if v else None)}
        for k, v in tm.items()
    }
    # Hoistable work per DiT step ≈ global_cam + block_linears + conv_static
    # (elemwise and conv_dynamic remain).
    hoistable = []
    for k in ("global_cam", "block_linears", "conv_static"):
        m = profile.get(k, {}).get("mean_ms")
        if m is not None:
            hoistable.append(m)
    # Also compare against full conv when split off
    verify = list(_HOIST.get("verify_records") or [])
    max_abs = max((r["max_abs"] for r in verify), default=None)
    return {
        "enabled": bool(_HOIST.get("enabled")),
        "flags": {
            "global_cam": bool(_HOIST.get("global_cam")),
            "block_cam": bool(_HOIST.get("block_cam")),
            "conv_split": bool(_HOIST.get("conv_split")),
            "profile": bool(_HOIST.get("profile")),
            "verify": bool(_HOIST.get("verify")),
        },
        "counts": {
            "global_compute": int(_HOIST.get("n_global_compute", 0)),
            "global_reuse": int(_HOIST.get("n_global_reuse", 0)),
            "block_compute": int(_HOIST.get("n_block_compute", 0)),
            "block_reuse": int(_HOIST.get("n_block_reuse", 0)),
            "conv_split": int(_HOIST.get("n_conv_split", 0)),
            "conv_full": int(_HOIST.get("n_conv_full", 0)),
        },
        "profile_ms": profile,
        "verify": {
            "n": len(verify),
            "max_abs": max_abs,
            "records_tail": verify[-8:],
        },
        "assumptions": {
            "num_steps": s,
            "frac_saved_on_hoistable": frac_saved,
            "note": (
                "Theoretical save applies only to hoistable kernels "
                "(global cam + block linears + conv_static), not elemwise/cam inject."
            ),
        },
    }


# ── Camera global embedding ─────────────────────────────────────────────

def compute_global_cam_emb(model, raw_plucker_list) -> torch.Tensor:
    """raw list of [1,C,F,H,W] → token emb [B,L,dim] (before SP pad/chunk)."""
    from einops import rearrange
    c2ws = [
        rearrange(
            i,
            '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
            c1=model.patch_size[0],
            c2=model.patch_size[1],
            c3=model.patch_size[2],
        ) for i in raw_plucker_list
    ]
    c2ws = torch.cat(c2ws, dim=1)
    c2ws = model.patch_embedding_wancamctrl(c2ws)
    hidden = model.c2ws_hidden_states_layer2(
        F.silu(model.c2ws_hidden_states_layer1(c2ws)))
    return c2ws + hidden


def get_or_compute_global_cam(model, dit_cond_dict) -> Optional[torch.Tensor]:
    if dit_cond_dict is None or "c2ws_plucker_emb" not in dit_cond_dict:
        return None
    raw = dit_cond_dict["c2ws_plucker_emb"]
    # Already hoisted token emb? (marked)
    if dit_cond_dict.get("_cam_emb_ready"):
        return raw
    if hoist_enabled() and _HOIST.get("global_cam") and _HOIST.get("cam_emb") is not None:
        _HOIST["n_global_reuse"] = int(_HOIST["n_global_reuse"]) + 1
        return _HOIST["cam_emb"]

    def _run():
        return compute_global_cam_emb(model, raw)

    emb, _ = _cuda_ms(_run, "global_cam" if hoist_enabled() else None)
    _HOIST["n_global_compute"] = int(_HOIST["n_global_compute"]) + 1
    if hoist_enabled() and _HOIST.get("global_cam"):
        _HOIST["cam_emb"] = emb
    return emb


# ── Per-block scale/shift ───────────────────────────────────────────────

def ensure_block_cam_slots(num_layers: int):
    """Allocate per-layer cache slots (lazy fill under FSDP)."""
    n = max(1, int(num_layers))
    if _HOIST.get("block_scale") is None or len(_HOIST["block_scale"]) != n:
        _HOIST["block_scale"] = [None] * n
        _HOIST["block_shift"] = [None] * n


def precompute_block_cam(model, cam_emb: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Eager precompute — ONLY safe when parameters are fully materialized
    (no FSDP sharding). Prefer lazy fill via ``block_cam_mod`` under FSDP.
    """
    ensure_block_cam_slots(len(model.blocks))

    def _run():
        for i, block in enumerate(model.blocks):
            h = block.cam_injector_layer2(
                F.silu(block.cam_injector_layer1(cam_emb)))
            h = h + cam_emb
            _HOIST["block_scale"][i] = block.cam_scale_layer(h)
            _HOIST["block_shift"][i] = block.cam_shift_layer(h)
        return _HOIST["block_scale"], _HOIST["block_shift"]

    (scales, shifts), _ = _cuda_ms(_run, "block_linears")
    _HOIST["n_block_compute"] = int(_HOIST["n_block_compute"]) + 1
    return scales, shifts


def block_cam_mod(block, x, dit_cond_dict, block_index: int = 0):
    """Apply camera modulation; hoist Linear×4 when cached.

    Under FSDP, scale/shift are computed lazily on the first call for each
    block (parameters are unsharded in that block's forward), then reused.
    """
    if dit_cond_dict is None:
        return x
    # Prefer explicit precomputed tensors in dit_cond_dict
    if "cam_scale" in dit_cond_dict and "cam_shift" in dit_cond_dict:
        scale = dit_cond_dict["cam_scale"]
        shift = dit_cond_dict["cam_shift"]
        if hoist_enabled() and _HOIST.get("profile"):
            def _elem():
                return (1.0 + scale) * x + shift
            out, _ = _cuda_ms(_elem, "elemwise")
            return out
        return (1.0 + scale) * x + shift

    if "c2ws_plucker_emb" not in dit_cond_dict:
        return x

    bi = int(block_index)
    use_hoist = hoist_enabled() and bool(_HOIST.get("block_cam"))
    if use_hoist:
        # Lazy slots — filled the first time this block runs (FSDP-safe).
        if _HOIST.get("block_scale") is None:
            # Unknown layer count: grow on demand
            ensure_block_cam_slots(max(bi + 1, 40))
        elif bi >= len(_HOIST["block_scale"]):
            # Grow if needed
            extra = bi + 1 - len(_HOIST["block_scale"])
            _HOIST["block_scale"].extend([None] * extra)
            _HOIST["block_shift"].extend([None] * extra)

        if _HOIST["block_scale"][bi] is not None:
            scale = _HOIST["block_scale"][bi]
            shift = _HOIST["block_shift"][bi]
            _HOIST["n_block_reuse"] = int(_HOIST["n_block_reuse"]) + 1
            if _HOIST.get("profile"):
                def _elem():
                    return (1.0 + scale) * x + shift
                out, _ = _cuda_ms(_elem, "elemwise")
                return out
            return (1.0 + scale) * x + shift

    # Live path (also first-touch fill under hoist)
    emb = dit_cond_dict["c2ws_plucker_emb"]

    def _linears():
        h = block.cam_injector_layer2(
            F.silu(block.cam_injector_layer1(emb)))
        h = h + emb
        scale = block.cam_scale_layer(h)
        shift = block.cam_shift_layer(h)
        return scale, shift

    (scale, shift), _ = _cuda_ms(
        _linears, "block_linears" if (hoist_enabled() and _HOIST.get("profile")) else None)
    _HOIST["n_block_compute"] = int(_HOIST["n_block_compute"]) + 1

    if use_hoist:
        _HOIST["block_scale"][bi] = scale
        _HOIST["block_shift"][bi] = shift

    def _elem():
        return (1.0 + scale) * x + shift

    out, _ = _cuda_ms(
        _elem, "elemwise" if (hoist_enabled() and _HOIST.get("profile")) else None)
    return out


# ── Conv3d static / dynamic split ───────────────────────────────────────

def conv3d_static_contribution(model, condition_list) -> List[torch.Tensor]:
    """Conv_c(c) + bias for each sample. condition: [C_c,F,H,W]."""
    w = model.patch_embedding.weight
    b = model.patch_embedding.bias
    in_ch = int(w.shape[1])
    outs = []

    def _run():
        local = []
        for c in condition_list:
            cy = int(c.shape[0])
            assert cy < in_ch, "condition channels must be the trailing split"
            cx = in_ch - cy
            w_c = w[:, cx:cx + cy]
            # F.conv3d: input [N,C,D,H,W]
            inp = c.unsqueeze(0)
            y = F.conv3d(inp, w_c, bias=b, stride=model.patch_embedding.stride,
                         padding=model.patch_embedding.padding)
            local.append(y)
        return local

    outs, _ = _cuda_ms(_run, "conv_static")
    return outs


def patch_embed_split(model, x_list, y_list, static_list=None) -> List[torch.Tensor]:
    """Exact: Conv([x,y]) = Conv_x(x) + Conv_y(y) + bias."""
    w = model.patch_embedding.weight
    b = model.patch_embedding.bias
    in_ch = int(w.shape[1])
    stride = model.patch_embedding.stride
    padding = model.patch_embedding.padding
    outs = []

    use_cache = (
        hoist_enabled() and _HOIST.get("conv_split")
        and static_list is not None
    )
    if use_cache:
        _HOIST["n_conv_split"] = int(_HOIST["n_conv_split"]) + 1
    else:
        _HOIST["n_conv_full"] = int(_HOIST["n_conv_full"]) + 1

    for i, (x, y) in enumerate(zip(x_list, y_list)):
        cx, cy = int(x.shape[0]), int(y.shape[0])
        assert cx + cy == in_ch, f"channel split mismatch {cx}+{cy} vs {in_ch}"
        if use_cache:
            w_x = w[:, :cx]

            def _dyn(x=x, w_x=w_x):
                return F.conv3d(
                    x.unsqueeze(0), w_x, bias=None,
                    stride=stride, padding=padding)

            dyn, _ = _cuda_ms(_dyn, "conv_dynamic")
            outs.append(dyn + static_list[i])
        else:
            cat = torch.cat([x, y], dim=0)

            def _full(cat=cat):
                return model.patch_embedding(cat.unsqueeze(0))

            out, _ = _cuda_ms(_full, "conv_full")
            outs.append(out)
    return outs


def estimate_block_cam_cache_bytes(
    num_layers: int, tokens: int, dim: int, dtype_bytes: int = 2,
) -> Dict[str, float]:
    """Memory for caching scale+shift per layer: 2 * L * tokens * dim * dtype."""
    per = 2.0 * float(tokens) * float(dim) * float(dtype_bytes)
    total = per * float(num_layers)
    return {
        "bytes_per_layer": per,
        "bytes_total": total,
        "mb_total": total / (1024.0 ** 2),
        "gb_total": total / (1024.0 ** 3),
        "num_layers": int(num_layers),
        "tokens": int(tokens),
        "dim": int(dim),
        "dtype_bytes": int(dtype_bytes),
    }
