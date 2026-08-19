"""Future Coverage Oracle labels for World-State CR (P0).

Converts per-step attention-mass teacher records into a *future survival*
utility target without leave-one-out DiT passes:

  y_{t,i} = sum_{h=1..H} gamma^{h-1} *
            [ alpha * AttnMass_{t+h}(i) + (1-alpha) * PoseReuse_{t+h}(i) ]

Inference still uses only causal 11-D features; this module is offline /
oracle-dump only. Existing World-State CR (attention_mass) is untouched.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from wan.modules.chunk_selector import (
    camera_forward_overlap,
    frustum_overlap_proxy,
)

LABEL_TYPE_FUTURE_USE = "future_use_v1"
LABEL_TYPE_ATTENTION_MASS = "attention_mass"

DEFAULT_H = 8
DEFAULT_GAMMA = 0.9
DEFAULT_ALPHA = 0.5


def pose_reuse(
    pose_cand: Optional[Dict[str, Any]],
    pose_future: Optional[Dict[str, Any]],
    cam_cand=None,
    cam_future=None,
    translation_scale: float = 1.0,
) -> float:
    """[0, 1] revisit / coverage proxy between a stored chunk and a future pose."""
    frustum = frustum_overlap_proxy(pose_cand, pose_future, translation_scale)
    if cam_cand is None and pose_cand is not None:
        cam_cand = pose_cand.get("camera_forward")
    if cam_future is None and pose_future is not None:
        cam_future = pose_future.get("camera_forward")
    forward = camera_forward_overlap(cam_cand, cam_future)
    # Map cosine [-1,1] -> [0,1], then average with frustum.
    forward01 = 0.5 * (forward + 1.0)
    return 0.5 * (float(frustum) + float(forward01))


def _mass_lookup(records: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], float]:
    table: Dict[Tuple[int, int], float] = {}
    for r in records:
        g = int(r["gen_chunk_id"])
        sid = int(r["seg_id"])
        table[(g, sid)] = float(r.get("mass", 0.0))
    return table


def _meta_pose(meta: Dict[Any, Any], chunk_id: int) -> Optional[Dict[str, Any]]:
    m = meta.get(chunk_id)
    if m is None:
        m = meta.get(str(chunk_id))
    if m is None:
        return None
    return m.get("pose")


def _meta_cam(meta: Dict[Any, Any], chunk_id: int):
    m = meta.get(chunk_id)
    if m is None:
        m = meta.get(str(chunk_id))
    if m is None:
        return None
    if m.get("camera_forward") is not None:
        return m.get("camera_forward")
    pose = m.get("pose") or {}
    return pose.get("camera_forward")


def future_use_utility(
    mass_table: Dict[Tuple[int, int], float],
    chunk_meta: Dict[Any, Any],
    t: int,
    seg_id: int,
    *,
    horizon: int = DEFAULT_H,
    gamma: float = DEFAULT_GAMMA,
    alpha: float = DEFAULT_ALPHA,
    translation_scale: float = 1.0,
) -> float:
    """Scalar future-use label for candidate ``seg_id`` at decision time ``t``."""
    total = 0.0
    weight_sum = 0.0
    for h in range(1, max(1, int(horizon)) + 1):
        w = float(gamma) ** (h - 1)
        fut = t + h
        attn = float(mass_table.get((fut, seg_id), 0.0))
        reuse = pose_reuse(
            _meta_pose(chunk_meta, seg_id),
            _meta_pose(chunk_meta, fut),
            cam_cand=_meta_cam(chunk_meta, seg_id),
            cam_future=_meta_cam(chunk_meta, fut),
            translation_scale=translation_scale,
        )
        total += w * (alpha * attn + (1.0 - alpha) * reuse)
        weight_sum += w
    if weight_sum <= 0:
        return 0.0
    return float(total)


def aggregate_future_use_records(
    records: Sequence[Dict[str, Any]],
    chunk_meta: Dict[Any, Any],
    *,
    horizon: int = DEFAULT_H,
    gamma: float = DEFAULT_GAMMA,
    alpha: float = DEFAULT_ALPHA,
    translation_scale: float = 1.0,
    sink_chunk_count: int = 0,
    recent_window: int = 1,
) -> List[Dict[str, Any]]:
    """Build future_use_v1 ranking records from attention_mass teacher rows.

    For each decision time ``t`` and archive candidate ``i`` (not sink, not in
    the last ``recent_window`` chunks before ``t``), emit a row whose ``mass``
    field holds ``y_{t,i}`` so ``train_selector.py`` can reuse its existing path.
    Original attention mass is preserved under ``attention_mass``.
    """
    mass_table = _mass_lookup(records)
    gen_ids = sorted({int(r["gen_chunk_id"]) for r in records})
    if not gen_ids:
        return []

    # All segment ids that ever appear as candidates / history.
    all_segs = sorted({int(r["seg_id"]) for r in records})
    out: List[Dict[str, Any]] = []
    for t in gen_ids:
        for sid in all_segs:
            if sid < int(sink_chunk_count):
                continue
            if sid >= t:
                continue
            if (t - sid) <= int(recent_window):
                continue
            # Need at least one future step with either mass or pose.
            has_future = any(
                ((t + h, sid) in mass_table) or (_meta_pose(chunk_meta, t + h) is not None)
                for h in range(1, max(1, int(horizon)) + 1)
            )
            if not has_future:
                continue
            y = future_use_utility(
                mass_table, chunk_meta, t, sid,
                horizon=horizon, gamma=gamma, alpha=alpha,
                translation_scale=translation_scale,
            )
            out.append({
                "gen_chunk_id": int(t),
                "seg_id": int(sid),
                "mass": float(y),
                "attention_mass": float(mass_table.get((t, sid), 0.0)),
                "n_layers": 1,
            })
    return out


def convert_oracle_payload(
    payload: Dict[str, Any],
    *,
    horizon: int = DEFAULT_H,
    gamma: float = DEFAULT_GAMMA,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """Return a new oracle payload with future_use_v1 labels."""
    cfg = dict(payload.get("config") or {})
    scale = float(cfg.get("translation_scale", 1.0) or 1.0)
    sink_n = int(cfg.get("sink_chunk_count", 0))
    recent = int(cfg.get("recent_window", 1))
    records = aggregate_future_use_records(
        payload.get("records") or [],
        payload.get("chunk_meta") or {},
        horizon=horizon,
        gamma=gamma,
        alpha=alpha,
        translation_scale=scale,
        sink_chunk_count=sink_n,
        recent_window=recent,
    )
    cfg = {
        **cfg,
        "label_type": LABEL_TYPE_FUTURE_USE,
        "label_version": LABEL_TYPE_FUTURE_USE,
        "future_horizon": int(horizon),
        "future_gamma": float(gamma),
        "future_alpha": float(alpha),
        "source_label_type": cfg.get("label_type", LABEL_TYPE_ATTENTION_MASS),
    }
    return {
        "records": records,
        "chunk_meta": payload.get("chunk_meta") or {},
        "config": cfg,
    }
