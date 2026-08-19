"""
Chunk selectors for Learned CR and World-State CR.

Heuristic CR ranks archive candidates by a hand-designed camera/latent
motion score. Learned CR (5-D) and World-State CR (11-D) replace that with
a tiny MLP that predicts per-chunk retention utility from cheap, causal
features. Both keep post-generation retention semantics (no demote/host path).

Trained offline via ``train_selector.py`` against attention-mass oracle rankings.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Versioned schemas. load_selector accepts current names and legacy aliases.
FEATURE_SCHEMA_VERSION = "world_state.v1"
_FEATURE_SCHEMA_ALIASES = {"world_state.v1", "m1.world_state.v1"}
# P0+P1: same 11-D layout, corrected formulas + future-use trained weights.
FEATURE_SCHEMA_VERSION_V2 = "world_state.v2"
_FEATURE_SCHEMA_V2_ALIASES = {
    "world_state.v2", "world_state.future.v1", "m1.world_state.v2",
}
FEATURE_NAMES = [
    "motion_norm",                # tanh(motion_score / motion_ref)
    "translation_distance",       # ||t_q - t_c|| / trajectory step scale
    "rotation_geodesic",          # SO(3) geodesic / pi
    "camera_forward_overlap",     # cosine of camera forward axes
    "frustum_overlap_proxy",      # sparse pose+intrinsics ray proxy (no depth)
    "relative_motion",            # normalized displacement per elapsed chunk
    "reachability",               # bounded motion/heading reachability proxy
    "key_affinity_proxy",         # cosine of key centroids
    "value_norm",                 # tanh(mean |V| / vnorm_ref)
    "revisit_count",              # log1p(soft pose revisits) / log(16)
    "time_since_last_observed",   # (now - last_observed) / now
]
FEATURE_DIM = len(FEATURE_NAMES)
# v2 keeps identical names/dim so checkpoints stay 11-D MLP-compatible.
FEATURE_NAMES_V2 = list(FEATURE_NAMES)
FEATURE_DIM_V2 = FEATURE_DIM
# Age / motion normalizers for v2 formulas (chunk units / step units).
V2_AGE_TAU = 16.0
V2_V_REF = 1.0
V2_V_MAX = 1.0
V2_EPS = 1e-6

LEGACY_FEATURE_SCHEMA_VERSION = "learned.v0"
_LEGACY_FEATURE_SCHEMA_ALIASES = {"learned.v0", "m1.legacy.v0"}
LEGACY_FEATURE_NAMES = [
    "motion_norm",
    "age_norm",
    "cam_angular",
    "kcent_affinity",
    "value_norm",
]
LEGACY_FEATURE_DIM = len(LEGACY_FEATURE_NAMES)


def rotation_geodesic(Ra: torch.Tensor, Rb: torch.Tensor) -> float:
    """Geodesic distance on SO(3) in radians: arccos((tr(Ra^T Rb) - 1) / 2)."""
    R = Ra.float().T @ Rb.float()
    tr = float(torch.trace(R).clamp(-1.0, 3.0).item())
    cos_theta = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    return float(math.acos(cos_theta))


def se3_distance(
    pose_a: Dict[str, Any],
    pose_b: Dict[str, Any],
    translation_scale: float = 1.0,
    w_trans: float = 1.0,
    w_rot: float = 1.0,
) -> float:
    """Weighted SE(3) distance between two pose dicts."""
    ta = torch.as_tensor(pose_a["translation"], dtype=torch.float32).flatten()
    tb = torch.as_tensor(pose_b["translation"], dtype=torch.float32).flatten()
    d_trans = float(torch.norm(ta - tb).item()) / max(translation_scale, 1e-6)
    Ra = torch.as_tensor(pose_a["rotation"], dtype=torch.float32).reshape(3, 3)
    Rb = torch.as_tensor(pose_b["rotation"], dtype=torch.float32).reshape(3, 3)
    d_rot = rotation_geodesic(Ra, Rb) / math.pi
    return w_trans * d_trans + w_rot * d_rot


def camera_forward_overlap(fa, fb) -> float:
    """Cosine similarity of unit forward vectors; 0 if either is missing."""
    if fa is None or fb is None:
        return 0.0
    a = torch.as_tensor(fa, dtype=torch.float32).flatten()
    b = torch.as_tensor(fb, dtype=torch.float32).flatten()
    if a.numel() != 3 or b.numel() != 3:
        return 0.0
    na, nb = a.norm(), b.norm()
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def _cos(a, b) -> float:
    if a is None or b is None:
        return 0.0
    a = torch.as_tensor(a, dtype=torch.float32).flatten()
    b = torch.as_tensor(b, dtype=torch.float32).flatten()
    if a.numel() == 0 or b.numel() == 0 or a.numel() != b.numel():
        return 0.0
    na, nb = a.norm(), b.norm()
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def _intrinsics4(pose: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    if not pose or pose.get("intrinsics") is None:
        return None
    k = torch.as_tensor(pose["intrinsics"], dtype=torch.float64).flatten()
    if k.numel() == 4:
        fx, fy, cx, cy = (float(x) for x in k)
    elif k.numel() == 9:
        K = k.reshape(3, 3)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    else:
        return None
    if min(fx, fy) <= 1e-8 or min(cx, cy) <= 0:
        return None
    return fx, fy, cx, cy


def _ray_visibility_fraction(
    src: Dict[str, Any],
    dst: Dict[str, Any],
    depths: Sequence[float],
) -> float:
    """Project sparse source-frustum rays into dst; not true visibility/occlusion."""
    Ks, Kd = _intrinsics4(src), _intrinsics4(dst)
    if Ks is None or Kd is None:
        return 0.0
    fsx, fsy, csx, csy = Ks
    fdx, fdy, cdx, cdy = Kd
    Rs = torch.as_tensor(src["rotation"], dtype=torch.float64).reshape(3, 3)
    Rd = torch.as_tensor(dst["rotation"], dtype=torch.float64).reshape(3, 3)
    ts = torch.as_tensor(src["translation"], dtype=torch.float64).flatten()
    td = torch.as_tensor(dst["translation"], dtype=torch.float64).flatten()
    pixels = [
        (csx, csy), (0.0, csy), (2 * csx, csy), (csx, 0.0), (csx, 2 * csy),
        (0.0, 0.0), (2 * csx, 0.0), (0.0, 2 * csy), (2 * csx, 2 * csy),
    ]
    hit, total = 0, 0
    for u, v in pixels:
        ray = torch.tensor([(u - csx) / fsx, (v - csy) / fsy, 1.0], dtype=torch.float64)
        ray = ray / ray.norm().clamp_min(1e-12)
        for depth in depths:
            world = Rs @ (ray * float(depth)) + ts
            cam = Rd.T @ (world - td)
            total += 1
            if float(cam[2]) <= 1e-8:
                continue
            ud = fdx * float(cam[0] / cam[2]) + cdx
            vd = fdy * float(cam[1] / cam[2]) + cdy
            eps = 1e-6
            if -eps <= ud <= 2 * cdx + eps and -eps <= vd <= 2 * cdy + eps:
                hit += 1
    return hit / max(total, 1)


def frustum_overlap_proxy(
    pose_a: Optional[Dict[str, Any]],
    pose_b: Optional[Dict[str, Any]],
    translation_scale: float = 1.0,
) -> float:
    """Symmetric sparse-frustum overlap proxy from pose and intrinsics."""
    if pose_a is None or pose_b is None:
        return 0.0
    if any(k not in pose_a or k not in pose_b for k in ("rotation", "translation")):
        return 0.0
    s = max(float(translation_scale), 1e-6)
    depths = (2.0 * s, 5.0 * s, 10.0 * s)
    return 0.5 * (
        _ray_visibility_fraction(pose_a, pose_b, depths)
        + _ray_visibility_fraction(pose_b, pose_a, depths)
    )


class ChunkSelector(nn.Module):
    """Small MLP mapping per-chunk features -> scalar retention utility."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 32, depth: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))

    def set_norm(self, mean: torch.Tensor, std: torch.Tensor):
        self.feat_mean.copy_(mean.detach().float())
        self.feat_std.copy_(std.detach().float().clamp_min(1e-6))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: [N, in_dim] -> utility [N]."""
        feats = (feats - self.feat_mean) / self.feat_std
        return self.net(feats).squeeze(-1)


def build_chunk_features(
    cand: dict,
    cur: dict,
    gen_chunk_id: int,
    motion_ref: float,
    vnorm_ref: float,
    translation_scale: float = 1.0,
) -> list[float]:
    """Build the ordered FEATURE_NAMES row for one archive candidate.

    Shared by training (train_selector.py) and inference
    (image2video_fast.py::_score_archive_learned).

    cand/cur may include: chunk_id, motion_score, camera_forward, k_centroid,
    value_norm, pose{translation,rotation,intrinsics}, revisit_count, last_observed.
    """
    motion = cand.get("motion_score", 0.0)
    ref = motion_ref if motion_ref and motion_ref > 1e-8 else 1.0
    motion_norm = 1.0 if not math.isfinite(motion) else math.tanh(motion / ref)

    pose_c = cand.get("pose")
    pose_q = cur.get("pose")
    scale = max(float(translation_scale), 1e-6)
    if pose_c is not None and pose_q is not None:
        ta = torch.as_tensor(pose_c["translation"], dtype=torch.float32).flatten()
        tb = torch.as_tensor(pose_q["translation"], dtype=torch.float32).flatten()
        trans_dist = float(torch.norm(ta - tb).item()) / scale
        Ra = torch.as_tensor(pose_c["rotation"], dtype=torch.float32).reshape(3, 3)
        Rb = torch.as_tensor(pose_q["rotation"], dtype=torch.float32).reshape(3, 3)
        rot_dist = rotation_geodesic(Ra, Rb) / math.pi
    else:
        ang = 1.0 - _cos(cand.get("camera_forward"), cur.get("camera_forward"))
        if cand.get("camera_forward") is None or cur.get("camera_forward") is None:
            ang = 0.0
        trans_dist = 0.0
        rot_dist = ang / 2.0

    forward_overlap = camera_forward_overlap(
        cand.get("camera_forward"), cur.get("camera_forward"))
    frustum = frustum_overlap_proxy(pose_c, pose_q, scale)
    last_observed = int(cand.get("last_observed", cand.get("chunk_id", gen_chunk_id)))
    elapsed = max(1, int(gen_chunk_id) - last_observed)
    # v1 formulas (frozen for selector_ws_v1.pt). Do not change.
    relative_motion = math.tanh(trans_dist / elapsed)
    reachability = math.exp(-trans_dist / elapsed) * max(0.0, 0.5 * (forward_overlap + 1.0))

    # key_affinity / value_norm: collected at layer-0 only (see image2video_fast).
    kcent_affinity = _cos(cand.get("k_centroid"), cur.get("k_centroid"))
    vref = vnorm_ref if vnorm_ref and vnorm_ref > 1e-8 else 1.0
    # value_norm meta = mean_{token,head} |V| at layer 0.
    value_norm = math.tanh(float(cand.get("value_norm", 0.0)) / vref)
    revisit_count = max(0, int(cand.get("revisit_count", int(bool(cand.get("revisited", False))))))
    revisit_norm = math.log1p(revisit_count) / math.log(16.0)
    time_since_observed = max(0, int(gen_chunk_id) - last_observed) / max(1, int(gen_chunk_id))

    return [
        motion_norm, trans_dist, rot_dist, forward_overlap, frustum,
        relative_motion, reachability, kcent_affinity, value_norm,
        revisit_norm, time_since_observed,
    ]


def build_chunk_features_v2(
    cand: dict,
    cur: dict,
    gen_chunk_id: int,
    motion_ref: float,
    vnorm_ref: float,
    translation_scale: float = 1.0,
    *,
    v_ref: float = V2_V_REF,
    v_max: float = V2_V_MAX,
    age_tau: float = V2_AGE_TAU,
    eps: float = V2_EPS,
) -> list[float]:
    """P1 feature formulas (schema ``world_state.v2``).

    Same 11-D layout as v1, but:
      relative_motion = tanh(d / (v_ref * Δt + eps))
      reachability    = exp(-d / (v_max * Δt + eps)) * (1+cos)/2
      time_since      = tanh((t - t_last) / tau)
    ``d`` is translation distance in trajectory-step units (already / scale).
    ``key_affinity_proxy`` / ``value_norm`` still use layer-0 summaries stored
    at collection time (no full-layer scan).
    """
    motion = cand.get("motion_score", 0.0)
    ref = motion_ref if motion_ref and motion_ref > 1e-8 else 1.0
    motion_norm = 1.0 if not math.isfinite(motion) else math.tanh(motion / ref)

    pose_c = cand.get("pose")
    pose_q = cur.get("pose")
    scale = max(float(translation_scale), 1e-6)
    if pose_c is not None and pose_q is not None:
        ta = torch.as_tensor(pose_c["translation"], dtype=torch.float32).flatten()
        tb = torch.as_tensor(pose_q["translation"], dtype=torch.float32).flatten()
        trans_dist = float(torch.norm(ta - tb).item()) / scale
        Ra = torch.as_tensor(pose_c["rotation"], dtype=torch.float32).reshape(3, 3)
        Rb = torch.as_tensor(pose_q["rotation"], dtype=torch.float32).reshape(3, 3)
        rot_dist = rotation_geodesic(Ra, Rb) / math.pi
    else:
        ang = 1.0 - _cos(cand.get("camera_forward"), cur.get("camera_forward"))
        if cand.get("camera_forward") is None or cur.get("camera_forward") is None:
            ang = 0.0
        trans_dist = 0.0
        rot_dist = ang / 2.0

    forward_overlap = camera_forward_overlap(
        cand.get("camera_forward"), cur.get("camera_forward"))
    frustum = frustum_overlap_proxy(pose_c, pose_q, scale)
    last_observed = int(cand.get("last_observed", cand.get("chunk_id", gen_chunk_id)))
    elapsed = max(1, int(gen_chunk_id) - last_observed)
    relative_motion = math.tanh(trans_dist / (float(v_ref) * elapsed + float(eps)))
    reachability = (
        math.exp(-trans_dist / (float(v_max) * elapsed + float(eps)))
        * max(0.0, 0.5 * (forward_overlap + 1.0))
    )

    kcent_affinity = _cos(cand.get("k_centroid"), cur.get("k_centroid"))
    vref = vnorm_ref if vnorm_ref and vnorm_ref > 1e-8 else 1.0
    value_norm = math.tanh(float(cand.get("value_norm", 0.0)) / vref)
    revisit_count = max(0, int(cand.get("revisit_count", int(bool(cand.get("revisited", False))))))
    revisit_norm = math.log1p(revisit_count) / math.log(16.0)
    age = max(0, int(gen_chunk_id) - last_observed)
    time_since_observed = math.tanh(age / max(float(age_tau), float(eps)))

    return [
        motion_norm, trans_dist, rot_dist, forward_overlap, frustum,
        relative_motion, reachability, kcent_affinity, value_norm,
        revisit_norm, time_since_observed,
    ]


def build_legacy_chunk_features(
    cand: dict,
    cur: dict,
    gen_chunk_id: int,
    motion_ref: float,
    vnorm_ref: float,
    translation_scale: float = 1.0,
) -> list[float]:
    """Original 5-D Learned CR feature row (frozen baseline vs world-state)."""
    del translation_scale  # unused; signature matches build_chunk_features
    motion = cand.get("motion_score", 0.0)
    ref = motion_ref if motion_ref and motion_ref > 1e-8 else 1.0
    motion_norm = 1.0 if not math.isfinite(motion) else math.tanh(motion / ref)
    age = max(0, int(gen_chunk_id) - int(cand.get("chunk_id", gen_chunk_id)))
    age_norm = age / max(1, int(gen_chunk_id))
    cam_angular = 1.0 - _cos(cand.get("camera_forward"), cur.get("camera_forward"))
    if cand.get("camera_forward") is None or cur.get("camera_forward") is None:
        cam_angular = 0.0
    kcent_affinity = _cos(cand.get("k_centroid"), cur.get("k_centroid"))
    vref = vnorm_ref if vnorm_ref and vnorm_ref > 1e-8 else 1.0
    value_norm = math.tanh(float(cand.get("value_norm", 0.0)) / vref)
    return [motion_norm, age_norm, cam_angular, kcent_affinity, value_norm]


def features_for_schema(
    feature_names: Sequence[str],
    cand: dict,
    cur: dict,
    gen_chunk_id: int,
    motion_ref: float,
    vnorm_ref: float,
    translation_scale: float = 1.0,
    schema_version: str | None = None,
) -> list[float]:
    names = list(feature_names)
    schema = schema_version or FEATURE_SCHEMA_VERSION
    if names == LEGACY_FEATURE_NAMES:
        return build_legacy_chunk_features(
            cand, cur, gen_chunk_id, motion_ref, vnorm_ref, translation_scale)
    if names == FEATURE_NAMES or names == FEATURE_NAMES_V2:
        if schema in _FEATURE_SCHEMA_V2_ALIASES:
            return build_chunk_features_v2(
                cand, cur, gen_chunk_id, motion_ref, vnorm_ref, translation_scale)
        return build_chunk_features(
            cand, cur, gen_chunk_id, motion_ref, vnorm_ref, translation_scale)
    raise ValueError(f"unsupported selector feature schema: {names}")


def save_selector(
    model: ChunkSelector,
    path: str,
    meta: dict | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    schema_version: str | None = None,
):
    names = list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
    schema = schema_version or FEATURE_SCHEMA_VERSION
    # Infer hidden width from first Linear weight if present.
    hidden = 32
    sd = model.state_dict()
    w0 = sd.get("net.0.weight")
    if w0 is not None and w0.ndim == 2:
        hidden = int(w0.shape[0])
    payload = {
        "state_dict": sd,
        "in_dim": model.in_dim,
        "hidden": hidden,
        "feature_names": names,
        "schema_version": schema,
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_selector(path: str, map_location="cpu") -> ChunkSelector:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    names = list(payload.get("feature_names", FEATURE_NAMES))
    raw_schema = payload.get("schema_version", FEATURE_SCHEMA_VERSION)
    if names == FEATURE_NAMES or names == FEATURE_NAMES_V2:
        if raw_schema in _FEATURE_SCHEMA_V2_ALIASES:
            schema = FEATURE_SCHEMA_VERSION_V2
        elif raw_schema in _FEATURE_SCHEMA_ALIASES:
            schema = FEATURE_SCHEMA_VERSION
        else:
            # Unknown alias on 11-D: default to v1 for back-compat.
            schema = FEATURE_SCHEMA_VERSION
        in_dim = payload.get("in_dim", FEATURE_DIM)
    elif names == LEGACY_FEATURE_NAMES:
        schema = payload.get("schema_version", LEGACY_FEATURE_SCHEMA_VERSION)
        if schema not in _LEGACY_FEATURE_SCHEMA_ALIASES:
            schema = LEGACY_FEATURE_SCHEMA_VERSION
        else:
            schema = LEGACY_FEATURE_SCHEMA_VERSION
        in_dim = payload.get("in_dim", LEGACY_FEATURE_DIM)
    else:
        raise AssertionError(
            f"selector checkpoint feature schema {names} unsupported; "
            f"expected world-state {FEATURE_NAMES} or learned {LEGACY_FEATURE_NAMES}."
        )
    hidden = int(payload.get("hidden", 32))
    if "hidden" not in payload:
        w0 = payload.get("state_dict", {}).get("net.0.weight")
        if w0 is not None and getattr(w0, "ndim", 0) == 2:
            hidden = int(w0.shape[0])
    model = ChunkSelector(in_dim=in_dim, hidden=hidden)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model._meta = payload.get("meta", {}) or {}
    model._schema_version = schema
    model._feature_names = names
    return model
