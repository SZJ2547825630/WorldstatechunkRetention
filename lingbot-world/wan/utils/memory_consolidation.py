"""Memory Consolidation for World-State CR.

Distinct from WorldKV (pose-top-k external bank + anchor/novelty prune):
this module manages the *lifecycle* of in-cache archive segments —

* C1 — EMA utility + hysteresis (scores have memory)
* C2 — tiered demotion L0→L1→L2→L3 instead of binary discard
* revisit-coverage probe (Phase-0 diagnostic)

C3 (cross-chunk merge) is intentionally deferred; RoPE/index semantics need a
dedicated pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


@dataclass
class ConsolidationConfig:
    mode: str = "off"  # off | ema | full
    beta: float = 0.7
    patience: int = 2
    stabilize_thr: float = 0.6  # EMA above this for `patience` steps → stabilized
    stabilize_bonus: float = 0.05
    rank_alpha: float = 0.5
    # Among kept archive chunks, demote the bottom this fraction to L2 (gist).
    # Fixes the old low_streak trigger that almost never fired for kept chunks.
    # 0 → never force L2 via rank position; 0.5 → bottom half of kept → L2.
    l2_bottom_ratio: float = 0.5
    gist_tokens: int = 64
    gist_budget_tokens: int = 512  # hard cap on total L2 tokens across archive
    swtp_keep_ratio: float = 0.5
    swtp_num_summary: int = 64
    swtp_energy_cover: float = 0.9

    @property
    def enabled(self) -> bool:
        return self.mode not in (None, "", "off", "none", "false", "0")

    @property
    def tiers_enabled(self) -> bool:
        return self.mode == "full"


def update_utility_ema(
    meta: Dict[str, Any],
    score: float,
    *,
    beta: float,
    patience: int,
    stabilize_thr: float,
    keep_threshold: Optional[float] = None,
) -> float:
    """Update per-chunk EMA utility and hysteresis counters in ``meta``."""
    prev = meta.get("u_ema")
    if prev is None:
        u = float(score)
    else:
        u = float(beta) * float(prev) + (1.0 - float(beta)) * float(score)
    meta["u_ema"] = u

    # Stabilization: consecutive high-EMA steps.
    if u >= float(stabilize_thr):
        meta["high_streak"] = int(meta.get("high_streak", 0)) + 1
    else:
        meta["high_streak"] = 0
    if int(meta.get("high_streak", 0)) >= int(patience):
        meta["stabilized"] = True

    # Demotion patience: consecutive steps below the keep threshold.
    if keep_threshold is not None and u < float(keep_threshold):
        meta["low_streak"] = int(meta.get("low_streak", 0)) + 1
    else:
        meta["low_streak"] = 0
    return u


def rank_score(seg: Dict[str, Any], meta: Optional[Dict[str, Any]], cfg: ConsolidationConfig) -> float:
    """Archive ranking key under consolidation.

    Mixes instantaneous selector score ``s`` with EMA utility:
    ``α·s + (1-α)·u_ema`` (plus stabilize bonus on the EMA term path).
    """
    base = float(seg.get("_sel_score", seg.get("motion_score", 0.0)))
    if not cfg.enabled or meta is None:
        return base
    u = meta.get("u_ema", base)
    u = float(u) if u is not None else base
    if meta.get("stabilized"):
        u = u + float(cfg.stabilize_bonus)
    alpha = float(getattr(cfg, "rank_alpha", 0.0) or 0.0)
    alpha = max(0.0, min(1.0, alpha))
    return float(alpha) * base + (1.0 - float(alpha)) * u


def assign_archive_tiers(
    archive_segments: Sequence[Dict[str, Any]],
    *,
    keep_count: int,
    chunk_meta: Dict[int, Dict[str, Any]],
    cfg: ConsolidationConfig,
) -> Dict[int, str]:
    """Map chunk_id → tier label among {L1, L2, L3}.

    L0 is reserved for sink/recent (caller never passes those here).
    With ``mode=ema`` everyone that would have been kept stays L1 and the rest
    L3 (behaviourally close to v2, but ranked by EMA mix).
    With ``mode=full``, the bottom ``l2_bottom_ratio`` of the kept set demotes
    to L2 (gist) unless stabilized; overflow beyond keep_count goes L3.
    ``low_streak ≥ patience`` remains an additional L2 trigger.
    """
    if not archive_segments:
        return {}
    ranked = sorted(
        archive_segments,
        key=lambda s: rank_score(s, chunk_meta.get(int(s["chunk_id"])), cfg),
        reverse=True,
    )
    tiers: Dict[int, str] = {}
    kept = ranked[: max(0, int(keep_count))]
    dropped = ranked[max(0, int(keep_count)):]

    if not cfg.tiers_enabled:
        for s in kept:
            tiers[int(s["chunk_id"])] = "L1"
        for s in dropped:
            tiers[int(s["chunk_id"])] = "L3"
        return tiers

    n_kept = len(kept)
    ratio = float(getattr(cfg, "l2_bottom_ratio", 0.5) or 0.0)
    ratio = max(0.0, min(1.0, ratio))
    n_l2 = int(math.ceil(n_kept * ratio)) if ratio > 0 and n_kept > 0 else 0
    # Never demote the entire kept set to gist if there is more than one keep.
    if n_kept > 1:
        n_l2 = min(n_l2, n_kept - 1)
    l2_ids = {int(s["chunk_id"]) for s in kept[n_kept - n_l2:]} if n_l2 > 0 else set()

    for s in kept:
        cid = int(s["chunk_id"])
        meta = chunk_meta.get(cid, {})
        already = s.get("memory_tier") or s.get("is_swtp") and s.get("is_gist")
        if meta.get("stabilized"):
            tiers[cid] = "L1"
        elif cid in l2_ids:
            tiers[cid] = "L2"
        elif int(meta.get("low_streak", 0)) >= int(cfg.patience):
            tiers[cid] = "L2"
        elif s.get("is_gist"):
            tiers[cid] = "L2"
        elif already == "L2":
            tiers[cid] = "L2"
        else:
            tiers[cid] = "L1"
    for s in dropped:
        tiers[int(s["chunk_id"])] = "L3"
    return tiers


def enforce_gist_budget(
    tiers: Dict[int, str],
    archive_segments: Sequence[Dict[str, Any]],
    chunk_meta: Dict[int, Dict[str, Any]],
    cfg: ConsolidationConfig,
) -> Dict[int, str]:
    """If too many L2 segments, demote the weakest to L3 until budget fits.

    Budget is counted in *target* gist tokens (``gist_tokens`` each), not the
    pre-compression size — we don't want a transient spike to force deletes.
    """
    if cfg.gist_budget_tokens <= 0:
        return tiers
    l2 = [s for s in archive_segments if tiers.get(int(s["chunk_id"])) == "L2"]
    if not l2:
        return tiers
    capacity = max(1, int(cfg.gist_budget_tokens) // max(1, int(cfg.gist_tokens)))
    if len(l2) <= capacity:
        return tiers
    l2_sorted = sorted(
        l2,
        key=lambda s: rank_score(s, chunk_meta.get(int(s["chunk_id"])), cfg),
    )
    out = dict(tiers)
    for s in l2_sorted[: max(0, len(l2) - capacity)]:
        out[int(s["chunk_id"])] = "L3"
    return out


def update_revisit_coverage(
    stats: Dict[str, Any],
    revisited_ids: Iterable[int],
    retained_ids: Set[int],
) -> None:
    """Phase-0 probe: was a revisited chunk still in the KV cache?"""
    hits = stats.setdefault("revisit_coverage_hits", 0)
    misses = stats.setdefault("revisit_coverage_misses", 0)
    total = stats.setdefault("revisit_events", 0)
    for cid in revisited_ids:
        total += 1
        if int(cid) in retained_ids:
            hits += 1
        else:
            misses += 1
    stats["revisit_coverage_hits"] = hits
    stats["revisit_coverage_misses"] = misses
    stats["revisit_events"] = total
    if total > 0:
        stats["revisit_coverage"] = hits / total
