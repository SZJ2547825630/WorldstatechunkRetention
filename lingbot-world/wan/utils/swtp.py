"""Saliency-Weighted Token Pruning helpers.

Spatial-cell pooling + energy-coverage keep selection replace the original
raster-strip mean-pool (which averaged tokens across a full image row and
produced near-dead summary keys).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch

TokenGrid = Tuple[int, int, int]  # (F, Ht, Wt)


def infer_token_grid(num_tokens: int, prefer: Optional[TokenGrid] = None) -> TokenGrid:
    """Recover (F, Ht, Wt) for a flattened token sequence.

    Prefers an explicit grid when it matches ``num_tokens``. Otherwise assumes
    the standard 3-frame latent chunk and factorises the remainder into a near
    16:9 token grid (matches 480x832 → 30x52).
    """
    if prefer is not None:
        f, h, w = prefer
        if f * h * w == num_tokens:
            return prefer
    for f in (3, 1, 2, 4, 5):
        if num_tokens % f:
            continue
        spatial = num_tokens // f
        # Prefer factors near 30x52 / 16:9.
        best = None
        for h in range(int(math.sqrt(spatial)), 0, -1):
            if spatial % h == 0:
                w = spatial // h
                score = abs((w / max(h, 1)) - (52 / 30))
                if best is None or score < best[0]:
                    best = (score, h, w)
        if best is not None:
            return f, best[1], best[2]
    return 1, 1, num_tokens


def choose_keep_count(
    saliency: torch.Tensor,
    keep_ratio: float,
    energy_cover: float = 0.9,
) -> int:
    """Smallest K ≤ keep_ratio·T whose saliency mass covers ``energy_cover``.

    Concentrated saliency → aggressive prune; diffuse → keep up to the ratio cap.
    """
    t = int(saliency.numel())
    if t <= 0:
        return 0
    k_cap = max(1, min(t, int(math.ceil(t * float(keep_ratio)))))
    if energy_cover is None or energy_cover <= 0.0 or energy_cover >= 1.0:
        return k_cap
    s = saliency.detach().float().flatten()
    total = float(s.sum().item())
    if total < 1e-12:
        return k_cap
    ordered = torch.sort(s, descending=True).values
    cum = torch.cumsum(ordered, dim=0) / total
    hits = (cum >= float(energy_cover)).nonzero(as_tuple=True)[0]
    k = int(hits[0].item()) + 1 if hits.numel() else k_cap
    return max(1, min(k_cap, k))


def gini_coefficient(x: torch.Tensor) -> float:
    x = x.flatten().float()
    if x.numel() < 2 or float(x.sum().item()) < 1e-12:
        return 0.0
    sorted_x = torch.sort(x).values
    n = sorted_x.numel()
    cum = torch.cumsum(sorted_x, dim=0) / sorted_x.sum()
    return float(1.0 - 2.0 * cum.sum().item() / n + 1.0 / n)


def _lattice_keep_indices(grid: TokenGrid, stride_h: int = 2, stride_w: int = 2) -> torch.Tensor:
    f, ht, wt = grid
    idx = []
    for fi in range(f):
        base = fi * ht * wt
        for h in range(0, ht, stride_h):
            for w in range(0, wt, stride_w):
                idx.append(base + h * wt + w)
    return torch.tensor(idx, dtype=torch.long)


def _spatial_summary_groups(
    drop_idx: torch.Tensor,
    grid: TokenGrid,
    num_summary: int,
) -> list[torch.Tensor]:
    """Partition dropped token indices into compact spatial cells (one frame)."""
    if drop_idx.numel() == 0 or num_summary <= 0:
        return []
    f, ht, wt = grid
    # Choose a cell grid whose capacity is ≈ num_summary.
    cells_per_frame = max(1, int(math.ceil(num_summary / max(f, 1))))
    aspect = wt / max(ht, 1)
    n_h = max(1, int(round(math.sqrt(cells_per_frame / max(aspect, 1e-6)))))
    n_w = max(1, int(math.ceil(cells_per_frame / n_h)))
    cell_h = max(1, int(math.ceil(ht / n_h)))
    cell_w = max(1, int(math.ceil(wt / n_w)))

    groups: list[torch.Tensor] = []
    drop_cpu = drop_idx.detach().to("cpu")
    for fi in range(f):
        in_f = drop_cpu[(drop_cpu // (ht * wt)) == fi]
        if in_f.numel() == 0:
            continue
        local = in_f % (ht * wt)
        hh, ww = local // wt, local % wt
        for h0 in range(0, ht, cell_h):
            for w0 in range(0, wt, cell_w):
                sel = in_f[(hh >= h0) & (hh < h0 + cell_h) & (ww >= w0) & (ww < w0 + cell_w)]
                if sel.numel():
                    groups.append(sel)
    if not groups:
        return []
    # If we overshot num_summary, keep the densest cells (most dropped tokens).
    if len(groups) > num_summary:
        groups = sorted(groups, key=lambda g: int(g.numel()), reverse=True)[:num_summary]
    return groups


def _norm_compensate(summary: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Rescale summary keys so mean ‖k‖ matches the kept-token reference."""
    if summary.numel() == 0 or reference.numel() == 0:
        return summary
    ref_norm = reference.float().norm(dim=-1).mean().clamp_min(1e-6)
    sum_norm = summary.float().norm(dim=-1).mean().clamp_min(1e-6)
    scale = (ref_norm / sum_norm).to(dtype=summary.dtype)
    return summary * scale


@torch.no_grad()
def apply_swtp_to_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    saliency: torch.Tensor,
    keep_ratio: float = 0.5,
    num_summary: int = 64,
    token_grid: Optional[TokenGrid] = None,
    energy_cover: float = 0.9,
    mode: str = "standard",
    compensate_summary_norm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prune ``[B, T, H, D]`` K/V along the token axis.

    Modes
    -----
    standard : top-K by saliency (energy-cover capped by keep_ratio) + spatial summaries
    gist     : summary tokens only (Memory Consolidation L2)
    uniform  : spatial lattice keep (low-Gini fallback) + spatial summaries
    """
    assert k.shape[1] == v.shape[1]
    t = int(saliency.numel())
    assert t == int(k.shape[1]), f"saliency length {t} != KV tokens {k.shape[1]}"
    grid = infer_token_grid(t, prefer=token_grid)
    device = k.device
    sal = saliency.to(device=device, dtype=torch.float32).flatten()

    mode = (mode or "standard").lower()
    if mode == "gist":
        keep_idx = torch.empty(0, dtype=torch.long, device=device)
    elif mode == "uniform":
        keep_idx = _lattice_keep_indices(grid).to(device)
        # Cap lattice density by keep_ratio so uniform never exceeds the budget.
        k_cap = max(1, min(t, int(math.ceil(t * float(keep_ratio)))))
        if keep_idx.numel() > k_cap:
            # Subsample lattice evenly.
            step = max(1, int(math.ceil(keep_idx.numel() / k_cap)))
            keep_idx = keep_idx[::step][:k_cap]
    else:
        k_keep = choose_keep_count(sal, keep_ratio=keep_ratio, energy_cover=energy_cover)
        keep_idx = sal.topk(k_keep).indices.sort().values

    keep_mask = torch.zeros(t, dtype=torch.bool, device=device)
    if keep_idx.numel():
        keep_mask[keep_idx] = True
        k_kept = k.index_select(1, keep_idx)
        v_kept = v.index_select(1, keep_idx)
    else:
        k_kept = k[:, :0]
        v_kept = v[:, :0]

    parts_k: list[torch.Tensor] = [k_kept] if k_kept.shape[1] else []
    parts_v: list[torch.Tensor] = [v_kept] if v_kept.shape[1] else []

    if num_summary > 0 and (~keep_mask).any():
        drop_idx = (~keep_mask).nonzero(as_tuple=True)[0]
        groups = _spatial_summary_groups(drop_idx.detach().cpu(), grid, num_summary)
        if groups:
            # Move group indices to device once.
            k_sums, v_sums = [], []
            for g in groups:
                gi = g.to(device=device, dtype=torch.long)
                k_sums.append(k.index_select(1, gi).mean(dim=1, keepdim=True))
                v_sums.append(v.index_select(1, gi).mean(dim=1, keepdim=True))
            k_summary = torch.cat(k_sums, dim=1)
            v_summary = torch.cat(v_sums, dim=1)
            if compensate_summary_norm and k_kept.shape[1] > 0:
                k_summary = _norm_compensate(k_summary, k_kept)
            elif compensate_summary_norm and k_kept.shape[1] == 0:
                # Gist-only: compensate against pre-prune mean key norm.
                k_summary = _norm_compensate(k_summary, k)
            parts_k.append(k_summary)
            parts_v.append(v_summary)
        elif drop_idx.numel():
            # Degenerate grid: fall back to contiguous pooling (legacy).
            t_dropped = int(drop_idx.numel())
            m = min(num_summary, t_dropped)
            group = max(1, t_dropped // m)
            usable = m * group
            kd = k.index_select(1, drop_idx[:usable])
            vd = v.index_select(1, drop_idx[:usable])
            k_summary = kd.view(k.shape[0], m, group, *k.shape[2:]).mean(dim=2)
            v_summary = vd.view(v.shape[0], m, group, *v.shape[2:]).mean(dim=2)
            if compensate_summary_norm and k_kept.shape[1] > 0:
                k_summary = _norm_compensate(k_summary, k_kept)
            parts_k.append(k_summary)
            parts_v.append(v_summary)

    if not parts_k:
        # Absolute fallback: keep a single mean token so the segment stays valid.
        k_reduced = k.mean(dim=1, keepdim=True)
        v_reduced = v.mean(dim=1, keepdim=True)
        return k_reduced, v_reduced, keep_idx

    k_reduced = torch.cat(parts_k, dim=1) if len(parts_k) > 1 else parts_k[0]
    v_reduced = torch.cat(parts_v, dim=1) if len(parts_v) > 1 else parts_v[0]
    return k_reduced, v_reduced, keep_idx.detach().cpu()
