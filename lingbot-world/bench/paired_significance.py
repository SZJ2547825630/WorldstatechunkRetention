#!/usr/bin/env python3
"""Paired significance testing on per-clip metrics from eval_worldkv_memory.py dumps.

Clips are the resampling unit: every method sees the same clips and the same
revisit pairs, so differences are paired and a clip-level bootstrap / signed-rank
test is the right instrument. Set-level metrics (FID) have no per-clip value and
are therefore skipped.

Example:
  python bench/paired_significance.py \
      --eval_json output/.../worldkv_memory_eval_default_all.json \
      --clips bench/realcamvid/subsets/default_loop.txt \
      --pairs world_state_cr_future:world_state_cr world_state_cr_future:window
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

# Metrics where a smaller value is better.
LOWER_BETTER = {"lpips", "fid", "mean_pose_dist"}
DEFAULT_METRICS = ("psnr", "ssim", "lpips")


def load_per_clip(paths: Sequence[str]) -> Dict[str, Dict[str, dict]]:
    """clip_id -> method -> metric dict, merged across eval dumps."""
    merged: Dict[str, Dict[str, dict]] = {}
    for path in paths:
        with open(path) as fh:
            data = json.load(fh)
        for clip, rec in (data.get("per_clip") or {}).items():
            slot = merged.setdefault(clip, {})
            for method, metrics in (rec.get("methods") or {}).items():
                slot.setdefault(method, metrics)
    return merged


def read_clip_list(path: str) -> List[str]:
    ids = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(os.path.splitext(os.path.basename(line))[0])
    return ids


def paired_bootstrap(delta: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, delta.size, size=(n_boot, delta.size))
    means = delta[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze(
    per_clip: Dict[str, Dict[str, dict]],
    clips: Sequence[str],
    method: str,
    baseline: str,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict | None:
    a, b, weights, used = [], [], [], []
    for clip in clips:
        slot = per_clip.get(clip) or {}
        m_rec, b_rec = slot.get(method), slot.get(baseline)
        if not m_rec or not b_rec:
            continue
        if m_rec.get(metric) is None or b_rec.get(metric) is None:
            continue
        a.append(float(m_rec[metric]))
        b.append(float(b_rec[metric]))
        weights.append(float(min(m_rec.get("n_pairs", 1), b_rec.get("n_pairs", 1))))
        used.append(clip)
    if len(used) < 3:
        return None

    arr_a, arr_b, w = np.array(a), np.array(b), np.array(weights)
    delta = arr_a - arr_b
    sign = -1.0 if metric in LOWER_BETTER else 1.0
    gain = sign * delta  # positive == method is better

    lo, hi = paired_bootstrap(delta, n_boot, seed)
    wins = int((gain > 0).sum())
    losses = int((gain < 0).sum())
    ties = int((gain == 0).sum())
    nonzero = wins + losses

    try:
        w_stat = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        p_wilcoxon = float(w_stat.pvalue)
    except ValueError:  # all-zero differences
        p_wilcoxon = 1.0
    p_sign = float(stats.binomtest(wins, nonzero, 0.5).pvalue) if nonzero else 1.0
    p_ttest = float(stats.ttest_rel(arr_a, arr_b).pvalue)
    sd = float(delta.std(ddof=1))
    # Smallest paired difference this many clips could detect at alpha=.05, power=.8.
    mde80 = 2.802 * sd / np.sqrt(len(used))
    return {
        "metric": metric,
        "n_clips": len(used),
        "mean_method": float(arr_a.mean()),
        "mean_baseline": float(arr_b.mean()),
        "mean_delta": float(delta.mean()),
        "ci95": (lo, hi),
        "delta_pair_weighted": float((delta * w).sum() / w.sum()),
        "cohens_dz": float(delta.mean() / sd) if sd > 0 else 0.0,
        "mde80": float(mde80),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "p_wilcoxon": p_wilcoxon,
        "p_sign": p_sign,
        "p_ttest": p_ttest,
        "better_if_positive": sign > 0,
        "significant": (lo > 0 and hi > 0) or (lo < 0 and hi < 0),
    }


def fmt(v: float, nd: int) -> str:
    return f"{v:+.{nd}f}" if abs(v) < 1e4 else f"{v:+.3e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", nargs="+", required=True)
    ap.add_argument("--clips", default=None, help="file with clip ids to restrict to")
    ap.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="method:baseline pairs, e.g. world_state_cr_future:window",
    )
    ap.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_json", default=None)
    args = ap.parse_args()

    per_clip = load_per_clip(args.eval_json)
    clips = read_clip_list(args.clips) if args.clips else sorted(per_clip)
    clips = [c for c in clips if c in per_clip]
    print(f"[sig] clips={len(clips)} boot={args.n_boot} sources={len(args.eval_json)}")

    results = {}
    for pair in args.pairs:
        method, baseline = pair.split(":", 1)
        rows = []
        for metric in args.metrics:
            row = analyze(
                per_clip, clips, method, baseline, metric, args.n_boot, args.seed
            )
            if row:
                rows.append(row)
        if not rows:
            print(f"[sig] no paired data for {pair}")
            continue
        results[pair] = rows
        print(f"\n### {method} vs {baseline}  (n={rows[0]['n_clips']} clips)")
        print(
            "| metric | method | baseline | Δ (mean) | 95% CI (bootstrap) | "
            "win/loss/tie | dz | p(Wilcoxon) | p(sign) | MDE@80% | verdict |"
        )
        print("|---|---:|---:|---:|:--:|:--:|---:|---:|---:|---:|:--|")
        for r in rows:
            nd = 3 if r["metric"] == "psnr" else 4
            lo, hi = r["ci95"]
            better = (r["mean_delta"] > 0) == r["better_if_positive"]
            if r["significant"]:
                verdict = "significant win" if better else "significant loss"
            else:
                verdict = "no difference"
            print(
                f"| {r['metric']} | {r['mean_method']:.{nd}f} | {r['mean_baseline']:.{nd}f} | "
                f"{fmt(r['mean_delta'], nd)} | [{fmt(lo, nd)}, {fmt(hi, nd)}] | "
                f"{r['wins']}/{r['losses']}/{r['ties']} | {r['cohens_dz']:+.2f} | "
                f"{r['p_wilcoxon']:.4f} | {r['p_sign']:.4f} | "
                f"{r['mde80']:.{nd}f} | {verdict} |"
            )

    if args.save_json:
        with open(args.save_json, "w") as fh:
            json.dump({"clips": clips, "results": results}, fh, indent=2)
        print(f"\n[sig] saved -> {args.save_json}")


if __name__ == "__main__":
    main()
