#!/usr/bin/env python3
"""Clip-level paired bootstrap confidence intervals for the revisit FID.

FID is a set-level statistic, so eval_worldkv_memory.py reports a single number
per method with no per-clip value to test. This script re-extracts the same
Inception features the eval uses, caches them per clip, and then resamples clips
with replacement -- using the *same* clip draw for every method -- so the FID
difference between two methods can be given a confidence interval.

Example:
  python bench/fid_bootstrap.py \
      --out_dir output/realcamvid_ws_vs_window_default_all_v2 \
      --clips_dir bench/realcamvid/clips_default_all \
      --methods window,world_state_cr,world_state_cr_future \
      --pairs world_state_cr_future:world_state_cr world_state_cr_future:window
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_worldkv_memory import (  # noqa: E402
    InceptionFeatures,
    fid_from_features,
    find_revisit_pairs,
    read_video_frames,
    scene_translation_scale,
    subsample_pairs,
    video_num_frames,
)


def extract_features(args, clips: List[str], methods: List[str]) -> Dict[str, dict]:
    out_dir = Path(args.out_dir).resolve()
    videos_dir = out_dir / "videos"
    clips_dir = Path(args.clips_dir)
    inception = InceptionFeatures(args.device, weights_path=args.inception_weights)
    print(f"[model] Inception-v3 on {args.device}")

    store: Dict[str, dict] = {}
    for ci, clip in enumerate(clips):
        pose_path = clips_dir / clip / "poses.npy"
        if not pose_path.is_file():
            continue
        vids = {m: videos_dir / f"{clip}_{m}.mp4" for m in methods}
        if not all(v.is_file() for v in vids.values()):
            print(f"[skip] {clip}: missing video for some method")
            continue
        poses = np.load(pose_path)[: min(video_num_frames(str(v)) for v in vids.values())]
        ts = (
            scene_translation_scale(poses)
            if args.translation_scale_mode == "scene"
            else args.translation_scale
        )
        pairs = subsample_pairs(
            find_revisit_pairs(
                poses, radius=args.radius, min_gap=args.min_gap, translation_scale=ts
            ),
            args.max_pairs_per_clip,
        )
        if not pairs:
            print(f"[{ci+1}/{len(clips)}] {clip}: no revisit pairs")
            continue
        rec = {}
        for m in methods:
            need = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
            fmap = read_video_frames(str(vids[m]), need)
            kept = [(s, t) for s, t, _ in pairs if s in fmap and t in fmap]
            if not kept:
                break
            fa = np.stack([fmap[s] for s, _ in kept], axis=0)
            fb = np.stack([fmap[t] for _, t in kept], axis=0)
            rec[m] = (inception.extract(fa), inception.extract(fb))
        if len(rec) == len(methods):
            store[clip] = rec
            print(f"[{ci+1}/{len(clips)}] {clip}: pairs={len(kept)} features cached")
    return store


def save_cache(path: str, store: Dict[str, dict], methods: List[str]) -> None:
    flat = {}
    for clip, rec in store.items():
        for m, (fa, fb) in rec.items():
            flat[f"{clip}|{m}|first"] = fa
            flat[f"{clip}|{m}|rev"] = fb
    np.savez_compressed(path, **flat)
    print(f"[cache] saved {len(store)} clips x {len(methods)} methods -> {path}")


def load_cache(path: str) -> Dict[str, dict]:
    z = np.load(path)
    store: Dict[str, dict] = {}
    for key in z.files:
        clip, method, which = key.split("|")
        slot = store.setdefault(clip, {}).setdefault(method, {})
        slot[which] = z[key]
    return {
        c: {m: (v["first"], v["rev"]) for m, v in rec.items()} for c, rec in store.items()
    }


def fid_for_clips(store: Dict[str, dict], clips: List[str], method: str) -> float | None:
    fa = np.concatenate([store[c][method][0] for c in clips], axis=0)
    fb = np.concatenate([store[c][method][1] for c in clips], axis=0)
    return fid_from_features(fa, fb)


def _psd_sqrt(mat: "torch.Tensor") -> "torch.Tensor":
    ev, vec = torch.linalg.eigh(mat)
    return (vec * ev.clamp_min(0).sqrt()) @ vec.T


def fid_torch(fa: "torch.Tensor", fb: "torch.Tensor") -> float:
    """FID via symmetric eigendecomposition.

    tr((S1 S2)^1/2) == sum of sqrt(eig(S1^1/2 S2 S1^1/2)), which avoids the
    dense scipy.linalg.sqrtm that dominates bootstrap cost, and runs on GPU.
    """
    mu1, mu2 = fa.mean(0), fb.mean(0)
    s1 = torch.cov(fa.T)
    s2 = torch.cov(fb.T)
    s1h = _psd_sqrt(s1)
    inner = torch.linalg.eigvalsh(s1h @ s2 @ s1h).clamp_min(0)
    diff = mu1 - mu2
    return float(diff @ diff + torch.trace(s1) + torch.trace(s2) - 2 * inner.sqrt().sum())


def to_device_store(
    store: Dict[str, dict], clips: List[str], methods: List[str], device: str
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for c in clips:
        out[c] = {
            m: tuple(
                torch.as_tensor(x, dtype=torch.float64, device=device)
                for x in store[c][m]
            )
            for m in methods
        }
    return out


def fid_gpu_for_clips(dev_store: Dict[str, dict], clips: List[str], method: str) -> float:
    fa = torch.cat([dev_store[c][method][0] for c in clips], dim=0)
    fb = torch.cat([dev_store[c][method][1] for c in clips], dim=0)
    return fid_torch(fa, fb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--methods", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="method:baseline")
    ap.add_argument("--clips", default=None, help="restrict to clip ids in this file")
    ap.add_argument("--radius", type=float, default=0.15)
    ap.add_argument("--min_gap", type=int, default=30)
    ap.add_argument("--max_pairs_per_clip", type=int, default=64)
    ap.add_argument("--translation_scale", type=float, default=1.0)
    ap.add_argument("--translation_scale_mode", choices=["fixed", "scene"], default="fixed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--inception_weights", default=None)
    ap.add_argument("--cache", default=None, help="npz path to read/write features")
    ap.add_argument("--verify", action="store_true",
                    help="cross-check the eigh-based FID against scipy sqrtm")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.cache and os.path.isfile(args.cache):
        store = load_cache(args.cache)
        print(f"[cache] loaded {len(store)} clips from {args.cache}")
    else:
        clips_dir = Path(args.clips_dir)
        all_clips = sorted(p.name for p in clips_dir.iterdir() if p.is_dir())
        store = extract_features(args, all_clips, methods)
        if args.cache:
            save_cache(args.cache, store, methods)

    clips = sorted(store)
    if args.clips:
        want = {
            os.path.splitext(os.path.basename(l.strip()))[0]
            for l in open(args.clips)
            if l.strip() and not l.startswith("#")
        }
        clips = [c for c in clips if c in want]
    print(f"[fid] clips={len(clips)} boot={args.n_boot}")

    dev_store = to_device_store(store, clips, methods, args.device)
    full = {m: fid_gpu_for_clips(dev_store, clips, m) for m in methods}
    for m, v in full.items():
        line = f"  FID[{m}] = {v:.3f}"
        if args.verify:
            line += f"   (scipy sqrtm: {fid_for_clips(store, clips, m):.3f})"
        print(line, flush=True)

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(clips), size=(args.n_boot, len(clips)))
    boot = {m: np.empty(args.n_boot) for m in methods}
    for i in range(args.n_boot):
        sample = [clips[j] for j in draws[i]]
        for m in methods:
            boot[m][i] = fid_gpu_for_clips(dev_store, sample, m)
        if (i + 1) % 100 == 0:
            print(f"  bootstrap {i+1}/{args.n_boot}", flush=True)

    print()
    print("| comparison | FID(method) | FID(baseline) | ΔFID | 95% CI | P(Δ<0) | verdict |")
    print("|---|---:|---:|---:|:--:|---:|:--|")
    for pair in args.pairs:
        method, baseline = pair.split(":", 1)
        d = boot[method] - boot[baseline]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_better = float((d < 0).mean())
        verdict = (
            "significant win" if hi < 0
            else "significant loss" if lo > 0
            else "no difference"
        )
        print(
            f"| {method} vs {baseline} | {full[method]:.2f} | {full[baseline]:.2f} | "
            f"{d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p_better:.3f} | {verdict} |"
        )


if __name__ == "__main__":
    main()
