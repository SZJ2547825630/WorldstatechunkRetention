#!/usr/bin/env python3
"""Pairwise video quality: PSNR / SSIM / LPIPS between method videos."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _read_mp4(path: str, max_frames: int = 0) -> torch.Tensor:
    """Return float tensor [T,3,H,W] in [0,1]."""
    import torchvision.io as tio
    v, _, _ = tio.read_video(path, pts_unit="sec")
    if max_frames > 0:
        v = v[:max_frames]
    x = v.permute(0, 3, 1, 2).float() / 255.0
    return x


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _ssim_simple(a: torch.Tensor, b: torch.Tensor) -> float:
    # Per-frame mean SSIM via torchvision if available; else correlation proxy.
    try:
        from torchvision.transforms.functional import rgb_to_grayscale
        from torchmetrics.functional.image import structural_similarity_index_measure as ssim
        return float(ssim(a, b, data_range=1.0).item())
    except Exception:
        a = a.flatten(1)
        b = b.flatten(1)
        a = a - a.mean(1, keepdim=True)
        b = b - b.mean(1, keepdim=True)
        num = (a * b).sum(1)
        den = a.norm(dim=1) * b.norm(dim=1) + 1e-8
        return float((num / den).mean().item())


@torch.no_grad()
def compare_pair(path_a: str, path_b: str, lpips_fn=None, max_frames: int = 0, device="cuda"):
    a = _read_mp4(path_a, max_frames)
    b = _read_mp4(path_b, max_frames)
    t = min(a.shape[0], b.shape[0])
    a, b = a[:t], b[:t]
    if a.shape[-2:] != b.shape[-2:]:
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
    out = {
        "frames": int(t),
        "psnr": _psnr(a, b),
        "ssim": _ssim_simple(a, b),
        "mae": float((a - b).abs().mean().item()),
    }
    if lpips_fn is not None:
        # LPIPS expects [-1,1]
        aa = (a * 2 - 1).to(device)
        bb = (b * 2 - 1).to(device)
        vals = []
        bs = 8
        for i in range(0, t, bs):
            vals.append(lpips_fn(aa[i:i+bs], bb[i:i+bs]).mean().item())
        out["lpips"] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument(
        "--methods", nargs="+",
        default=["learned_cr", "world_state_cr", "heuristic_cr"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(args.device).eval()
    except Exception as e:
        print(f"[warn] LPIPS unavailable: {e}")

    rows = []
    for clip in args.clips:
        paths = {}
        for m in args.methods:
            p = os.path.join(args.videos_dir, f"{clip}_{m}.mp4")
            if os.path.isfile(p):
                paths[m] = p
        if len(paths) < 2:
            print(f"[skip] {clip}: need >=2 methods, found {list(paths)}")
            continue
        methods = list(paths)
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                a, b = methods[i], methods[j]
                r = compare_pair(paths[a], paths[b], lpips_fn, args.max_frames, args.device)
                r.update({"clip": clip, "a": a, "b": b})
                rows.append(r)
                print(f"{clip} {a} vs {b}: mae={r['mae']:.4f} psnr={r['psnr']:.2f} "
                      f"ssim={r['ssim']:.4f} lpips={r.get('lpips', float('nan')):.4f}")

    # Aggregate pairwise vs reference method if present
    summary = {"pairs": rows}
    for ref in args.methods:
        vals = [r for r in rows if r["a"] == ref or r["b"] == ref]
        if not vals:
            continue
        # Prefer pairs against the first method listed after ref.
        summary[f"vs_{ref}"] = {
            "n": len(vals),
            "mae_mean": float(np.mean([r["mae"] for r in vals])),
            "psnr_mean": float(np.mean([r["psnr"] for r in vals])),
            "ssim_mean": float(np.mean([r["ssim"] for r in vals])),
            "lpips_mean": float(np.mean([r["lpips"] for r in vals if "lpips" in r])) if any("lpips" in r for r in vals) else None,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
