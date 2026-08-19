#!/usr/bin/env python3
"""WorldKV-style world-memory evaluation on RealCam-Vid rollouts.

Protocol (Yi et al., WorldKV, arXiv:2605.22718):
  - Locate revisit frames via GT camera poses (loop closure / same viewpoint).
  - Compare each revisit frame to its first-visit frame at that viewpoint with
    PSNR / SSIM / LPIPS.
  - FID between the set of revisit frames and the set of first-visit frames.
  - Efficiency: throughput (FPS), attention context, peak GPU memory.

Pose pairing is method-independent (same pairs for all methods on a clip).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pose geometry
# ---------------------------------------------------------------------------

def _rot_geodesic(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Ra.T @ Rb
    tr = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(tr))


def se3_distance_c2w(
    pa: np.ndarray,
    pb: np.ndarray,
    translation_scale: float = 1.0,
    w_trans: float = 1.0,
    w_rot: float = 1.0,
) -> float:
    """SE(3) distance on 4x4 c2w matrices (matches wan.modules.chunk_selector)."""
    d_trans = float(np.linalg.norm(pa[:3, 3] - pb[:3, 3])) / max(translation_scale, 1e-6)
    d_rot = _rot_geodesic(pa[:3, :3], pb[:3, :3]) / math.pi
    return w_trans * d_trans + w_rot * d_rot


def scene_translation_scale(poses: np.ndarray) -> float:
    """Optional scene-normalized scale (bbox diagonal). Default pairing uses 1.0."""
    t = poses[:, :3, 3]
    diag = float(np.linalg.norm(t.max(0) - t.min(0)))
    return max(diag, 1e-3)


def find_revisit_pairs(
    poses: np.ndarray,
    radius: float = 0.15,
    min_gap: int = 30,
    translation_scale: Optional[float] = None,
) -> List[Tuple[int, int, float]]:
    """Return (first_visit_idx, revisit_idx, se3_dist) for each revisiting frame.

    For every frame t, search s in [0, t - min_gap] for the closest pose within
    ``radius``. Ties prefer the earliest visit (true first-visit).
    """
    T = len(poses)
    if T <= min_gap + 1:
        return []
    # Default scale=1.0 (metric-ish c2w units). Pass a float or use scene bbox via caller.
    ts = float(translation_scale) if translation_scale is not None else 1.0
    pairs: List[Tuple[int, int, float]] = []
    for t in range(min_gap, T):
        best_s = -1
        best_d = float("inf")
        for s in range(0, t - min_gap + 1):
            d = se3_distance_c2w(poses[s], poses[t], translation_scale=ts)
            if d <= radius and (d < best_d - 1e-12 or (abs(d - best_d) <= 1e-12 and s < best_s)):
                best_d = d
                best_s = s
        if best_s >= 0:
            pairs.append((best_s, t, float(best_d)))
    return pairs


def subsample_pairs(
    pairs: Sequence[Tuple[int, int, float]],
    max_pairs: int,
) -> List[Tuple[int, int, float]]:
    if max_pairs <= 0 or len(pairs) <= max_pairs:
        return list(pairs)
    idx = np.linspace(0, len(pairs) - 1, num=max_pairs, dtype=np.int64)
    return [pairs[int(i)] for i in idx]


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def read_video_frames(path: str, indices: Sequence[int]) -> Dict[int, np.ndarray]:
    """Return {frame_idx: uint8 RGB HxWx3} for the given indices."""
    want = sorted(set(int(i) for i in indices if int(i) >= 0))
    if not want:
        return {}
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        n = len(vr)
        idxs = [i for i in want if i < n]
        if not idxs:
            return {}
        batch = vr.get_batch(idxs).asnumpy()
        return {i: batch[k] for k, i in enumerate(idxs)}
    except Exception:
        import torchvision.io as tio
        v, _, _ = tio.read_video(path, pts_unit="sec")  # T,H,W,C uint8
        n = int(v.shape[0])
        idxs = [i for i in want if i < n]
        if not idxs:
            return {}
        arr = v[idxs].numpy()
        return {i: arr[k] for k, i in enumerate(idxs)}


def video_num_frames(path: str) -> int:
    try:
        import decord
        return len(decord.VideoReader(path, ctx=decord.cpu(0)))
    except Exception:
        import torchvision.io as tio
        v, _, _ = tio.read_video(path, pts_unit="sec")
        return int(v.shape[0])


# ---------------------------------------------------------------------------
# Image metrics
# ---------------------------------------------------------------------------

def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def ssim_uint8(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    # channel_axis for RGB; win_size auto-capped by image size
    h = min(a.shape[0], a.shape[1])
    win = 7 if h >= 7 else (h if h % 2 == 1 else max(h - 1, 3))
    return float(structural_similarity(a, b, channel_axis=2, data_range=255, win_size=win))


class LPIPSMeter:
    def __init__(self, device: str, net: str = "alex"):
        import lpips
        self.device = device
        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, a: np.ndarray, b: np.ndarray, batch_size: int = 16) -> List[float]:
        """a,b: [N,H,W,3] uint8 -> list of LPIPS scores."""
        if len(a) == 0:
            return []
        ta = torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255.0
        tb = torch.from_numpy(b).permute(0, 3, 1, 2).float() / 255.0
        ta = ta * 2 - 1
        tb = tb * 2 - 1
        if ta.shape[-2:] != tb.shape[-2:]:
            tb = F.interpolate(tb, size=ta.shape[-2:], mode="bilinear", align_corners=False)
        vals: List[float] = []
        for i in range(0, ta.shape[0], batch_size):
            aa = ta[i:i + batch_size].to(self.device)
            bb = tb[i:i + batch_size].to(self.device)
            vals.extend(self.model(aa, bb).view(-1).cpu().tolist())
        return [float(v) for v in vals]


def _load_inception_v3(device: str, weights_path: Optional[str] = None):
    """Load Inception-v3; prefer local weights to avoid download failures."""
    from torchvision.models import inception_v3, Inception_V3_Weights
    candidates = []
    if weights_path:
        candidates.append(Path(weights_path))
    env = os.environ.get("INCEPTION_WEIGHTS", "")
    if env:
        candidates.append(Path(env))
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    candidates.append(torch_home / "hub" / "checkpoints" / "inception_v3_google-0cc3c7bd.pth")
    # Common project-local cache used on this machine
    candidates.append(Path("/DATA/YuanZhen/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth"))

    for p in candidates:
        if p.is_file() and p.stat().st_size > 1_000_000:
            net = inception_v3(weights=None, transform_input=False, init_weights=False)
            state = torch.load(str(p), map_location="cpu")
            net.load_state_dict(state)
            net.fc = torch.nn.Identity()
            return net.eval().to(device)

    # Last resort: torchvision download (may fail offline)
    weights = Inception_V3_Weights.IMAGENET1K_V1
    net = inception_v3(weights=weights, transform_input=False)
    net.fc = torch.nn.Identity()
    return net.eval().to(device)


class InceptionFeatures:
    """Inception-v3 pool features for FID (299x299, ImageNet norm)."""

    def __init__(self, device: str, weights_path: Optional[str] = None):
        self.net = _load_inception_v3(device, weights_path=weights_path)
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract(self, frames: np.ndarray, batch_size: int = 32) -> np.ndarray:
        if len(frames) == 0:
            return np.zeros((0, 2048), dtype=np.float64)
        feats = []
        for i in range(0, len(frames), batch_size):
            x = torch.from_numpy(frames[i:i + batch_size]).permute(0, 3, 1, 2).float() / 255.0
            x = x.to(self.device)
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            f = self.net(x)
            if isinstance(f, tuple):
                f = f[0]
            feats.append(f.detach().float().cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float64)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    from scipy import linalg
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def fid_from_features(fa: np.ndarray, fb: np.ndarray) -> Optional[float]:
    if fa.shape[0] < 2 or fb.shape[0] < 2:
        return None
    mu1, mu2 = fa.mean(0), fb.mean(0)
    sigma1 = np.cov(fa, rowvar=False)
    sigma2 = np.cov(fb, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


# ---------------------------------------------------------------------------
# Stats / efficiency
# ---------------------------------------------------------------------------

def load_runtime_stats(stats_path: Path) -> Dict:
    d = json.loads(stats_path.read_text())
    s = d.get("stats", d)
    cfg = d.get("config", {})
    t = s.get("total_generation_time_s")
    frames = cfg.get("frame_num")
    tail = s.get("tail_chunk_time_s")
    ctx = s.get("avg_attention_context_tokens")
    # Official WorldKV logs active latent frames; map to token-scale for the table.
    if ctx is None and s.get("worldkv_active_latent_frames") is not None:
        tpf = s.get("tokens_per_latent_frame") or cfg.get("tokens_per_latent_frame") or 1560.0
        ctx = float(s["worldkv_active_latent_frames"]) * float(tpf)
    out = {
        "total_generation_time_s": t,
        "avg_attention_context_tokens": ctx,
        "max_attention_context_tokens": s.get("max_attention_context_tokens"),
        "peak_memory_allocated_gb": s.get("peak_memory_allocated_gb"),
        "retained_chunks": s.get("retained_chunks"),
        "total_chunks": s.get("total_chunks"),
        "tail_chunk_time_s": tail,
        "avg_chunk_time_s": s.get("avg_chunk_time_s"),
        "config_frame_num": frames,
        "feature_schema": s.get("feature_schema"),
        "translation_scale": s.get("translation_scale"),
        "worldkv_active_latent_frames": s.get("worldkv_active_latent_frames"),
    }
    return out


def discover_clips(videos_dir: Path, methods: Sequence[str]) -> List[str]:
    """Clips that have videos for every requested method (methods[0] as anchor)."""
    if not methods:
        return []
    suffix = f"_{methods[0]}.mp4"
    clips = []
    for p in sorted(videos_dir.glob(f"*{suffix}")):
        clip = p.name[: -len(suffix)]
        if all((videos_dir / f"{clip}_{m}.mp4").is_file() for m in methods):
            clips.append(clip)
    return clips


def filter_available_methods(videos_dir: Path, methods: Sequence[str]) -> List[str]:
    """Drop methods with zero videos so partial runs still evaluate."""
    keep = []
    for m in methods:
        if any(videos_dir.glob(f"*_{m}.mp4")):
            keep.append(m)
        else:
            print(f"[warn] no videos for method={m}; skipping")
    return keep


def filter_subset(clips: Sequence[str], clips_dir: Optional[Path]) -> List[str]:
    if clips_dir is None:
        return list(clips)
    keep = {p.name for p in clips_dir.iterdir() if p.is_dir()}
    return [c for c in clips if c in keep]


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

def eval_clip_method(
    video_path: Path,
    poses: np.ndarray,
    pairs: Sequence[Tuple[int, int, float]],
    lpips_meter: Optional[LPIPSMeter],
    inception: Optional[InceptionFeatures],
    device: str,
) -> Dict:
    if not pairs:
        return {
            "n_pairs": 0,
            "psnr": None, "ssim": None, "lpips": None,
            "first_feats": None, "revisit_feats": None,
            "video_frames": video_num_frames(str(video_path)),
        }
    first_idx = [p[0] for p in pairs]
    rev_idx = [p[1] for p in pairs]
    need = sorted(set(first_idx + rev_idx))
    fmap = read_video_frames(str(video_path), need)
    pairs = [(s, t, d) for s, t, d in pairs if s in fmap and t in fmap]
    first_idx = [p[0] for p in pairs]
    rev_idx = [p[1] for p in pairs]
    if not pairs:
        return {
            "n_pairs": 0,
            "psnr": None, "ssim": None, "lpips": None,
            "first_feats": None, "revisit_feats": None,
            "video_frames": video_num_frames(str(video_path)),
        }

    fa = np.stack([fmap[i] for i in first_idx], axis=0)
    fb = np.stack([fmap[i] for i in rev_idx], axis=0)

    psnrs = [psnr_uint8(fa[i], fb[i]) for i in range(len(pairs))]
    ssims = [ssim_uint8(fa[i], fb[i]) for i in range(len(pairs))]
    lpips_vals = lpips_meter(fa, fb) if lpips_meter is not None else []

    first_feats = inception.extract(fa) if inception is not None else None
    revisit_feats = inception.extract(fb) if inception is not None else None

    out = {
        "n_pairs": len(pairs),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpips_vals)) if lpips_vals else None,
        "psnr_std": float(np.std(psnrs)),
        "ssim_std": float(np.std(ssims)),
        "lpips_std": float(np.std(lpips_vals)) if lpips_vals else None,
        "mean_pose_dist": float(np.mean([p[2] for p in pairs])),
        "first_feats": first_feats,
        "revisit_feats": revisit_feats,
        "video_frames": max(need) + 1 if need else 0,
    }
    return out


def mean_or_none(xs: List[Optional[float]]) -> Optional[float]:
    vs = [x for x in xs if x is not None]
    return float(np.mean(vs)) if vs else None


def main():
    ap = argparse.ArgumentParser(description="WorldKV-style revisit memory evaluation")
    ap.add_argument("--out_dir", required=True, help="Generation out dir (videos/ + stats/)")
    ap.add_argument(
        "--clips_dir",
        default=None,
        help="Clip metadata dir with poses.npy (default: bench/realcamvid/clips_default_loop)",
    )
    ap.add_argument(
        "--methods",
        default="window,world_state_cr",
        help="Comma-separated method suffixes matching video filenames",
    )
    ap.add_argument("--radius", type=float, default=0.15,
                    help="SE(3) revisit radius (WorldKV-style / CR default)")
    ap.add_argument("--min_gap", type=int, default=30,
                    help="Min frame gap between first-visit and revisit")
    ap.add_argument("--max_pairs_per_clip", type=int, default=64,
                    help="Cap pairs per clip (even subsample); 0 = all")
    ap.add_argument("--translation_scale", type=float, default=1.0,
                    help="SE(3) translation scale (default 1.0; use scene bbox with --translation_scale_mode scene)")
    ap.add_argument("--translation_scale_mode", choices=["fixed", "scene"], default="fixed",
                    help="fixed: use --translation_scale; scene: per-clip bbox diagonal")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--no_fid", action="store_true")
    ap.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--inception_weights", default=None,
                    help="Path to inception_v3_google-*.pth (or set INCEPTION_WEIGHTS)")
    ap.add_argument("--save_name", default="worldkv_memory_eval.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    videos_dir = out_dir / "videos"
    stats_dir = out_dir / "stats"
    if not videos_dir.is_dir():
        print(f"[error] missing videos dir: {videos_dir}", file=sys.stderr)
        sys.exit(1)

    root = Path(__file__).resolve().parents[1]  # lingbot-world
    clips_dir = Path(args.clips_dir) if args.clips_dir else (root / "bench/realcamvid/clips_default_loop")
    clips_dir = clips_dir.resolve()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods = filter_available_methods(videos_dir, methods)
    if not methods:
        print("[error] no methods with videos found", file=sys.stderr)
        sys.exit(1)

    clips = discover_clips(videos_dir, methods)
    clips = filter_subset(clips, clips_dir if clips_dir.is_dir() else None)
    if not clips:
        print(f"[error] no overlapping clips in {videos_dir} ∩ {clips_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[worldkv-eval] clips={len(clips)} methods={methods} radius={args.radius} "
          f"min_gap={args.min_gap} max_pairs={args.max_pairs_per_clip}")

    lpips_meter = None
    if not args.no_lpips:
        try:
            lpips_meter = LPIPSMeter(args.device, net=args.lpips_net)
            print(f"[model] LPIPS({args.lpips_net}) on {args.device}")
        except Exception as e:
            print(f"[warn] LPIPS unavailable: {e}")

    inception = None
    if not args.no_fid:
        try:
            inception = InceptionFeatures(args.device, weights_path=args.inception_weights)
            print(f"[model] Inception-v3 features on {args.device}")
        except Exception as e:
            print(f"[warn] Inception/FID unavailable: {e}")

    per_clip: Dict[str, Dict] = {}
    feats_first: Dict[str, List[np.ndarray]] = defaultdict(list)
    feats_rev: Dict[str, List[np.ndarray]] = defaultdict(list)
    runtime_rows: Dict[str, List[Dict]] = defaultdict(list)

    for ci, clip in enumerate(clips):
        pose_path = clips_dir / clip / "poses.npy"
        if not pose_path.is_file():
            print(f"[skip] {clip}: no poses.npy")
            continue
        poses_full = np.load(pose_path)
        # Align pose length to shortest available video among methods
        n_vid = min(video_num_frames(str(videos_dir / f"{clip}_{m}.mp4")) for m in methods)
        poses = poses_full[:n_vid]
        if args.translation_scale_mode == "scene":
            ts = scene_translation_scale(poses)
        else:
            ts = args.translation_scale
        pairs_all = find_revisit_pairs(
            poses, radius=args.radius, min_gap=args.min_gap, translation_scale=ts)
        pairs = subsample_pairs(pairs_all, args.max_pairs_per_clip)
        print(f"[{ci+1}/{len(clips)}] {clip}: frames={n_vid} revisits={len(pairs_all)} "
              f"eval_pairs={len(pairs)} scale={scene_translation_scale(poses):.4f}")

        clip_row = {
            "frames": n_vid,
            "n_revisit_all": len(pairs_all),
            "n_revisit_eval": len(pairs),
            "methods": {},
        }
        for m in methods:
            vpath = videos_dir / f"{clip}_{m}.mp4"
            r = eval_clip_method(vpath, poses, pairs, lpips_meter, inception, args.device)
            # strip large arrays from per-clip dump
            compact = {k: v for k, v in r.items() if k not in ("first_feats", "revisit_feats")}
            clip_row["methods"][m] = compact
            if r.get("first_feats") is not None and len(r["first_feats"]):
                feats_first[m].append(r["first_feats"])
                feats_rev[m].append(r["revisit_feats"])
            sp = stats_dir / f"{clip}_{m}.json"
            if sp.is_file():
                st = load_runtime_stats(sp)
                # WorldKV throughput: prefer wall FPS from decoded frames
                t = st.get("total_generation_time_s")
                vf = r.get("video_frames") or n_vid
                st["throughput_fps"] = (float(vf) / float(t)) if t and t > 0 else None
                # last-chunk proxy (3 latents ≈ chunk; pixel frames ≈ n_vid / total_chunks)
                tc = st.get("total_chunks") or 0
                tail = st.get("tail_chunk_time_s")
                if tc and tail and tail > 0 and vf:
                    st["last_chunk_fps"] = (float(vf) / float(tc)) / float(tail)
                runtime_rows[m].append(st)
            print(
                f"    {m}: pairs={r['n_pairs']} "
                f"PSNR={r['psnr']:.3f} SSIM={r['ssim']:.4f} "
                f"LPIPS={r['lpips'] if r['lpips'] is not None else float('nan'):.4f}"
                if r["n_pairs"] else f"    {m}: no pairs"
            )
        per_clip[clip] = clip_row

    # Aggregate
    summary = {
        "protocol": {
            "name": "worldkv_revisit_memory",
            "reference": "WorldKV (arXiv:2605.22718) — PSNR/SSIM/LPIPS/FID on revisit vs first-visit",
            "radius": args.radius,
            "min_gap": args.min_gap,
            "max_pairs_per_clip": args.max_pairs_per_clip,
            "translation_scale": args.translation_scale,
            "clips_dir": str(clips_dir),
            "out_dir": str(out_dir),
            "n_clips": len(per_clip),
            "methods": methods,
        },
        "by_method": {},
        "per_clip": per_clip,
    }

    for m in methods:
        psnrs, ssims, lpips_list = [], [], []
        n_pairs_total = 0
        for clip, row in per_clip.items():
            mr = row["methods"].get(m, {})
            if mr.get("n_pairs", 0) > 0:
                psnrs.append(mr["psnr"])
                ssims.append(mr["ssim"])
                if mr.get("lpips") is not None:
                    lpips_list.append(mr["lpips"])
                n_pairs_total += mr["n_pairs"]
        fid = None
        if feats_first[m] and feats_rev[m]:
            fa = np.concatenate(feats_first[m], axis=0)
            fb = np.concatenate(feats_rev[m], axis=0)
            fid = fid_from_features(fa, fb)

        rt = runtime_rows[m]
        def _rmean(key):
            vs = [r[key] for r in rt if r.get(key) is not None]
            return float(np.mean(vs)) if vs else None

        summary["by_method"][m] = {
            "n_clips": sum(1 for row in per_clip.values() if row["methods"].get(m, {}).get("n_pairs", 0) > 0),
            "n_pairs_total": n_pairs_total,
            "psnr": float(np.mean(psnrs)) if psnrs else None,
            "ssim": float(np.mean(ssims)) if ssims else None,
            "lpips": float(np.mean(lpips_list)) if lpips_list else None,
            "fid": fid,
            "throughput_fps": _rmean("throughput_fps"),
            "last_chunk_fps": _rmean("last_chunk_fps"),
            "mean_time_s": _rmean("total_generation_time_s"),
            "mean_ctx": _rmean("avg_attention_context_tokens"),
            "mean_peak_gb": _rmean("peak_memory_allocated_gb"),
            "mean_retained": _rmean("retained_chunks"),
        }

    save_path = out_dir / args.save_name
    # JSON-safe: drop any leftover ndarrays
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    save_path.write_text(json.dumps(_sanitize(summary), indent=2))
    print("\n======== WorldKV Memory Eval ========")
    print(f"{'method':20s} {'PSNR↑':>8s} {'SSIM↑':>8s} {'LPIPS↓':>8s} {'FID↓':>8s} "
          f"{'FPS↑':>8s} {'ctx':>10s} {'peakGB':>8s}")
    for m in methods:
        r = summary["by_method"][m]
        print(
            f"{m:20s} "
            f"{(r['psnr'] if r['psnr'] is not None else float('nan')):8.3f} "
            f"{(r['ssim'] if r['ssim'] is not None else float('nan')):8.4f} "
            f"{(r['lpips'] if r['lpips'] is not None else float('nan')):8.4f} "
            f"{(r['fid'] if r['fid'] is not None else float('nan')):8.2f} "
            f"{(r['throughput_fps'] if r['throughput_fps'] is not None else float('nan')):8.3f} "
            f"{(r['mean_ctx'] if r['mean_ctx'] is not None else float('nan')):10.1f} "
            f"{(r['mean_peak_gb'] if r['mean_peak_gb'] is not None else float('nan')):8.2f}"
        )
    print(f"[saved] {save_path}")


if __name__ == "__main__":
    main()
