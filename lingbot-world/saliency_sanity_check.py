"""
MoSaiC Saliency Sanity Check
─────────────────────────────────────────────────────────────────────────

The MoSaiC design hypothesis:
  - In a generated long video, the per-token saliency (defined as the magnitude
    of temporal change at each spatial patch) is highly NON-UNIFORM.
  - High-saliency tokens (moving subjects, dynamic regions) carry most of the
    information needed for long-range coherence.
  - Low-saliency tokens (static background) can be aggressively merged/pruned
    with minimal quality loss.

This script validates the hypothesis quickly (~10s) by:
  1. Sampling consecutive chunk boundaries from a generated video.
  2. Computing patch-level temporal-difference saliency maps.
  3. Saving heatmap visualizations for visual inspection.
  4. Printing concentration statistics with a clear GO/NO-GO verdict.

If concentration is high (>= 0.65), SWTP can drop the low half of tokens
while retaining >= 90% of the saliency signal → MoSaiC is viable.

Usage:
    python saliency_sanity_check.py --video output/example04_MoCE.mp4

    # Custom patch size (DiT vae+patch downsample → effective 16 px/patch):
    python saliency_sanity_check.py --video <path> --patch_size 16

    # Inspect specific timestamps:
    python saliency_sanity_check.py --video <path> --probe_times 5 15 22 28

Output:
    output/<video_stem>_saliency/
      ├── saliency_t<TIME>s.jpg         (visualization at each probe time)
      ├── saliency_t<TIME>s_overlay.jpg (heatmap overlaid on frame)
      └── saliency_stats.json           (numeric stats)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# Plücker saliency support (world-model-aware)
# ════════════════════════════════════════════════════════════════════════════

_PLUCKER_DEPS_LOADED = False

def _lazy_load_plucker_deps():
    global _PLUCKER_DEPS_LOADED, interpolate_camera_poses, compute_relative_poses, get_plucker_embeddings
    if _PLUCKER_DEPS_LOADED:
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch  # noqa: F401  (needed by cam_utils)
    from wan.utils.cam_utils import (
        interpolate_camera_poses as _interpolate,
        compute_relative_poses as _relative,
        get_plucker_embeddings as _plucker,
    )
    globals()['interpolate_camera_poses'] = _interpolate
    globals()['compute_relative_poses'] = _relative
    globals()['get_plucker_embeddings'] = _plucker
    globals()['torch'] = torch
    _PLUCKER_DEPS_LOADED = True


def load_plucker_full_video(action_path, frame_h, frame_w, total_video_frames, chunk_size=3):
    """
    Mimic the pipeline's Plücker preprocessing for the full video.

    Replicates the logic in `image2video_fast.py:generate()` to produce a
    Plücker tensor at latent-frame resolution. The latent-frame count
    matches what the pipeline actually consumes.

    Returns:
        plucker:  np.ndarray [lat_f, frame_h, frame_w, 6]
        lat_f:    int (number of latent frames)
        wasd:     np.ndarray [F, 3] or None (act mode WASD signal,
                                              aligned to latent frames)
    """
    _lazy_load_plucker_deps()

    poses_path = os.path.join(action_path, 'poses.npy')
    intrinsics_path = os.path.join(action_path, 'intrinsics.npy')
    if not (os.path.isfile(poses_path) and os.path.isfile(intrinsics_path)):
        raise RuntimeError(
            f"Plücker mode requires poses.npy and intrinsics.npy in {action_path}")

    poses_np = np.load(poses_path)
    intrinsics_np = np.load(intrinsics_path)

    # Pipeline rounds frame_num to 4n+1 and caps at len(poses)
    # video frames = (lat_f - 1) * 4 + 1 corresponds to total_video_frames
    # → recover lat_f from total_video_frames
    # Pipeline: lat_f = (frame_num - 1) // 4 + 1, then rounded down to multiple of chunk_size
    inferred_lat_f = (total_video_frames - 1) // 4 + 1
    inferred_lat_f = inferred_lat_f - (inferred_lat_f % chunk_size)
    num_video_frames = (inferred_lat_f - 1) * 4 + 1
    num_video_frames = min(num_video_frames, len(poses_np))

    poses_np = poses_np[:num_video_frames]

    # Interpolate to latent frame count (pipeline does this)
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, len(poses_np) - 1, len(poses_np)),
        src_rot_mat=poses_np[:, :3, :3],
        src_trans_vec=poses_np[:, :3, 3],
        tgt_indices=np.linspace(0, len(poses_np) - 1, inferred_lat_f),
    )
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)

    # Build Ks for each latent frame (use first frame's intrinsics; pipeline
    # transforms these but we keep raw here for simplicity)
    Ks = torch.from_numpy(intrinsics_np[0]).float()
    Ks = Ks.unsqueeze(0).expand(inferred_lat_f, 4).contiguous()

    # Compute Plücker at full pixel resolution → [lat_f, H, W, 6]
    plucker = get_plucker_embeddings(
        c2ws_infer.float(), Ks, frame_h, frame_w, only_rays_d=False
    )

    # Optional WASD (act mode)
    wasd = None
    wasd_path = os.path.join(action_path, 'wasd_action.npy')
    if os.path.isfile(wasd_path):
        wasd_full = np.load(wasd_path)
        wasd_full = wasd_full[:num_video_frames]
        # Downsample to latent frame rate (every 4th frame)
        wasd = wasd_full[::4][:inferred_lat_f]

    return plucker.cpu().numpy(), inferred_lat_f, wasd


def compute_plucker_saliency_at_time(plucker, lat_f, target_time_s,
                                      video_fps, video_lat_fps,
                                      window_lat_frames=3,
                                      patch_size=16,
                                      mode='spatial+temporal'):
    """
    Aggregate Plücker-based saliency at a target time, pool to patch grid.

    mode options:
        'spatial':           |∇_xy plucker|, time-averaged across window
        'temporal':          |∇_t plucker|, spatially as-is
        'spatial+temporal':  sum of both (default, most robust)

    Returns (saliency_patch_grid, central_lat_frame_idx) or (None, None) if
    out of range.
    """
    F, H, W, C = plucker.shape
    # Convert target time → latent frame index
    target_lat = int(round(target_time_s * video_lat_fps))
    half = max(1, window_lat_frames // 2)
    start = max(0, target_lat - half)
    end = min(F, target_lat + half + 1)
    if end - start < 2:
        return None, None

    chunk = plucker[start:end]  # [w, H, W, C]
    sal = np.zeros((H, W), dtype=np.float32)

    if 'spatial' in mode:
        # Per-frame spatial gradient magnitude, then time-mean
        dx = np.abs(chunk[:, :, 1:] - chunk[:, :, :-1]).mean(axis=-1)  # [w, H, W-1]
        dy = np.abs(chunk[:, 1:, :] - chunk[:, :-1, :]).mean(axis=-1)  # [w, H-1, W]
        sx = np.zeros((H, W), dtype=np.float32)
        sy = np.zeros((H, W), dtype=np.float32)
        sx[:, :-1] = dx.mean(axis=0)
        sy[:-1, :] = dy.mean(axis=0)
        sal += sx + sy

    if 'temporal' in mode and chunk.shape[0] > 1:
        dt = np.abs(chunk[1:] - chunk[:-1]).mean(axis=-1)  # [w-1, H, W]
        sal += dt.mean(axis=0)

    # Pool to patch grid
    H_p, W_p = H // patch_size, W // patch_size
    cropped = sal[:H_p * patch_size, :W_p * patch_size]
    return cropped.reshape(H_p, patch_size, W_p, patch_size).mean(axis=(1, 3)), target_lat


def load_video_frames(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps, total


def patch_pool(diff_hw, patch_size):
    """Mean-pool an [H, W] image to [H/patch, W/patch]."""
    H, W = diff_hw.shape
    H_p, W_p = H // patch_size, W // patch_size
    cropped = diff_hw[:H_p * patch_size, :W_p * patch_size]
    return cropped.reshape(H_p, patch_size, W_p, patch_size).mean(axis=(1, 3))


def compute_saliency_at_time(frames, fps, target_time_s, window_frames=12, patch_size=16):
    """
    Aggregate temporal-difference saliency over a chunk-sized window around
    target_time_s. Returns (patch-level saliency map, reference frame).

    The window size 12 corresponds to one "chunk" in MoCE
    (chunk_size=3 latent frames × vae_stride=4 video frames).
    """
    center_idx = int(round(target_time_s * fps))
    start = max(0, center_idx - window_frames // 2)
    end = min(len(frames), start + window_frames + 1)
    if end - start < 2:
        return None, None

    diffs = []
    for i in range(start + 1, end):
        d = np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)).mean(axis=-1)
        diffs.append(d)
    if not diffs:
        return None, None
    # Aggregate over window
    diff_map = np.stack(diffs, axis=0).mean(axis=0)
    saliency = patch_pool(diff_map, patch_size)
    ref_frame = frames[center_idx if center_idx < len(frames) else len(frames) - 1]
    return saliency, ref_frame


def saliency_stats(saliency_map):
    """
    Compute concentration statistics.

    Returns:
        dict with keys:
            top_K_concentration_50pct  - fraction of total saliency in top 50% of tokens
            top_K_concentration_25pct  - fraction in top 25%
            top_K_concentration_10pct  - fraction in top 10%
            gini                       - Gini coefficient (0 uniform, 1 concentrated)
            cv                         - coefficient of variation (std/mean)
    """
    flat = saliency_map.flatten().astype(np.float64)
    n = len(flat)
    total = flat.sum() + 1e-12
    sorted_desc = np.sort(flat)[::-1]

    def top_frac(k_frac):
        k = max(1, int(round(n * k_frac)))
        return float(sorted_desc[:k].sum() / total)

    sorted_asc = np.sort(flat)
    cum = np.cumsum(sorted_asc) / total
    # Gini coefficient
    gini = float(1.0 - 2.0 * np.sum(cum) / n + 1.0 / n)

    cv = float(flat.std() / (flat.mean() + 1e-12))

    return {
        "top_50pct": top_frac(0.5),
        "top_25pct": top_frac(0.25),
        "top_10pct": top_frac(0.1),
        "gini": gini,
        "cv": cv,
    }


def visualize_saliency(saliency_map, ref_frame, out_path, overlay_path):
    """Save (1) raw heatmap and (2) heatmap overlaid on reference frame."""
    sal = saliency_map.copy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-12)

    # Upsample saliency to frame size
    H_f, W_f = ref_frame.shape[:2]
    sal_up = cv2.resize(sal, (W_f, H_f), interpolation=cv2.INTER_NEAREST)
    heat_u8 = (sal_up * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_path), heatmap_bgr)

    # Overlay on frame
    ref_bgr = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(ref_bgr, 0.55, heatmap_bgr, 0.45, 0)
    cv2.imwrite(str(overlay_path), overlay)


def verdict(stats):
    """Pass/fail decision for MoSaiC viability based on aggregate stats."""
    top50 = np.mean([s["top_50pct"] for s in stats])
    top25 = np.mean([s["top_25pct"] for s in stats])
    gini = np.mean([s["gini"] for s in stats])

    if top50 >= 0.70 and gini >= 0.35:
        return "PASS", (
            f"Saliency is highly concentrated (top 50% of tokens carry "
            f"{top50:.1%} of total signal, Gini={gini:.3f}). "
            f"SWTP can drop low-saliency tokens with minimal information loss. "
            f"MoSaiC is viable — proceed to implementation."
        )
    elif top50 >= 0.60 and gini >= 0.25:
        return "MARGINAL", (
            f"Saliency is moderately concentrated (top 50% carry {top50:.1%}, "
            f"Gini={gini:.3f}). SWTP may work with conservative compression "
            f"(keep top 70%+). Worth trying but expect smaller gains than ideal."
        )
    else:
        return "FAIL", (
            f"Saliency is too uniform (top 50% only carry {top50:.1%}, "
            f"Gini={gini:.3f}). Pruning low-saliency tokens would lose too much "
            f"information. Consider alternative signals: attention rollout from "
            f"DiT, or per-channel motion analysis."
        )


def main():
    p = argparse.ArgumentParser(description="MoSaiC saliency sanity check")
    p.add_argument("--video", required=True, help="Path to generated .mp4")
    p.add_argument("--patch_size", type=int, default=16,
                   help="Patch size in pixels for spatial pooling (default 16; matches DiT 8x VAE × 2x patch)")
    p.add_argument("--window_frames", type=int, default=12,
                   help="Temporal window for saliency aggregation (default 12 = 1 chunk @ 16fps)")
    p.add_argument("--probe_times", type=float, nargs="+",
                   default=[3.0, 8.0, 13.0, 18.0, 23.0, 27.0],
                   help="Timestamps (seconds) at which to compute saliency probes")
    p.add_argument("--output_dir", default=None,
                   help="Where to save visualizations (default: <video>_saliency_<source>/)")
    p.add_argument("--saliency_source", default="pixel",
                   choices=["pixel", "plucker_spatial", "plucker_temporal", "plucker_combined"],
                   help="Source of saliency signal: 'pixel' = video frame diff (original); "
                        "'plucker_spatial' = Plücker ray-field spatial gradient (world-model causal signal); "
                        "'plucker_temporal' = Plücker temporal gradient (camera motion intensity); "
                        "'plucker_combined' = spatial + temporal (most robust)")
    p.add_argument("--action_path", default=None,
                   help="Path to action data (poses.npy + intrinsics.npy). "
                        "Required when --saliency_source is plucker_*. "
                        "Example: examples/04")
    p.add_argument("--chunk_size", type=int, default=3,
                   help="Chunk size in latent frames (matches MoCE config, default 3)")
    args = p.parse_args()

    if not os.path.isfile(args.video):
        raise RuntimeError(f"Video not found: {args.video}")

    # Validate Plücker requirements
    if args.saliency_source.startswith("plucker") and args.action_path is None:
        raise RuntimeError(
            f"--saliency_source {args.saliency_source} requires --action_path "
            f"(e.g., --action_path examples/04)"
        )

    if args.output_dir is None:
        stem = Path(args.video).stem
        source_suffix = args.saliency_source.replace("plucker_", "p_")
        args.output_dir = str(Path(args.video).parent / f"{stem}_saliency_{source_suffix}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[load] Reading {args.video}")
    frames, fps, total = load_video_frames(args.video)
    duration = total / fps
    print(f"[load] {total} frames @ {fps:.2f} FPS ({duration:.1f}s)")

    print(f"[setup] patch_size={args.patch_size}px  "
          f"window={args.window_frames} frames  "
          f"source={args.saliency_source}  "
          f"probes={args.probe_times}")

    H, W = frames[0].shape[:2]
    print(f"[frame] {W}×{H}  → patch grid {W // args.patch_size}×{H // args.patch_size}  "
          f"= {(W // args.patch_size) * (H // args.patch_size)} tokens per frame")

    # Pre-compute Plücker if needed (one-time cost; reused for all probes)
    plucker_tensor = None
    lat_f = None
    if args.saliency_source.startswith("plucker"):
        print(f"[plucker] Loading and preprocessing camera signals from {args.action_path}")
        plucker_tensor, lat_f, wasd = load_plucker_full_video(
            args.action_path, H, W, total, chunk_size=args.chunk_size,
        )
        video_lat_fps = fps / 4  # vae temporal stride
        print(f"[plucker] {plucker_tensor.shape}  lat_f={lat_f}  "
              f"video_lat_fps={video_lat_fps:.2f}  "
              f"WASD={'available' if wasd is not None else 'absent (cam mode)'}")
    print()

    plucker_mode_map = {
        "plucker_spatial": "spatial",
        "plucker_temporal": "temporal",
        "plucker_combined": "spatial+temporal",
    }

    all_stats = []
    for t in args.probe_times:
        # Reference frame for overlay (always from video)
        ref_idx = min(int(round(t * fps)), len(frames) - 1)
        ref_frame = frames[ref_idx]

        if args.saliency_source == "pixel":
            sal, ref = compute_saliency_at_time(frames, fps, t,
                                                 window_frames=args.window_frames,
                                                 patch_size=args.patch_size)
        else:
            mode = plucker_mode_map[args.saliency_source]
            # Latent-frame window: 1 chunk = chunk_size latent frames
            window_lat = args.chunk_size
            sal, central_lat = compute_plucker_saliency_at_time(
                plucker_tensor, lat_f, t, fps, fps / 4,
                window_lat_frames=window_lat,
                patch_size=args.patch_size,
                mode=mode,
            )
            ref = ref_frame  # use video frame for visualization

        if sal is None:
            print(f"  [probe] t={t}s  SKIP (insufficient frames)")
            continue
        stats = saliency_stats(sal)
        stats["time_s"] = t
        all_stats.append(stats)

        out = Path(args.output_dir) / f"saliency_t{int(t*10):03d}s.jpg"
        overlay = Path(args.output_dir) / f"saliency_t{int(t*10):03d}s_overlay.jpg"
        visualize_saliency(sal, ref, out, overlay)

        print(f"  [probe] t={t:5.1f}s  "
              f"top10%={stats['top_10pct']:.3f}  "
              f"top25%={stats['top_25pct']:.3f}  "
              f"top50%={stats['top_50pct']:.3f}  "
              f"Gini={stats['gini']:.3f}  "
              f"CV={stats['cv']:.3f}")

    # ── Summary ─────────────────────────────────────────────────
    print()
    print("═" * 70)
    print("  AGGREGATE STATISTICS")
    print("═" * 70)
    if all_stats:
        for key in ["top_10pct", "top_25pct", "top_50pct", "gini", "cv"]:
            vals = [s[key] for s in all_stats]
            print(f"  {key:<20} mean={np.mean(vals):.4f}  "
                  f"std={np.std(vals):.4f}  "
                  f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    print()
    print("═" * 70)
    print("  MoSaiC VIABILITY VERDICT")
    print("═" * 70)
    if all_stats:
        v, msg = verdict(all_stats)
        print(f"  {v}: {msg}")
    else:
        v, msg = "ERROR", "No valid probes"
        print(f"  {v}: {msg}")

    # ── Save JSON ────────────────────────────────────────────────
    out_json = Path(args.output_dir) / "saliency_stats.json"
    summary = {
        "video": args.video,
        "saliency_source": args.saliency_source,
        "action_path": args.action_path,
        "fps": fps,
        "total_frames": total,
        "duration_s": duration,
        "frame_size": [W, H],
        "patch_size": args.patch_size,
        "window_frames": args.window_frames,
        "probes": all_stats,
        "aggregate": {
            key: {
                "mean": float(np.mean([s[key] for s in all_stats])),
                "std": float(np.std([s[key] for s in all_stats])),
            } for key in ["top_10pct", "top_25pct", "top_50pct", "gini", "cv"]
        } if all_stats else {},
        "verdict": v,
        "verdict_message": msg,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"  Heatmaps: {args.output_dir}/saliency_t*.jpg")
    print(f"  Overlays: {args.output_dir}/saliency_t*_overlay.jpg")
    print(f"  JSON:     {out_json}")
    print("═" * 70)


if __name__ == "__main__":
    main()
