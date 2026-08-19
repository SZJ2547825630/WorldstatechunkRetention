"""
Optical flow evaluation for LingBot-World generated videos.

Checks three things:
  1. Flow magnitude per frame — detects sudden jumps caused by KV eviction artifacts
  2. Flow direction vs camera poses — checks if camera control is obeyed
  3. Temporal smoothness — detects flickering / discontinuities

Usage:
  # Basic (no camera reference):
  python eval_flow.py --video output/xxx.mp4

  # With camera reference (recommended):
  python eval_flow.py --video output/xxx.mp4 \
      --poses examples/03/poses.npy \
      --intrinsics examples/03/intrinsics.npy

  # Aggressive anomaly threshold:
  python eval_flow.py --video output/xxx.mp4 --anomaly_zscore 2.0
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ── RAFT (preferred) ────────────────────────────────────────────────────────
try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    _RAFT_AVAILABLE = True
except ImportError:
    _RAFT_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# Video I/O
# ════════════════════════════════════════════════════════════════════════════

def load_frames(video_path: str, max_frames: int = None):
    """Return list of (H, W, 3) uint8 RGB frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


# ════════════════════════════════════════════════════════════════════════════
# Optical flow backends
# ════════════════════════════════════════════════════════════════════════════

class RAFTFlow:
    def __init__(self, device):
        self.device = device
        weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=weights).to(device).eval()
        self.transforms = weights.transforms()

    @torch.no_grad()
    def __call__(self, frame1: np.ndarray, frame2: np.ndarray):
        """Return flow (H, W, 2) numpy array (dx, dy in pixels)."""
        def to_tensor(f):
            t = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float()
            return t.to(self.device)
        t1, t2 = self.transforms(to_tensor(frame1), to_tensor(frame2))
        flows = self.model(t1, t2)
        flow = flows[-1][0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        return flow


class FarnebackFlow:
    """CPU fallback using OpenCV Farnebäck."""
    def __call__(self, frame1: np.ndarray, frame2: np.ndarray):
        g1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        return flow  # (H, W, 2)


def build_flow_estimator(device):
    if _RAFT_AVAILABLE:
        print("[flow] Using RAFT (torchvision)")
        return RAFTFlow(device)
    print("[flow] RAFT not available, falling back to Farnebäck (OpenCV)")
    return FarnebackFlow()


# ════════════════════════════════════════════════════════════════════════════
# Per-frame metrics
# ════════════════════════════════════════════════════════════════════════════

def flow_magnitude(flow: np.ndarray) -> float:
    """Mean L2 magnitude of the flow field."""
    return float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())


def flow_direction_entropy(flow: np.ndarray, bins: int = 36) -> float:
    """
    Shannon entropy of the flow direction histogram.
    Low entropy → coherent motion (camera pan/zoom).
    High entropy → chaotic / unstable motion.
    """
    angles = np.arctan2(flow[..., 1], flow[..., 0])
    hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
    hist = hist.astype(float) + 1e-8
    hist /= hist.sum()
    return float(-(hist * np.log(hist)).sum())


def flow_smoothness(mag_series: list) -> float:
    """Mean absolute frame-to-frame change in flow magnitude."""
    if len(mag_series) < 2:
        return 0.0
    diffs = [abs(mag_series[i] - mag_series[i - 1]) for i in range(1, len(mag_series))]
    return float(np.mean(diffs))


# ════════════════════════════════════════════════════════════════════════════
# Camera-pose reference (optional)
# ════════════════════════════════════════════════════════════════════════════

def expected_translation_magnitude(poses: np.ndarray, frame_idx: int) -> float:
    """
    Rough proxy for expected optical flow magnitude from camera translation.
    Uses the norm of the relative translation vector between frame i and i+1.
    No depth → can't get pixel units, but the relative ranking is valid.
    """
    if frame_idx + 1 >= len(poses):
        return 0.0
    t_curr = poses[frame_idx, :3, 3]
    t_next = poses[frame_idx + 1, :3, 3]
    return float(np.linalg.norm(t_next - t_curr))


def pose_flow_correlation(actual_mags: list, poses: np.ndarray) -> float:
    """
    Pearson correlation between actual flow magnitudes and camera translation
    magnitudes. Close to 1.0 → camera control is being respected.
    """
    n = min(len(actual_mags), len(poses) - 1)
    expected = [expected_translation_magnitude(poses, i) for i in range(n)]
    actual = actual_mags[:n]
    if np.std(expected) < 1e-8 or np.std(actual) < 1e-8:
        return float("nan")
    corr = np.corrcoef(actual, expected)[0, 1]
    return float(corr)


# ════════════════════════════════════════════════════════════════════════════
# Anomaly detection
# ════════════════════════════════════════════════════════════════════════════

def detect_anomaly_frames(mag_series: list, zscore_thr: float = 2.5) -> list:
    """
    Frames where flow magnitude deviates > zscore_thr standard deviations
    from the rolling mean. Returns list of (frame_idx, magnitude, zscore).
    """
    arr = np.array(mag_series)
    mean, std = arr.mean(), arr.std()
    if std < 1e-8:
        return []
    zscores = (arr - mean) / std
    anomalies = [(int(i), float(arr[i]), float(zscores[i]))
                 for i in range(len(arr)) if abs(zscores[i]) > zscore_thr]
    return anomalies


# ════════════════════════════════════════════════════════════════════════════
# ASCII plot (no matplotlib dependency)
# ════════════════════════════════════════════════════════════════════════════

def ascii_plot(values: list, title: str, width: int = 60, height: int = 10):
    if not values:
        return
    vmin, vmax = min(values), max(values)
    vrange = vmax - vmin or 1.0
    print(f"\n  {title}")
    print(f"  max={vmax:.3f}")
    rows = []
    for row in range(height):
        threshold = vmax - (row / (height - 1)) * vrange
        line = ""
        for v in values:
            line += "█" if v >= threshold else " "
        rows.append(line)
    for row in rows:
        print(f"  |{row}|")
    print(f"  min={vmin:.3f}")
    # x-axis tick marks every 10% of frames
    n = len(values)
    tick_line = " " * 3
    for i in range(width):
        fi = int(i * n / width)
        tick_line += "|" if fi % max(1, n // 10) == 0 else "-"
    print(f"  {tick_line}")
    print(f"  frame 0{' ' * (width - 10)}frame {n}")


# ════════════════════════════════════════════════════════════════════════════
# Save per-frame flow visualization
# ════════════════════════════════════════════════════════════════════════════

def flow_to_bgr(flow: np.ndarray) -> np.ndarray:
    """Convert flow field to HSV color wheel image (for visualization)."""
    h, w = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2        # hue = direction
    hsv[..., 1] = 255                            # full saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Optical flow evaluation for LingBot-World videos")
    parser.add_argument("--video", required=True, help="Path to generated .mp4 video")
    parser.add_argument("--poses", default=None, help="Optional: path to poses.npy for camera reference")
    parser.add_argument("--intrinsics", default=None, help="Optional: path to intrinsics.npy")
    parser.add_argument("--output_dir", default=None, help="Directory to save results (default: next to video)")
    parser.add_argument("--anomaly_zscore", type=float, default=2.5,
                        help="Z-score threshold for anomaly frame detection (lower = more sensitive)")
    parser.add_argument("--save_flow_frames", action="store_true",
                        help="Save color-coded flow images for anomaly frames")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of frames to process")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── output dir ──────────────────────────────────────────────────────────
    if args.output_dir is None:
        args.output_dir = os.path.splitext(args.video)[0] + "_flow_eval"
    os.makedirs(args.output_dir, exist_ok=True)

    # ── load video ──────────────────────────────────────────────────────────
    print(f"[load] Reading {args.video}")
    frames, fps = load_frames(args.video, max_frames=args.max_frames)
    print(f"[load] {len(frames)} frames @ {fps:.1f} FPS  ({len(frames)/fps:.1f}s)")

    if len(frames) < 2:
        print("[error] Need at least 2 frames.")
        sys.exit(1)

    # ── load optional camera data ────────────────────────────────────────────
    poses = np.load(args.poses) if args.poses else None
    if poses is not None:
        print(f"[poses] Loaded {poses.shape[0]} poses from {args.poses}")

    # ── build flow estimator ─────────────────────────────────────────────────
    estimator = build_flow_estimator(args.device)

    # ── compute per-frame flow ───────────────────────────────────────────────
    magnitudes = []
    entropies = []
    flows = []

    print(f"[flow] Computing optical flow for {len(frames)-1} frame pairs...")
    for i in range(len(frames) - 1):
        flow = estimator(frames[i], frames[i + 1])
        flows.append(flow)
        magnitudes.append(flow_magnitude(flow))
        entropies.append(flow_direction_entropy(flow))
        if (i + 1) % 20 == 0 or i == len(frames) - 2:
            print(f"  {i+1}/{len(frames)-1} frames processed")

    # ── aggregate metrics ────────────────────────────────────────────────────
    smoothness = flow_smoothness(magnitudes)
    anomalies = detect_anomaly_frames(magnitudes, zscore_thr=args.anomaly_zscore)

    corr = None
    if poses is not None:
        corr = pose_flow_correlation(magnitudes, poses)

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  OPTICAL FLOW EVALUATION SUMMARY")
    print("═" * 60)
    print(f"  Video          : {os.path.basename(args.video)}")
    print(f"  Frames         : {len(frames)}  ({len(frames)/fps:.1f}s @ {fps:.0f}FPS)")
    print(f"  Flow estimator : {'RAFT' if _RAFT_AVAILABLE else 'Farnebäck'}")
    print()
    print(f"  Avg flow magnitude   : {np.mean(magnitudes):.3f} px")
    print(f"  Max flow magnitude   : {np.max(magnitudes):.3f} px")
    print(f"  Temporal smoothness  : {smoothness:.3f}  (lower = smoother)")
    print(f"  Avg direction entropy: {np.mean(entropies):.3f}  (lower = more coherent)")
    if corr is not None:
        print(f"  Pose-flow correlation: {corr:.3f}  (closer to 1.0 = camera control obeyed)")
    print()

    # ── anomaly report ────────────────────────────────────────────────────────
    if anomalies:
        print(f"  ⚠  {len(anomalies)} anomaly frames detected (|z| > {args.anomaly_zscore}):")
        for fidx, mag, z in anomalies[:10]:
            time_s = fidx / fps
            print(f"     frame {fidx:4d}  ({time_s:.2f}s)  magnitude={mag:.3f}  z={z:+.2f}")
        if len(anomalies) > 10:
            print(f"     ... and {len(anomalies)-10} more")
    else:
        print(f"  ✓  No anomaly frames detected (|z| > {args.anomaly_zscore})")

    # ── verdict ───────────────────────────────────────────────────────────────
    print()
    verdict_items = []
    if len(anomalies) == 0:
        verdict_items.append("✓ temporal stability: OK")
    elif len(anomalies) <= 2:
        verdict_items.append("⚠ temporal stability: minor artifacts")
    else:
        verdict_items.append("✗ temporal stability: likely degraded")

    if smoothness < 1.0:
        verdict_items.append("✓ smoothness: OK")
    elif smoothness < 3.0:
        verdict_items.append("⚠ smoothness: some jitter")
    else:
        verdict_items.append("✗ smoothness: significant jitter")

    if corr is not None:
        if corr > 0.6:
            verdict_items.append(f"✓ camera control: obeyed (r={corr:.2f})")
        elif corr > 0.3:
            verdict_items.append(f"⚠ camera control: weak alignment (r={corr:.2f})")
        else:
            verdict_items.append(f"✗ camera control: not following poses (r={corr:.2f})")

    for item in verdict_items:
        print(f"  {item}")
    print("═" * 60)

    # ── ASCII plot ────────────────────────────────────────────────────────────
    ascii_plot(magnitudes, "Flow Magnitude per Frame", width=min(60, len(magnitudes)))

    # ── save flow visualization for anomaly frames ────────────────────────────
    if args.save_flow_frames and anomalies:
        vis_dir = os.path.join(args.output_dir, "anomaly_frames")
        os.makedirs(vis_dir, exist_ok=True)
        for fidx, mag, z in anomalies:
            if fidx < len(flows):
                vis = flow_to_bgr(flows[fidx])
                frame_overlay = cv2.cvtColor(frames[fidx], cv2.COLOR_RGB2BGR)
                combined = np.concatenate([frame_overlay, vis], axis=1)
                path = os.path.join(vis_dir, f"frame_{fidx:04d}_z{z:+.2f}.jpg")
                cv2.imwrite(path, combined)
        print(f"\n[save] Flow visualizations saved to {vis_dir}/")

    # ── save JSON results ─────────────────────────────────────────────────────
    results = {
        "video": args.video,
        "num_frames": len(frames),
        "fps": fps,
        "duration_s": len(frames) / fps,
        "flow_estimator": "RAFT" if _RAFT_AVAILABLE else "Farneback",
        "metrics": {
            "avg_magnitude": float(np.mean(magnitudes)),
            "max_magnitude": float(np.max(magnitudes)),
            "std_magnitude": float(np.std(magnitudes)),
            "temporal_smoothness": smoothness,
            "avg_direction_entropy": float(np.mean(entropies)),
            "pose_flow_correlation": corr,
        },
        "anomalies": [
            {"frame": fidx, "time_s": fidx / fps, "magnitude": mag, "zscore": z}
            for fidx, mag, z in anomalies
        ],
        "per_frame_magnitude": magnitudes,
    }
    json_path = os.path.join(args.output_dir, "flow_eval.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] Results saved to {json_path}")


if __name__ == "__main__":
    main()
