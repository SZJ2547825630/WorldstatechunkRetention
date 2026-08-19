"""
Semantic evaluation for LingBot-World generated videos.

Quantifies four failure modes that optical-flow metrics cannot see:

  1. Whole-image identity drift     — DINOv2 CLS cos_sim(frame_0, frame_t)
  2. Subject-region identity drift  — DINOv2 patch ROI cos_sim(frame_0, frame_t)
                                      (ROI focuses on subject-occupied region;
                                       suppresses camera-motion noise that drowns
                                       CLS-level signal)
  3. Prompt-adherence decay         — CLIP cos_sim(frame_t, prompt)
  4. Perceptual color/texture drift — LPIPS(frame_0, frame_t)  &  LPIPS(t, t+1)
                                      (sensitive to color/texture changes that
                                       CLS/CLIP features average away)

Usage:
  # Single video
  python eval_semantics.py \
      --video output/example00_MoCE.mp4 \
      --prompt_file examples/00/prompt.txt

  # Compare two videos
  python eval_semantics.py \
      --video output/example00_MoCE.mp4 output/example00_baseline.mp4 \
      --prompt_file examples/00/prompt.txt

  # Use local model weights
  python eval_semantics.py --video <...> --prompt_file <...> \
      --dino_model /DATA/YuanZhen/Lingbot/dinov2-base \
      --clip_model /DATA/YuanZhen/Lingbot/clip-vit-base-patch32

  # Choose ROI for patch-level DINOv2 (default 'subject' = bottom 60% + middle 60%)
  python eval_semantics.py ... --roi bottom-center
  python eval_semantics.py ... --roi 0.4,1.0,0.2,0.8     # custom fractions
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
)

try:
    import lpips
    HAVE_LPIPS = True
except ImportError:
    HAVE_LPIPS = False
    lpips = None


# ════════════════════════════════════════════════════════════════════════════
# ROI presets — fractional (row_start, row_end, col_start, col_end)
# ════════════════════════════════════════════════════════════════════════════

ROI_PRESETS = {
    "full":          (0.0, 1.0, 0.0, 1.0),
    "bottom-center": (0.5, 1.0, 0.25, 0.75),
    "bottom-half":   (0.5, 1.0, 0.0, 1.0),
    "center":        (0.25, 0.75, 0.25, 0.75),
    "subject":       (0.4, 1.0, 0.2, 0.8),
}


def parse_roi(s):
    if s in ROI_PRESETS:
        return ROI_PRESETS[s]
    parts = s.split(",")
    if len(parts) != 4:
        raise ValueError(
            f"ROI must be a preset name {list(ROI_PRESETS)} "
            f"or 'r1,r2,c1,c2' fractions, got: {s}")
    return tuple(float(x) for x in parts)


def fractional_roi_to_grid(roi_frac, grid):
    r1f, r2f, c1f, c2f = roi_frac
    r1 = max(0, min(grid - 1, int(r1f * grid)))
    r2 = max(r1 + 1, min(grid, int(round(r2f * grid))))
    c1 = max(0, min(grid - 1, int(c1f * grid)))
    c2 = max(c1 + 1, min(grid, int(round(c2f * grid))))
    return r1, r2, c1, c2


# ════════════════════════════════════════════════════════════════════════════
# Video I/O
# ════════════════════════════════════════════════════════════════════════════

def load_frames(video_path, max_frames=None, sample_every=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        idx += 1
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


# ════════════════════════════════════════════════════════════════════════════
# Model loaders
# ════════════════════════════════════════════════════════════════════════════

def load_dinov2(device, model_name="facebook/dinov2-base"):
    print(f"[model] Loading DINOv2 ({model_name})")
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return model, proc


def load_clip(device, model_name="openai/clip-vit-base-patch32"):
    print(f"[model] Loading CLIP ({model_name})")
    proc = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    return model, proc


def load_lpips_model(device, net="alex"):
    if not HAVE_LPIPS:
        raise RuntimeError("lpips package not installed: pip install lpips")
    print(f"[model] Loading LPIPS ({net} backbone)")
    model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def dino_features_full(model, proc, frames, device, batch_size=16):
    """
    Returns:
      cls    [N, D]            L2-normalised CLS token
      patches [N, grid, grid, D]  raw patch tokens (un-normalised, ROI-pooled later)
      grid   int               grid side length (e.g. 16 for 224x224 / 14)
    """
    cls_chunks, patch_chunks = [], []
    grid = None
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        inputs = proc(images=batch, return_tensors="pt").to(device)
        out = model(**inputs).last_hidden_state           # [B, 1+P, D]
        cls = F.normalize(out[:, 0], dim=-1)              # [B, D]
        patches = out[:, 1:]                              # [B, P, D]
        if grid is None:
            P = patches.shape[1]
            grid = int(P ** 0.5)
            if grid * grid != P:
                raise RuntimeError(f"Non-square patch grid: {P} patches")
        patches = patches.reshape(-1, grid, grid, patches.shape[-1])
        cls_chunks.append(cls.cpu())
        patch_chunks.append(patches.cpu())
    return torch.cat(cls_chunks, dim=0), torch.cat(patch_chunks, dim=0), grid


def roi_pool(patches, roi):
    """Mean-pool patch features within ROI box, then L2-normalise."""
    r1, r2, c1, c2 = roi
    pooled = patches[:, r1:r2, c1:c2, :].mean(dim=(1, 2))   # [N, D]
    return F.normalize(pooled, dim=-1)


@torch.no_grad()
def clip_text_similarity(model, proc, frames, prompt, device, batch_size=16):
    text_inputs = proc(text=[prompt], return_tensors="pt",
                       padding=True, truncation=True).to(device)
    text_feat = F.normalize(model.get_text_features(**text_inputs), dim=-1)
    sims = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        img_inputs = proc(images=batch, return_tensors="pt").to(device)
        img_feat = F.normalize(model.get_image_features(**img_inputs), dim=-1)
        sims.append((img_feat @ text_feat.T).squeeze(-1).cpu())
    return torch.cat(sims, dim=0).numpy()


@torch.no_grad()
def compute_lpips_series(model, frames, device, batch_size=4):
    """
    LPIPS distances (lower = more similar).
    Returns (anchor_dist [N], f2f_dist [N-1]).
    """
    def to_tensor(pil_img):
        arr = np.asarray(pil_img, dtype=np.float32) / 127.5 - 1.0    # → [-1, 1]
        return torch.from_numpy(arr).permute(2, 0, 1)                # [C, H, W]

    tensors = torch.stack([to_tensor(f) for f in frames])            # [N, 3, H, W]
    n = len(tensors)
    anchor = tensors[0:1].to(device)

    # Anchor distance
    anchor_dists = []
    for i in range(0, n, batch_size):
        batch = tensors[i:i + batch_size].to(device)
        anc = anchor.expand(batch.shape[0], -1, -1, -1)
        d = model(anc, batch).view(-1)
        anchor_dists.append(d.cpu())
    anchor_dists = torch.cat(anchor_dists, dim=0).numpy()

    # Frame-to-frame distance
    f2f_dists = []
    if n >= 2:
        for i in range(0, n - 1, batch_size):
            end = min(i + batch_size, n - 1)
            a = tensors[i:end].to(device)
            b = tensors[i + 1:end + 1].to(device)
            d = model(a, b).view(-1)
            f2f_dists.append(d.cpu())
        f2f_dists = torch.cat(f2f_dists, dim=0).numpy()
    else:
        f2f_dists = np.array([])

    return anchor_dists, f2f_dists


# ════════════════════════════════════════════════════════════════════════════
# Derived metrics
# ════════════════════════════════════════════════════════════════════════════

def anchor_similarity(features):
    """cos_sim(features[0], features[t]) for each t."""
    return (features @ features[0:1].T).squeeze(-1).numpy()


def short_term_consistency(features, window=16):
    if len(features) <= window:
        return np.array([])
    return (features[:-window] * features[window:]).sum(dim=-1).numpy()


def thirds(arr):
    n = len(arr)
    return arr[:n // 3], arr[n // 3:2 * n // 3], arr[2 * n // 3:]


def aggregate(cls_anchor, roi_anchor, short_term, clip_sim,
              lpips_anchor=None, lpips_f2f=None):
    a1, a2, a3 = thirds(cls_anchor)
    r1, r2, r3 = thirds(roi_anchor)
    c1, _, c3  = thirds(clip_sim)

    m = {
        # DINOv2 CLS (global identity)
        "dino_anchor_first_third":      float(a1.mean()),
        "dino_anchor_middle_third":     float(a2.mean()),
        "dino_anchor_last_third":       float(a3.mean()),
        "dino_anchor_final":            float(cls_anchor[-1]),
        "dino_anchor_decay_slope":      float(np.polyfit(np.arange(len(cls_anchor)), cls_anchor, 1)[0]),
        "dino_short_term_mean":         float(short_term.mean()) if short_term.size else None,
        "dino_short_term_min":          float(short_term.min())  if short_term.size else None,

        # DINOv2 ROI (subject-region identity)
        "dino_roi_anchor_first_third":  float(r1.mean()),
        "dino_roi_anchor_middle_third": float(r2.mean()),
        "dino_roi_anchor_last_third":   float(r3.mean()),
        "dino_roi_anchor_final":        float(roi_anchor[-1]),
        "dino_roi_anchor_decay_slope":  float(np.polyfit(np.arange(len(roi_anchor)), roi_anchor, 1)[0]),

        # CLIP-text
        "clip_text_first_third":        float(c1.mean()),
        "clip_text_last_third":         float(c3.mean()),
        "clip_text_drift":              float(c3.mean() - c1.mean()),
        "clip_text_decay_slope":        float(np.polyfit(np.arange(len(clip_sim)), clip_sim, 1)[0]),
        "clip_text_mean":               float(clip_sim.mean()),
    }

    if lpips_anchor is not None and len(lpips_anchor) > 0:
        la1, _, la3 = thirds(lpips_anchor)
        m.update({
            "lpips_anchor_first_third": float(la1.mean()),
            "lpips_anchor_last_third":  float(la3.mean()),
            "lpips_anchor_final":       float(lpips_anchor[-1]),
            "lpips_anchor_slope":       float(np.polyfit(np.arange(len(lpips_anchor)), lpips_anchor, 1)[0]),
            "lpips_anchor_mean":        float(lpips_anchor.mean()),
        })
        if lpips_f2f is not None and len(lpips_f2f) > 0:
            m.update({
                "lpips_f2f_mean":       float(lpips_f2f.mean()),
                "lpips_f2f_max":        float(lpips_f2f.max()),
                "lpips_f2f_std":        float(lpips_f2f.std()),
            })
    return m


# ════════════════════════════════════════════════════════════════════════════
# ASCII overlay plot
# ════════════════════════════════════════════════════════════════════════════

def ascii_overlay(curves, labels, title, width=70, height=14, ymin=None, ymax=None):
    if not curves or len(curves[0]) == 0:
        return
    resampled = []
    for c in curves:
        c = np.asarray(c)
        if len(c) >= width:
            idx = np.linspace(0, len(c) - 1, width).astype(int)
            resampled.append(c[idx])
        else:
            pad = np.full(width - len(c), c[-1])
            resampled.append(np.concatenate([c, pad]))

    if ymin is None:
        ymin = min(c.min() for c in resampled)
    if ymax is None:
        ymax = max(c.max() for c in resampled)
    yrange = (ymax - ymin) or 1.0

    chars = ['●', '○', '×', '+']
    canvas = [[' '] * width for _ in range(height)]
    for ci, c in enumerate(resampled):
        ch = chars[ci % len(chars)]
        for x, v in enumerate(c):
            y = int((ymax - v) / yrange * (height - 1))
            y = max(0, min(height - 1, y))
            canvas[y][x] = '◆' if canvas[y][x] not in (' ', ch) else ch

    print(f"\n  {title}")
    print(f"  ymax={ymax:.4f}")
    for row in canvas:
        print(f"  |{''.join(row)}|")
    print(f"  ymin={ymin:.4f}")
    legend = "  legend: " + "  ".join(f"{chars[i % len(chars)]}={lbl}"
                                       for i, lbl in enumerate(labels))
    if len(curves) > 1:
        legend += "  ◆=overlap"
    print(legend)
    n_orig = len(curves[0])
    print("  " + "0" + " " * (width - 9) + f"frame {n_orig}")


# ════════════════════════════════════════════════════════════════════════════
# Per-video evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate_video(video_path, prompt, args, dino, clip, lpips_model):
    print(f"\n[load] {video_path}")
    frames, fps = load_frames(video_path, max_frames=args.max_frames,
                              sample_every=args.sample_every)
    print(f"[load] {len(frames)} frames @ {fps:.2f} FPS  "
          f"(sampled every {args.sample_every})")
    if len(frames) < 2:
        raise RuntimeError("Need at least 2 frames")

    print(f"[dino] Extracting CLS + patch features for {len(frames)} frames")
    cls_feat, patch_feat, grid = dino_features_full(
        *dino[:2], frames, args.device, args.batch_size)

    roi_frac = parse_roi(args.roi)
    roi_grid = fractional_roi_to_grid(roi_frac, grid)
    print(f"[dino] ROI '{args.roi}' on {grid}x{grid} patch grid → "
          f"rows[{roi_grid[0]}:{roi_grid[1]}]  cols[{roi_grid[2]}:{roi_grid[3]}]  "
          f"({(roi_grid[1]-roi_grid[0])*(roi_grid[3]-roi_grid[2])} patches)")
    roi_feat = roi_pool(patch_feat, roi_grid)

    print(f"[clip] Computing prompt similarity")
    csim = clip_text_similarity(*clip[:2], frames, prompt,
                                args.device, args.batch_size)

    lpips_anchor, lpips_f2f = None, None
    if lpips_model is not None:
        print(f"[lpips] Computing perceptual distances (batch_size={args.lpips_batch_size})")
        lpips_anchor, lpips_f2f = compute_lpips_series(
            lpips_model, frames, args.device, args.lpips_batch_size)

    cls_asim = anchor_similarity(cls_feat)
    roi_asim = anchor_similarity(roi_feat)
    stcs     = short_term_consistency(cls_feat, window=args.consistency_window)
    eff_fps  = fps / args.sample_every

    result = {
        "video":               video_path,
        "num_frames_sampled":  len(frames),
        "effective_fps":       eff_fps,
        "duration_s":          len(frames) / eff_fps,
        "prompt":              prompt,
        "sample_every":        args.sample_every,
        "consistency_window":  args.consistency_window,
        "roi_preset":          args.roi,
        "roi_grid":            list(roi_grid),
        "dino_grid_size":      grid,
        "dino_anchor_sim":     cls_asim.tolist(),
        "dino_roi_anchor_sim": roi_asim.tolist(),
        "dino_short_term_sim": stcs.tolist(),
        "clip_text_sim":       csim.tolist(),
        "metrics":             aggregate(cls_asim, roi_asim, stcs, csim,
                                         lpips_anchor, lpips_f2f),
    }
    if lpips_anchor is not None:
        result["lpips_anchor_dist"] = lpips_anchor.tolist()
        result["lpips_f2f_dist"]    = lpips_f2f.tolist() if lpips_f2f is not None else None
    return result


# ════════════════════════════════════════════════════════════════════════════
# Reports
# ════════════════════════════════════════════════════════════════════════════

def print_single(r):
    m = r["metrics"]
    name = os.path.basename(r["video"])
    print("\n" + "═" * 72)
    print(f"  SEMANTIC EVALUATION — {name}")
    print("═" * 72)
    print(f"  Frames sampled: {r['num_frames_sampled']}  Duration: {r['duration_s']:.1f}s")
    print(f"  ROI: '{r['roi_preset']}'  grid={r['dino_grid_size']}x{r['dino_grid_size']}  "
          f"box=rows[{r['roi_grid'][0]}:{r['roi_grid'][1]}], cols[{r['roi_grid'][2]}:{r['roi_grid'][3]}]")
    print(f"  Prompt: {r['prompt'][:60]}{'…' if len(r['prompt']) > 60 else ''}")
    print()
    print("  ─── DINOv2 CLS  (whole-image identity, higher = less drift) ───")
    print(f"    anchor first 1/3       : {m['dino_anchor_first_third']:.4f}")
    print(f"    anchor middle 1/3      : {m['dino_anchor_middle_third']:.4f}")
    print(f"    anchor last 1/3        : {m['dino_anchor_last_third']:.4f}")
    print(f"    anchor final frame     : {m['dino_anchor_final']:.4f}")
    print(f"    decay slope per frame  : {m['dino_anchor_decay_slope']:+.5f}")
    if m['dino_short_term_mean'] is not None:
        print(f"    short-term (±{r['consistency_window']}f) mean/min: "
              f"{m['dino_short_term_mean']:.4f} / {m['dino_short_term_min']:.4f}")
    print()
    print(f"  ─── DINOv2 ROI-patch  (subject-region identity, higher = less drift) ───")
    print(f"    ROI anchor first 1/3   : {m['dino_roi_anchor_first_third']:.4f}")
    print(f"    ROI anchor middle 1/3  : {m['dino_roi_anchor_middle_third']:.4f}")
    print(f"    ROI anchor last 1/3    : {m['dino_roi_anchor_last_third']:.4f}")
    print(f"    ROI anchor final frame : {m['dino_roi_anchor_final']:.4f}")
    print(f"    decay slope per frame  : {m['dino_roi_anchor_decay_slope']:+.5f}")
    print()
    print("  ─── CLIP-text  (prompt adherence, higher = closer to prompt) ───")
    print(f"    first 1/3              : {m['clip_text_first_third']:.4f}")
    print(f"    last 1/3               : {m['clip_text_last_third']:.4f}")
    print(f"    drift (last − first)   : {m['clip_text_drift']:+.4f}")
    print(f"    decay slope per frame  : {m['clip_text_decay_slope']:+.6f}")
    if "lpips_anchor_mean" in m:
        print()
        print("  ─── LPIPS perceptual distance  (LOWER = more similar) ───")
        print(f"    anchor first 1/3       : {m['lpips_anchor_first_third']:.4f}")
        print(f"    anchor last 1/3        : {m['lpips_anchor_last_third']:.4f}")
        print(f"    anchor final frame     : {m['lpips_anchor_final']:.4f}")
        print(f"    anchor slope per frame : {m['lpips_anchor_slope']:+.5f}")
        if m.get("lpips_f2f_mean") is not None:
            print(f"    frame-to-frame mean    : {m['lpips_f2f_mean']:.4f}  "
                  f"max: {m['lpips_f2f_max']:.4f}  std: {m['lpips_f2f_std']:.4f}")

    ascii_overlay([np.array(r['dino_anchor_sim'])],
                  ['cls_anchor'],
                  "DINOv2 CLS  cos_sim(frame_0, frame_t)  (higher = better)")
    ascii_overlay([np.array(r['dino_roi_anchor_sim'])],
                  ['roi_anchor'],
                  f"DINOv2 ROI '{r['roi_preset']}'  cos_sim(frame_0, frame_t)  (higher = better)")
    ascii_overlay([np.array(r['clip_text_sim'])],
                  ['clip_text'],
                  "CLIP-text  cos_sim(frame_t, prompt)  (higher = better)")
    if r.get('lpips_anchor_dist') is not None:
        ascii_overlay([np.array(r['lpips_anchor_dist'])],
                      ['lpips_anchor'],
                      "LPIPS  dist(frame_0, frame_t)  (lower = better)")
    print("═" * 72)


def print_comparison(results):
    """Compare 2+ videos side-by-side. First video is the reference (Δ vs ref)."""
    names = [os.path.splitext(os.path.basename(r["video"]))[0] for r in results]
    metrics = [r["metrics"] for r in results]
    n = len(results)
    col_w = max(12, min(16, max(len(x) for x in names)))
    label_w = 40
    width = label_w + 2 + (col_w + 2) * n + 12

    def row(label, key, higher_is_better=True):
        vals = [m.get(key) for m in metrics]
        if any(v is None for v in vals):
            return
        ref = vals[0]
        best_i = int(np.argmax(vals) if higher_is_better else np.argmin(vals))
        cells = "  ".join(f"{v:{col_w}.4f}" for v in vals)
        delta = vals[best_i] - ref if best_i != 0 else 0.0
        # Prefer unique suffixes so seed42_ws_cr / seed42_ws_cr_tich don't both
        # truncate to "seed42_w".
        if best_i == 0:
            mark = "ref"
        else:
            mark = names[best_i]
            if len(mark) > 14:
                mark = mark[-14:]
        print(f"  {label:<{label_w}} {cells}  best={mark}")

    a = results[0]
    print("\n" + "═" * min(width, 120))
    print("  SEMANTIC COMPARISON  (first video = reference)")
    for i, name in enumerate(names):
        print(f"    [{i}] {name}")
    print(f"    ROI = '{a['roi_preset']}'  (rows[{a['roi_grid'][0]}:{a['roi_grid'][1]}], "
          f"cols[{a['roi_grid'][2]}:{a['roi_grid'][3]}] on {a['dino_grid_size']}x{a['dino_grid_size']} grid)")
    print("═" * min(width, 120))
    hdr = "  " + f"{'Metric':<{label_w}} " + "  ".join(f"{nm[:col_w]:>{col_w}}" for nm in names)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    print("  ─── DINOv2 CLS  (whole image, higher better) ───")
    row("  cls anchor first 1/3",       "dino_anchor_first_third",      True)
    row("  cls anchor middle 1/3",      "dino_anchor_middle_third",     True)
    row("  cls anchor last 1/3",        "dino_anchor_last_third",       True)
    row("  cls anchor final frame",     "dino_anchor_final",            True)
    row("  cls decay slope/frame",      "dino_anchor_decay_slope",      True)
    row("  short_term mean",            "dino_short_term_mean",         True)
    row("  short_term min",             "dino_short_term_min",          True)
    print()
    print("  ─── DINOv2 ROI patch  (subject region, higher better) ───")
    row("  ROI anchor first 1/3",       "dino_roi_anchor_first_third",  True)
    row("  ROI anchor middle 1/3",      "dino_roi_anchor_middle_third", True)
    row("  ROI anchor last 1/3",        "dino_roi_anchor_last_third",   True)
    row("  ROI anchor final frame",     "dino_roi_anchor_final",        True)
    row("  ROI decay slope/frame",      "dino_roi_anchor_decay_slope",  True)
    print()
    print("  ─── CLIP-text  (higher better) ───")
    row("  clip first 1/3",             "clip_text_first_third",        True)
    row("  clip last 1/3",              "clip_text_last_third",         True)
    row("  clip drift (last - first)",  "clip_text_drift",              True)
    row("  clip decay slope/frame",     "clip_text_decay_slope",        True)
    if "lpips_anchor_mean" in metrics[0]:
        print()
        print("  ─── LPIPS perceptual distance  (LOWER better) ───")
        row("  lpips anchor first 1/3",     "lpips_anchor_first_third", False)
        row("  lpips anchor last 1/3",      "lpips_anchor_last_third",  False)
        row("  lpips anchor final",         "lpips_anchor_final",       False)
        row("  lpips anchor slope/frame",   "lpips_anchor_slope",       False)
        row("  lpips anchor mean",          "lpips_anchor_mean",        False)
        row("  lpips f2f mean",             "lpips_f2f_mean",           False)
        row("  lpips f2f max",              "lpips_f2f_max",            False)
        row("  lpips f2f std",              "lpips_f2f_std",            False)

    ascii_overlay([np.array(r["dino_anchor_sim"]) for r in results],
                  names,
                  "DINOv2 CLS  cos_sim(frame_0, frame_t)  (higher = better)")
    ascii_overlay([np.array(r["dino_roi_anchor_sim"]) for r in results],
                  names,
                  f"DINOv2 ROI '{a['roi_preset']}'  cos_sim(frame_0, frame_t)  (higher = better)")
    ascii_overlay([np.array(r["clip_text_sim"]) for r in results],
                  names,
                  "CLIP-text  cos_sim(frame_t, prompt)  (higher = better)")
    if all(r.get("lpips_anchor_dist") is not None for r in results):
        ascii_overlay([np.array(r["lpips_anchor_dist"]) for r in results],
                      names,
                      "LPIPS  dist(frame_0, frame_t)  (LOWER = better)")
    print("═" * min(width, 120))


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="DINOv2 (CLS + patch-ROI) + CLIP-text + LPIPS evaluation")
    p.add_argument("--video", nargs="+", required=True,
                   help="One or more .mp4 paths. ≥2 paths triggers comparison mode "
                        "(first video is the reference).")
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt_file", default=None,
                   help="Read prompt from file (e.g. examples/00/prompt.txt)")
    p.add_argument("--output_dir", default=None,
                   help="Directory for results (default: <video>_semantics_eval/)")
    p.add_argument("--sample_every", type=int, default=1,
                   help="Sample every Nth frame to speed up (default: 1)")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size for DINOv2 / CLIP")
    p.add_argument("--consistency_window", type=int, default=16,
                   help="Frame distance for short-term DINO consistency (default 16 ≈ 1s @ 16fps)")
    p.add_argument("--dino_model", default="facebook/dinov2-base")
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    p.add_argument("--roi", default="subject",
                   help=f"DINOv2 patch ROI: preset {list(ROI_PRESETS)} "
                        f"or 'r1,r2,c1,c2' fractions. Default 'subject' (rows[0.4:1.0], cols[0.2:0.8])")
    p.add_argument("--no_lpips", action="store_true", help="Disable LPIPS computation")
    p.add_argument("--lpips_net", default="alex",
                   choices=["alex", "vgg", "squeeze"],
                   help="LPIPS backbone (default: alex, smallest+fastest)")
    p.add_argument("--lpips_batch_size", type=int, default=4,
                   help="LPIPS batch size — native-resolution frames can be large (default: 4)")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        p.error("--prompt or --prompt_file is required")

    for v in args.video:
        if not os.path.isfile(v):
            p.error(f"Video not found: {v}")

    try:
        parse_roi(args.roi)
    except ValueError as e:
        p.error(str(e))

    print(f"[setup] device={args.device}  videos={len(args.video)}  roi='{args.roi}'")
    print(f"[setup] prompt: {prompt[:80]}{'…' if len(prompt) > 80 else ''}")

    dino = load_dinov2(args.device, args.dino_model)
    clip = load_clip(args.device, args.clip_model)
    lpips_model = None
    if not args.no_lpips:
        if HAVE_LPIPS:
            lpips_model = load_lpips_model(args.device, args.lpips_net)
        else:
            print("[warn] lpips not installed — skipping LPIPS metrics. "
                  "Install with: pip install lpips")

    results = []
    for vpath in args.video:
        r = evaluate_video(vpath, prompt, args, dino, clip, lpips_model)
        results.append(r)

        out_dir = args.output_dir or (os.path.splitext(vpath)[0] + "_semantics_eval")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "semantics_eval.json")
        with open(out_path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"[save] {out_path}")

    if len(results) == 1:
        print_single(results[0])
    else:
        print_comparison(results)
        if args.output_dir:
            cmp_path = os.path.join(args.output_dir, "semantics_comparison.json")
            ref = results[0]["metrics"]
            diff_vs_ref = []
            for r in results[1:]:
                d = {}
                for k, v in ref.items():
                    v2 = r["metrics"].get(k)
                    if v is not None and v2 is not None:
                        d[k] = v - v2  # ref - other
                diff_vs_ref.append(d)
            with open(cmp_path, "w") as f:
                json.dump({
                    "videos":  [r["video"] for r in results],
                    "metrics": [r["metrics"] for r in results],
                    "diff_vs_ref": diff_vs_ref,
                    # keep legacy 2-video key when exactly two inputs
                    "diff": diff_vs_ref[0] if len(diff_vs_ref) == 1 else None,
                }, f, indent=2)
            print(f"[save] {cmp_path}")


if __name__ == "__main__":
    main()
