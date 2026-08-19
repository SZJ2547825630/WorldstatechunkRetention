#!/usr/bin/env python3
"""Summarize consolidation sweep stats (+ optional worldkv eval JSON).

Usage:
  python bench/summarize_consol_sweep.py output/realcamvid_consol_sweep_default_loop
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from collections import defaultdict
from glob import glob


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "output/realcamvid_consol_sweep_default_loop"
    stats_dir = os.path.join(root, "stats")
    by = defaultdict(list)
    for p in glob(os.path.join(stats_dir, "*.json")):
        name = os.path.basename(p)[:-5]  # strip .json
        # clip_method: method may contain underscores; match known suffixes
        method = None
        for m in sorted(
            [
                "world_state_cr_v2",
                "ws_v3_a05_g128",
                "ws_v3_a05_g64",
                "ws_v3_a05",
                "ws_v3_a0",
                "ws_v3_a1",
                "world_state_cr",
            ],
            key=len,
            reverse=True,
        ):
            if name.endswith("_" + m):
                method = m
                break
        if method is None:
            continue
        s = json.load(open(p))["stats"]
        by[method].append(s)

    rows = []
    for m, ss in sorted(by.items()):
        def mean(key, default=None):
            vals = [x.get(key) for x in ss if x.get(key) is not None]
            return st.mean(vals) if vals else default

        l2 = []
        for x in ss:
            tc = x.get("tier_counts") or {}
            l2.append(float(tc.get("L2", 0)))
        rows.append({
            "method": m,
            "n": len(ss),
            "ctx": mean("avg_attention_context_tokens"),
            "peak_gb": mean("peak_memory_allocated_gb"),
            "time_s": mean("total_generation_time_s"),
            "chunk_s": mean("avg_chunk_time_s"),
            "revisit": mean("revisit_coverage"),
            "L2": st.mean(l2) if l2 else 0.0,
            "consol": ss[0].get("consolidation") if ss else None,
        })

    print(f"root={root}")
    print(f"{'method':22s} {'n':>3} {'ctx':>8} {'peakGB':>7} {'time_s':>8} {'chunk_s':>7} {'revisit':>7} {'L2':>5}")
    for r in rows:
        print(
            f"{r['method']:22s} {r['n']:3d} "
            f"{(r['ctx'] or 0):8.1f} {(r['peak_gb'] or 0):7.2f} "
            f"{(r['time_s'] or 0):8.1f} {(r['chunk_s'] or 0):7.3f} "
            f"{(r['revisit'] or 0):7.3f} {(r['L2'] or 0):5.2f}"
        )

    eval_path = os.path.join(root, "worldkv_memory_eval_default_loop.json")
    if os.path.isfile(eval_path):
        ev = json.load(open(eval_path))
        print("\nquality (worldkv revisit):")
        print(f"{'method':22s} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'FID':>7} {'FPS':>6}")
        for m, v in sorted((ev.get("by_method") or {}).items()):
            print(
                f"{m:22s} {v.get('psnr', 0):7.3f} {v.get('ssim', 0):7.3f} "
                f"{v.get('lpips', 0):7.3f} {v.get('fid', 0):7.2f} "
                f"{v.get('throughput_fps', 0):6.3f}"
            )


if __name__ == "__main__":
    main()
