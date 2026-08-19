#!/usr/bin/env python3
"""Parse the [FastInferenceSummary] block from a generate_fast.py stdout log.

generate_fast.py logs timing/memory stats only to stdout (via logging.info) under
a "[FastInferenceSummary]" header followed by "  <key>: <value>" lines. This script
extracts that block into a structured JSON for benchmark aggregation.

Usage:
    python parse_summary.py --log <log.txt> --config baseline --example 00 --out <stats.json>
"""
import argparse
import json
import math
import re
import sys

# Keys emitted by _log_generation_summary (image2video_fast.py:725-741)
FLOAT_KEYS = {
    "retained_chunk_ratio", "avg_motion_score_kept", "avg_motion_score_evicted",
    "avg_attention_context_tokens", "total_generation_time_s", "avg_chunk_time_s",
    "tail_chunk_time_s", "peak_memory_allocated_gb", "peak_memory_reserved_gb",
}
INT_KEYS = {
    "total_chunks", "retained_chunks", "evicted_chunks", "rescue_count",
    "max_attention_context_tokens",
      
     
}
BOOL_KEYS = {
    "motion_adaptive_enabled", 
     
}
ALL_KEYS = FLOAT_KEYS | INT_KEYS | BOOL_KEYS

# Matches e.g. "[2026-06-17 07:23:30,123] INFO:   total_generation_time_s: 257.1700"
LINE_RE = re.compile(r"INFO:\s+([a-z0-9_]+):\s+(.+?)\s*$")


def coerce(key, raw):
    if raw in ("n/a", "None"):
        return None
    if raw == "inf":
        return math.inf
    if key in BOOL_KEYS:
        return raw.strip().lower() in ("true", "1", "yes")
    if key in INT_KEYS:
        try:
            return int(float(raw))
        except ValueError:
            return None
    if key in FLOAT_KEYS:
        try:
            return float(raw)
        except ValueError:
            return None
    return raw


def parse_log(path):
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    stats = {}
    in_block = False
    for ln in lines:
        if "[FastInferenceSummary]" in ln:
            in_block = True
            continue
        if not in_block:
            continue
        m = LINE_RE.search(ln)
        if m and m.group(1) in ALL_KEYS:
            stats[m.group(1)] = coerce(m.group(1), m.group(2))
            continue
        if not in_block or not stats:
            continue
        # End of summary block: blank line, non-INFO, or INFO key outside our schema
        # after we've already captured the last known TierQuant fields.
        if "Saving generated video" in ln or "Finished." in ln:
            break
        if m and m.group(1) not in ALL_KEYS and "peak_memory_reserved_gb" in stats:
            # Unknown key after mem stats — keep going (forward-compat) unless clearly done
            continue
        if "INFO:" not in ln and ln.strip() == "":
            if "quantized_segments" in stats or "peak_memory_reserved_gb" in stats:
                # allow a short grace; don't break on blank alone
                continue
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--example", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stats = parse_log(args.log)
    if "total_generation_time_s" not in stats:
        print(f"[parse_summary] WARNING: no [FastInferenceSummary] found in {args.log}",
              file=sys.stderr)
    record = {"config": args.config, "example": args.example, "stats": stats}
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
