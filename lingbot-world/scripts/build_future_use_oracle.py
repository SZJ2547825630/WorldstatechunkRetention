#!/usr/bin/env python3
"""Convert attention_mass oracle .pt files into future_use_v1 dumps (P0).

Does not modify World-State CR runtime. Example:

  python scripts/build_future_use_oracle.py \\
      --oracle artifacts/.../oracle_dense_*.pt \\
      --out_dir output/oracles_future_use_v1
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wan.utils.future_use_labels import convert_oracle_payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", nargs="+", required=True, help="input oracle .pt (globs ok)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    paths = []
    for pat in args.oracle:
        matched = sorted(glob.glob(pat))
        paths.extend(matched if matched else [pat])
    os.makedirs(args.out_dir, exist_ok=True)

    for p in paths:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        out = convert_oracle_payload(
            blob, horizon=args.horizon, gamma=args.gamma, alpha=args.alpha)
        name = os.path.basename(p).replace(".pt", "") + "_future_use_v1.pt"
        dest = os.path.join(args.out_dir, name)
        torch.save(out, dest)
        print(f"[ok] {p} -> {dest}  records={len(out['records'])}")


if __name__ == "__main__":
    main()
