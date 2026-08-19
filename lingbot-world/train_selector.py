"""
Train the World-State CR ChunkSelector (11-D schema).

Input : oracle .pt files from
        `generate_fast.py --collect_oracle --oracle_out <path>`.
Output: a ChunkSelector checkpoint via ``save_selector``.

Default labels are attention-mass rankings (``selector_ws_v1.pt``, ablation).
With ``--label_type future_use_v1``, attention-mass oracles are converted
offline into Future Coverage Oracle labels and written as
``selector_ws_future_v1.pt`` — the default ``world_state_cr`` runtime ckpt.

Offline NDCG vs Heuristic CR (motion score) is reported on a held-out split.
Learned CR (5-D ``selector_all4.pt``) is a frozen baseline and is not
retrained by this script.
"""
import argparse
import math

import torch

from wan.modules.chunk_selector import (
    ChunkSelector, save_selector, build_chunk_features, build_chunk_features_v2,
    FEATURE_NAMES, FEATURE_NAMES_V2, FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_V2, features_for_schema,
)
from wan.utils.future_use_labels import (
    LABEL_TYPE_ATTENTION_MASS,
    LABEL_TYPE_FUTURE_USE,
    convert_oracle_payload,
)


def _median(xs):
    xs = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not xs:
        return 1.0
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _translation_scale_from_meta(meta, cfg_scale=None):
    if cfg_scale is not None and float(cfg_scale) > 0:
        return float(cfg_scale)
    translations = []
    for m in meta.values():
        pose = m.get("pose")
        if pose and pose.get("translation") is not None:
            translations.append(torch.as_tensor(pose["translation"], dtype=torch.float32).flatten())
    if len(translations) < 2:
        return 1.0
    # Approximate step scale from successive chunk ids when available.
    by_id = sorted(
        ((int(m["chunk_id"]), torch.as_tensor(m["pose"]["translation"], dtype=torch.float32).flatten())
         for m in meta.values()
         if m.get("pose") and m.get("chunk_id") is not None and m["pose"].get("translation") is not None),
        key=lambda x: x[0],
    )
    steps = []
    for i in range(1, len(by_id)):
        steps.append(float(torch.norm(by_id[i][1] - by_id[i - 1][1]).item()))
    steps = [s for s in steps if s > 1e-8]
    return _median(steps) if steps else 1.0


def _normalize_meta_keys(meta):
    """Ensure chunk_meta is keyed by int chunk_id when possible."""
    out = {}
    for k, v in meta.items():
        try:
            ik = int(k)
        except (TypeError, ValueError):
            try:
                ik = int(v.get("chunk_id"))
            except Exception:
                ik = k
        if isinstance(v, dict) and v.get("chunk_id") is None:
            v = {**v, "chunk_id": ik}
        out[ik] = v
    return out


def load_groups(paths, label_type=LABEL_TYPE_ATTENTION_MASS,
                future_horizon=8, future_gamma=0.9, future_alpha=0.5):
    """Return (groups, motion_ref, vnorm_ref, translation_scale)."""
    groups = []
    motions, vnorms = [], []
    scales = []
    for p in paths:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        cfg = dict(blob.get("config", {}) or {})
        # Convert attention_mass dumps -> future_use on the fly when requested.
        want_future = (
            label_type == LABEL_TYPE_FUTURE_USE
            or cfg.get("label_type") == LABEL_TYPE_FUTURE_USE
        )
        if want_future and cfg.get("label_type") != LABEL_TYPE_FUTURE_USE:
            blob = convert_oracle_payload(
                blob,
                horizon=future_horizon,
                gamma=future_gamma,
                alpha=future_alpha,
            )
            cfg = dict(blob.get("config", {}) or {})
        meta = _normalize_meta_keys(blob["chunk_meta"])
        sink_n = int(cfg.get("sink_chunk_count", 0))
        recent = int(cfg.get("recent_window", 1))
        scale = _translation_scale_from_meta(meta, cfg.get("translation_scale"))
        scales.append(scale)
        by_gen = {}
        for r in blob["records"]:
            by_gen.setdefault(int(r["gen_chunk_id"]), {})[int(r["seg_id"])] = float(r["mass"])
        for m in meta.values():
            if math.isfinite(m.get("motion_score", float("inf"))):
                motions.append(m["motion_score"])
            vnorms.append(m.get("value_norm", 0.0))
        for gen, seg_mass in by_gen.items():
            cur = meta.get(gen)
            if cur is None:
                continue
            cands, mass = [], []
            for sid, ms in seg_mass.items():
                if sid < sink_n or sid >= gen or (gen - sid) <= recent:
                    continue
                cm = meta.get(sid)
                if cm is None:
                    continue
                cands.append(cm)
                mass.append(ms)
            if len(cands) >= 2:
                groups.append({
                    "gen": gen, "cands": cands, "cur": cur, "mass": mass,
                    "translation_scale": scale,
                    "label_type": cfg.get("label_type", LABEL_TYPE_ATTENTION_MASS),
                })
    return groups, _median(motions), _median(vnorms), _median(scales)


def featurize(group, motion_ref, vnorm_ref, translation_scale=None,
              schema_version=FEATURE_SCHEMA_VERSION):
    scale = translation_scale if translation_scale is not None else group.get("translation_scale", 1.0)
    names = FEATURE_NAMES_V2 if schema_version == FEATURE_SCHEMA_VERSION_V2 else FEATURE_NAMES
    rows = [
        features_for_schema(
            names, c, group["cur"], group["gen"], motion_ref, vnorm_ref,
            translation_scale=scale, schema_version=schema_version,
        )
        for c in group["cands"]
    ]
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(group["mass"], dtype=torch.float32)


def listwise_loss(util, mass):
    """KL(softmax(mass) || softmax(util)) up to const = -sum p*log q."""
    p = torch.softmax(mass, dim=0)
    logq = torch.log_softmax(util, dim=0)
    return -(p * logq).sum()


def pairwise_loss(util, mass, margin_scale=1.0):
    """RankNet-style pairwise logistic loss. For every candidate pair (i,j) with
    mass[i] != mass[j], push util toward the same order. Strong per-pair gradients
    (unlike listwise-softmax over near-flat mass, which collapses to a constant)."""
    n = util.numel()
    if n < 2:
        return util.sum() * 0.0
    ui = util.unsqueeze(1) - util.unsqueeze(0)          # [n,n] util_i - util_j
    mi = mass.unsqueeze(1) - mass.unsqueeze(0)          # [n,n] mass_i - mass_j
    mask = (mi.abs() > 1e-9)
    sign = torch.sign(mi)
    # want sign*(ui) large positive -> loss = softplus(-sign*ui)
    l = torch.nn.functional.softplus(-sign * ui * margin_scale)
    return (l * mask).sum() / mask.sum().clamp_min(1)


def ndcg(order_scores, mass, k=2):
    """NDCG@k of a ranking induced by order_scores against relevance=mass."""
    idx = torch.argsort(torch.as_tensor(order_scores), descending=True)
    rel = torch.as_tensor(mass, dtype=torch.float64)[idx]
    k = min(k, rel.numel())
    disc = 1.0 / torch.log2(torch.arange(2, 2 + k, dtype=torch.float64))
    dcg = (rel[:k] * disc).sum()
    ideal = torch.sort(torch.as_tensor(mass, dtype=torch.float64), descending=True).values[:k]
    idcg = (ideal * disc).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def eval_rankers(groups, model, motion_ref, vnorm_ref, k=2, device="cpu",
                 schema_version=FEATURE_SCHEMA_VERSION):
    """Mean NDCG@k of learned utility vs. motion-only vs. random baselines."""
    learned, motion, rand = [], [], []
    model.eval()
    with torch.no_grad():
        for g in groups:
            feats, mass = featurize(
                g, motion_ref, vnorm_ref, schema_version=schema_version)
            feats, mass = feats.to(device), mass.to(device)
            util = model(feats)
            learned.append(ndcg(util.cpu(), mass.cpu(), k))
            motion.append(ndcg(feats[:, 0].cpu(), mass.cpu(), k))  # col 0 == motion_norm
            rand.append(ndcg(torch.randn(len(mass)), mass.cpu(), k))
    n = max(1, len(groups))
    return sum(learned) / n, sum(motion) / n, sum(rand) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", nargs="+", required=True, help="oracle .pt file(s)")
    ap.add_argument("--out", required=True, help="output selector .pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--ndcg_k", type=int, default=2)
    ap.add_argument("--loss", choices=["pairwise", "listwise"], default="pairwise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--label_type",
        choices=[LABEL_TYPE_ATTENTION_MASS, LABEL_TYPE_FUTURE_USE],
        default=LABEL_TYPE_ATTENTION_MASS,
        help="Training target. future_use_v1 converts attention_mass oracles "
             "into Future Coverage Oracle labels (P0).")
    ap.add_argument(
        "--feature_schema",
        choices=["v1", "v2", FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION_V2],
        default="v1",
        help="Feature formula schema. v1=frozen World-State CR ablation; "
             "v2=P1 corrected formulas (default world_state_cr).")
    ap.add_argument("--future_horizon", type=int, default=8)
    ap.add_argument("--future_gamma", type=float, default=0.9)
    ap.add_argument("--future_alpha", type=float, default=0.5)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"),
                    help="training device; defaults to cuda (all training runs on GPU).")
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.feature_schema in ("v2", FEATURE_SCHEMA_VERSION_V2):
        schema_version = FEATURE_SCHEMA_VERSION_V2
        feature_names = FEATURE_NAMES_V2
    else:
        schema_version = FEATURE_SCHEMA_VERSION
        feature_names = FEATURE_NAMES

    torch.manual_seed(args.seed)
    groups, motion_ref, vnorm_ref, translation_scale = load_groups(
        args.oracle,
        label_type=args.label_type,
        future_horizon=args.future_horizon,
        future_gamma=args.future_gamma,
        future_alpha=args.future_alpha,
    )
    if len(groups) < 4:
        raise SystemExit(f"too few ranking groups ({len(groups)}); collect more clips/chunks.")
    print(
        f"[data] {len(groups)} ranking groups | label_type={args.label_type} "
        f"| motion_ref={motion_ref:.4g} "
        f"vnorm_ref={vnorm_ref:.4g} translation_scale={translation_scale:.4g} "
        f"| schema={schema_version} features={feature_names}"
    )

    # deterministic split
    n_val = max(1, int(len(groups) * args.val_frac))
    perm = torch.randperm(len(groups), generator=torch.Generator().manual_seed(args.seed))
    val_idx = set(perm[:n_val].tolist())
    train_g = [g for i, g in enumerate(groups) if i not in val_idx]
    val_g = [g for i, g in enumerate(groups) if i in val_idx]

    print(f"[device] training on {device}")
    model = ChunkSelector(hidden=args.hidden).to(device)
    # Set input standardization from the training-set feature stats.
    allfeat = torch.cat([
        featurize(g, motion_ref, vnorm_ref, schema_version=schema_version)[0]
        for g in train_g
    ], dim=0)
    model.set_norm(allfeat.mean(0), allfeat.std(0).detach())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    loss_fn = pairwise_loss if args.loss == "pairwise" else listwise_loss

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for g in train_g:
            feats, mass = featurize(
                g, motion_ref, vnorm_ref, schema_version=schema_version)
            feats, mass = feats.to(device), mass.to(device)
            loss = loss_fn(model(feats), mass)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach())
        if (ep + 1) % 50 == 0 or ep == 0:
            l, m, r = eval_rankers(
                val_g, model, motion_ref, vnorm_ref, args.ndcg_k, device,
                schema_version=schema_version)
            print(f"[ep {ep+1:4d}] train_loss={total/max(1,len(train_g)):.4f} "
                  f"| val NDCG@{args.ndcg_k}: learned={l:.4f} heuristic={m:.4f} random={r:.4f}")

    l, m, r = eval_rankers(
        val_g, model, motion_ref, vnorm_ref, args.ndcg_k, device,
        schema_version=schema_version)
    lt, mt, _ = eval_rankers(
        train_g, model, motion_ref, vnorm_ref, args.ndcg_k, device,
        schema_version=schema_version)
    tag = f"{args.label_type}/{schema_version}"
    print(f"\n================ {tag} offline ranking ================")
    print(f"val   NDCG@{args.ndcg_k}: learned={l:.4f}  heuristic={m:.4f}  random={r:.4f}")
    print(f"train NDCG@{args.ndcg_k}: learned={lt:.4f} heuristic={mt:.4f}")
    verdict = ("learned > Heuristic" if l > m + 1e-3
               else ("≈ tie" if abs(l - m) <= 1e-3 else "learned < Heuristic"))
    print(f"verdict (held-out): {verdict}  (Δ={l-m:+.4f})")
    print("===============================================================\n")

    save_selector(
        model, args.out,
        meta={
            "motion_ref": motion_ref,
            "vnorm_ref": vnorm_ref,
            "translation_scale": translation_scale,
            "schema_version": schema_version,
            "label_type": args.label_type,
            "future_horizon": args.future_horizon if args.label_type == LABEL_TYPE_FUTURE_USE else None,
            "future_gamma": args.future_gamma if args.label_type == LABEL_TYPE_FUTURE_USE else None,
            "future_alpha": args.future_alpha if args.label_type == LABEL_TYPE_FUTURE_USE else None,
            "val_ndcg_world_state": l,
            "val_ndcg_heuristic": m,
            "ndcg_k": args.ndcg_k,
            "n_groups": len(groups),
            "feature_names": list(feature_names),
        },
        feature_names=feature_names,
        schema_version=schema_version,
    )
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
