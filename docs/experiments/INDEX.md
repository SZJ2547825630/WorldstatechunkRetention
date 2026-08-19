# Experiment index (paper-facing)

Results retained for Chunk Retention / SWTP / Consolidation.

| Doc | Content |
|-----|---------|
| [`REALCAMVID.md`](REALCAMVID.md) | RealCam-Vid benchmark protocol |
| [`SWTP_RULE_RESULT.md`](SWTP_RULE_RESULT.md) | SWTP keep-ratio / gini rules |
| [`SWTP_FUTURE_FORCING_RESULT.md`](SWTP_FUTURE_FORCING_RESULT.md) | SWTP vs future-forcing notes |
| [`RATIO_SWEEP_RESULT.md`](RATIO_SWEEP_RESULT.md) | Archive keep-ratio sweeps |

Large raw benches and obsolete prototypes live under
`artifacts/archive_pre_paper/` (not part of the public API).

Selector checkpoints for reproduction:

- `lingbot-world/assets/selectors/selector_ws_future_v1.pt` — World-State CR (default)
- `lingbot-world/assets/selectors/selector_ws_v1.pt` — World-State CR v1 ablation
- `lingbot-world/assets/selectors/selector_all4.pt` — Learned CR
