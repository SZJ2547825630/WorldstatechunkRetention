# Weights (local only)

Large model / eval weights are **never committed**. This directory only holds
symlink stubs or empty placeholders. After clone, download checkpoints and
point `CKPT_DIR` (or refresh the symlinks).

## Suggested layout

| Path under `weights/` | Typical target | Role |
|-----------------------|----------------|------|
| `checkpoints/lingbot_world_fast` | `lingbot-world-base-cam` (HF) | LingBot-World-Fast 14B |
| `eval/dinov2-base` | DINOv2 | Semantics eval (optional) |
| `eval/clip-vit-base-patch32` | CLIP | Text eval (optional) |

## Environment

```bash
source env.sh
export CKPT_DIR=/path/to/lingbot-world-base-cam   # recommended
echo "$CKPT_DIR"
```

For the WorldKV baseline, the checkpoint path should contain the substring
`cam` (control-type detection). Prefer a folder named like
`lingbot-world-base-cam`.

## Download

- LingBot-World-Fast: [HuggingFace robbyant / lingbot-world](https://huggingface.co/collections/robbyant/lingbot-world)
- DINOv2 / CLIP: standard HuggingFace model IDs or your local cache

Small **ChunkSelector** weights (≈10KB) live in
`lingbot-world/assets/selectors/` and **are** tracked in git — they are not
placed under `weights/`.
