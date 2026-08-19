# RealCam-Vid Benchmark

RealCam-Vid 是面向 **相机可控视频生成 / 世界模型** 的外部相机轨迹评测集。本仓库用它做 Chunk Retention（CR）相对 Sliding Window（`no_cr` / `window`）的效率与质量对照。

## 数据从哪来

官方 test 元数据在归档目录：

- `artifacts/archive_pre_paper/bench/realcamvid/RealCam-Vid_test.csv`（5000 条）
- 同目录 `RealCam-Vid_test.npz`（轨迹/相机参数）

每条样本大致包含：

| 字段 | 含义 |
|------|------|
| `dataset_source` | 来源子集，常见 **RealEstate10K**（室内漫游）、**MiraData9K** 等 |
| `video_path` | 原始视频路径 |
| `short_caption` / `long_caption` | 文本提示 |
| `align_factor` / `camera_scale` | 相机尺度对齐相关 |
| `vtss_score` | 轨迹/视频质量筛选分 |

本仓流水线会把选中 clip **转换成 LingBot 推理格式**：

```
clip_id/
  image.jpg      # 条件首帧
  poses.npy      # c2w / 相机轨迹（从 w2c 等格式转换并去归一化）
  prompt.txt     # 通常取 caption
```

## 本仓常用子集

归档在 `artifacts/archive_pre_paper/bench/realcamvid/`：

| 子集 | 规模（约） | 用途 |
|------|------------|------|
| `clips/` | ~210 | 主批量（早期 10 / 210 clip 报告） |
| `clips_long/` | ~81 | 更长 horizon |
| `clips_loop/` | ~24 | **回访 / 闭环**（World-State CR 主战场） |

## 官方默认测试子集（已标记）

路径：`lingbot-world/bench/realcamvid/subsets/`

| 子集 | 条数 | 用途 |
|------|------|------|
| **`default_loop`** | 24 | 全部 loop  clip（回访主评测） |
| **`default_random`** | 40 | seed=42 随机前向漫游（泛化 sanity） |
| **`default_all`** | 64 | loop + random（**推荐默认电池**） |

清单文件：`default_loop.txt` / `default_random.txt` / `default_all.txt`；元数据见 `manifest.json`。

生成 symlink 目录（一次性）：

```bash
bash lingbot-world/bench/realcamvid/build_subsets.sh
# -> clips_default_loop/  clips_default_random/  clips_default_all/
```

批量跑默认电池：

```bash
# 方式 1
bash lingbot-world/bench/realcamvid/run_default.sh

# 方式 2
torchrun ... bench/batch_generate.py --subset default_all --out_dir output/realcamvid_default_all ...
```

环境变量：`REALCAMVID_SUBSET=default_loop|default_random|default_all`。
| `clips_subject/` | 少量 | 有显著主体的场景 |

典型长度设定（历史报告）：

- 短评测：约 **273 frames / ~23 chunks**（~17s）
- 长评测：`frame_num=481`（~30s / ~40 chunks），CR 优势更明显

## 评测协议（默认 = WorldKV）

**本仓默认质量评测协议**对齐 [WorldKV](https://arxiv.org/abs/2605.22718)：在 GT 相机轨迹上找回访帧，与同视角首次访问帧比较。

| 指标 | 含义 |
|------|------|
| PSNR↑ / SSIM↑ / LPIPS↓ | 逐对回访一致性 |
| FID↓ | 回访帧集合 vs 首访帧集合 |
| Throughput (FPS)↑ | 墙钟吞吐；另报 last-chunk FPS |
| ctx / peak GB | 注意力上下文与峰值显存 |

```bash
# 默认电池：default_loop + WorldKV 协议
REALCAMVID_OUT=output/realcamvid_ws_vs_window_default_all_v2 \
REALCAMVID_SUBSET=default_loop \
METHODS=window,world_state_cr,worldkv \
  bash bench/realcamvid/run_worldkv_eval.sh
# -> $OUT/worldkv_memory_eval_default_loop.json
```

脚本：`lingbot-world/bench/eval_worldkv_memory.py`。

**公平预算（生成侧）：**

- 同 seed / 同分辨率 / 同 SP；CR：`max_attention_size=47000`，`sink_size=1`
- Baseline **`window`**：`local_attn_size=30`
- CR：`heuristic_cr` / `learned_cr` / `world_state_cr`
- **官方 WorldKV**：`third_party/WorldKV`，见 `bench/realcamvid/run_worldkv_official.sh`

**辅助质量指标（`eval_semantics.py`，附录）：** LPIPS vs seed / DINO / CLIP drift。

历史结论摘要（Heuristic CR vs Window，10-clip）：效率稳定（约 −6% 时间、−19% peak mem）；质量混合——终点 identity 略好、CLIP 对齐偶有回退。World-State CR 更适合在 **`clips_loop` 回访子集** 上与 Window / Heuristic / Learned 对照。

## 怎么跑

```bash
source env.sh
cd "$LINGBOT_WORLD"

# 批量（方法名已是 CR）
torchrun --nproc_per_node=8 bench/batch_generate.py \
  --ckpt_dir "$CKPT_DIR" \
  --clips_dir /path/to/realcamvid/clips_loop \
  --out_dir /path/to/out \
  --methods window,heuristic_cr,learned_cr,world_state_cr \
  --frame_num 481 --ulysses_size 8
```

原始大体积视频/stats 不在官方源码树内，见 `artifacts/archive_pre_paper/bench/realcamvid/`。
