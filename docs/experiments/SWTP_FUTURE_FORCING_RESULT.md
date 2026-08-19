# SWTP Future Forcing(真注意力 saliency)结果(2026-06-27,执行机)

**任务**:实现 handoff §7.4 Stage 2 Future Forcing —— 用真注意力(参与式注意力)替代帧残差当 SWTP saliency,在 VBench 测能否提质。
**实现**:`sequence_parallel.py::sp_attn_forward_causal`(SP 路径,ulysses 实际走这)+ `model_fast.py`(非-SP)算 per-token `attn_importance = softmax分子 exp(key·mean_query − max)`;`image2video_fast.py` 加 `swtp_saliency_mode`(residual/attn/attn+residual),逐层用注意力重要度当 SWTP saliency。MoCE 没碰。
**两个实现坑(冒烟逐字节抓到)**:① 第一版改错路径(非-SP),ulysses 下永不触发;② `key·mean_q` 有负 → Gini 门把 SWTP 跳过 → 改 exp 分子修复。

**设置**:12 clip(clips_ratio),recent=2;residual 基线复用 sweep_ratio/s1_r2_a2(同配置)。

## 效率(attn 反而多压一点)
| 方法 | ctx_tok | time |
|---|---|---|
| MoSaiC(residual) | 22576 | 332s |
| MoSaiC_attn | 21546 (−4.6%) | 328s |
| MoSaiC_attnres | 21958 | 330s |
注意力重要度 Gini 更高 → SWTP 在更多 chunk 触发 → 多压 ~4.6% token。

## 质量(本地 DINOv2 + VBench)—— 不提质
本地(n=12 配对,高=好):
| 指标 | residual | attn | attnres |
|---|---|---|---|
| anchor_LT(长程身份) | **0.5141** | 0.5113 | 0.4984 |
| anchor_final | **0.5070** | 0.4684 | 0.4156 |
| short_min(最差帧抗抖) | 0.4756 | 0.4828 | **0.4864** |

VBench(6/7 维;subject_consistency DINO-v1 被墙):**MoSaiC_attn 赢 0/6 维**,全部 ±0.008 噪声内、略负;temporal_flickering −0.0006(本地的抗抖苗头未被 VBench 证实)。

## 结论
**Future Forcing(真注意力 saliency)在本模型上不提质**(VBench 0/6、本地身份略降),仅 attn 多压 ~4.6% token 是小效率红利,但代价是身份略降,非干净 Pareto。

**贯穿性结论(SWTP 质量信号已试 2 个轴)**:损坏门(残差离群)+ Future Forcing(真注意力)**都不提质**。→ **在这个相机控制、(多为)静态场景的世界模型上,per-token 信号撬不动质量**;MoSaiC 的真价值是**高效推理**(§2.6 最优 1:1:2 配比 −18% token/−6% time + SWTP 压缩),不是提质。建议收为效率向贡献(recent=1 + SWTP,质量中性),不再在 per-token 质量信号上磨。
