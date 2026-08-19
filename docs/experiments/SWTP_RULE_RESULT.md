# SWTP 损坏门 A/B 结果(2026-06-27,执行机)

**任务**:实现 handoff §7 的 SWTP 选择规则可切换(损坏门),在 VBench 测。
**设置**:clips_sweep 24 个 481 帧长视频,recent=2,8 卡。方法:MoCE(无SWTP)/ MoSaiC(topk,现状)/ MoSaiC_outlier(TokenTrim 损坏门)。MoSaiC_lowdrift OOM(第4方法显存碎片,已修 run_dp2.sh 加 expandable_segments,待重跑)。

**SWTP 确触发**:avg ctx tokens MoCE 25448 → MoSaiC 22565 → outlier 22691;outlier/topk 视频实质不同(frame120 像素差 40/255)。非 no-op。

## VBench(6/7 维;subject_consistency 因 DINO-v1 权重被墙跑不了)
| dim | MoCE | MoSaiC(topk) | outlier |
|---|---|---|---|
| background_consistency | 0.8951 | 0.8971 | 0.8964 |
| temporal_flickering | 0.9395 | 0.9392 | 0.9387 |
| motion_smoothness | 0.9638 | 0.9638 | 0.9635 |
| dynamic_degree | 1.0 | 1.0 | 1.0 |
| imaging_quality | 0.6550 | 0.6567 | 0.6532 |
| aesthetic_quality | 0.5217 | 0.5233 | 0.5235 |

全 Δ 在 ±0.004 内(VBench 噪声级)。outlier 1/6 维微胜。

## 本地 DINOv2 身份一致性(补 subject_consistency,n=24 配对)
| 指标(高=好) | MoCE | MoSaiC(topk) | outlier |
|---|---|---|---|
| anchor_last_third(长程身份) | 0.5252 | **0.5428** | 0.5239 |
| anchor_final | 0.4822 | 0.5067 | 0.5067 |
| short_term_min(最差抖动) | 0.4623 | 0.4894 | **0.5087** |

## 结论
1. **损坏门假设未被支持**:outlier 既没在 VBench 提质,也略伤本地长程身份(−0.019),仅最差帧抗抖小幅+0.020。**没翻正 SWTP**。
2. **现版 SWTP(topk)不有害**:VBench 中性偏正,本地身份优于 MoCE。→ "SWTP 留住损坏 token、中性/略伤"的前提在本模型上**不成立**。
3. **Caveat**:24 clip、Δ 都在噪声级,无统计显著;subject_consistency(最相关维)缺;lowdrift/mid 未跑。

## 建议下一步
- 损坏门方向(drop 高残差)不灵 → 不建议在 selection-rule 上继续磨。机器 A 自己的 §7.4 Stage 2(**future/output-aware 打分**,Future Forcing 报 +1.49× 主体一致性)是更高杠杆的方向,且本地结果显示"信息量/未来重要度"比"残差高低"更可能是对的轴。
