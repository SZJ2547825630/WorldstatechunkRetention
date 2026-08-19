# MoCE 分层配比扫描结果(2026-06-27,执行机)

**任务**:扫 sink:recent:archive(min_keep) 找最优 MoCE 分层配比。
**设置**:12 个均衡长视频(clips_ratio,4/源),MoSaiC only,recent=2 基线,8 卡。
**排名**:本地 DINOv2 长程身份 anchor_LT(VBench subject_consistency 的 DINO-v1 被墙)+ 抗抖 short_min + 效率(ctx/time/retained)。脚本 `realcamvid/rank_ratio_local.py`,数据 `sweep_ratio/ratio_ranking.json`。

| 排名 | sink:rec:arc | anchor_LT | short_min | ctx_tok | time_s | ret_chk |
|---|---|---|---|---|---|---|
| 1 | **1:1:2** | **0.5291** | 0.5090 | **18586** | **311** | 4.0 |
| 2 | 1:2:1 | 0.5181 | 0.4964 | 20147 | 319 | 4.0 |
| 3 | 1:2:2(基线) | 0.5141 | 0.4756 | 22576 | 332 | 5.0 |
| 4 | 2:2:2 | 0.5141(no-op,逐字节=基线) | 0.4756 | 22576 | 331 | 5.0 |
| 5 | 0:2:2 | 0.4993 | 0.5262 | 19753 | 318 | 4.0 |
| 6 | 1:3:2 | 0.4873 | 0.4661 | 26642 | 352 | 6.0 |
| 7 | 1:2:3 | 0.4646 | 0.4674 | 24883 | 343 | 6.0 |

## 结论:最优 = `1:1:2`(sink=1, recent=1, archive_min=2),Pareto 赢
- 长程身份最高(+0.015 vs 基线)、token −18%、time −6%、retained 4 vs 5。
- **recent=1 > recent=2**(质量+效率双赢)→ 把 handoff recent=4→2 再推一步。
- 加 recent/archive 均负(1:3:2 −0.027,1:2:3 −0.050)。
- sink=1 足够(sink=2 无效;sink=0 伤身份 −0.015)。

## Caveat / 下一步
- 12 clip、本地 DINOv2 指标、Δ 中等;recent=1 超出 handoff 已验证边界(只 1 个 dense 近邻,强局部运动可能伤连贯)。
- 已切换代码默认到 `1:1:2`（`ma_kv_recent_window=1`）。`1:2:2` 仍可用环境变量/CLI 复现。
