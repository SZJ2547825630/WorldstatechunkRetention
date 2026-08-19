# Timestep-Invariant Condition Hoisting（TICH）

## 定位

**Invariant-Control Hoisting for Low-Step World Models**

不是近似 activation 缓存，而是把 **chunk 内不随 timestep 变化** 的条件子图从 \(S\) 次降为 1 次。  
与 WS-CR / KV retention **正交**；训练无关、数据无关；除浮点累加外应数值等价。

动机：把 chunk 内不随 timestep 变化的条件子图从 \(S\) 次降为 1 次，走计算图严格不变量，而不是近似 activation 缓存。

## 三类精确消除

| # | 子图 | 做法 |
|---|------|------|
| 1 | Global cam embedding | `patch_embedding_wancamctrl` + 2×Linear 每 chunk 算一次 |
| 2 | Block cam 4×Linear | 缓存 `cam_scale/cam_shift`；逐元素 inject 仍每步执行 |
| 3 | I2V `Conv3d(cat(x,c))` | `Conv_x(x)+Conv_c(c)+b`，`Conv_c(c)+b` 每 chunk 一次 |

**禁止**用「整段 cam 17% × 75%」报收益；只报 hoistable kernel 实测。

## 运行

```bash
source env.sh
export CKPT_DIR=/path/to/lingbot-world-base-cam
cd lingbot-world

# vs window 基线同 MEMORY_POLICY
EXAMPLE=01 FRAME=361 NPROC=4 ULYSSES=4 \
MEMORY_POLICY=window COND_HOIST_PROFILE=1 \
bash scripts/run_cond_hoist.sh 2>&1 | tee output/example01/cond_hoist.log
```

看：

- `p50_chunk_time_s` / `total_generation_time_s` vs window
- `[COND-HOIST] profile[global_cam|block_linears|elemwise|conv_static|conv_dynamic]`
- `global reuse/compute`、`block reuse/compute`

## 正确性

理想：`max_abs(noise_pred_hoist - baseline)` / `x0` 接近机器精度；视频逐帧误差可忽略。  
可用 `--cond_hoist_verify`（调试）或与 window 出片目视对照。

## 显存

启动日志会打印 `block_scale_shift_cache≈X MB`。若过大可 `--cond_hoist_block_cam false`，先只用 global + conv_split。
