# C³ache 误差检测教程

这个流程用于回答：**相同动作状态、相同 σ、相同当前条件下，旧 chunk 的残差会让预测偏离多少？**
机器人始终执行完整模型给出的动作。检测结果不会写入机器人动作或历史 KV，也不会训练模型。
诊断模式的耗时包含额外计算和记录，不能用来报告加速比。

## 先区分两个参数

`chunk` 是一次预测的一组动作；`step` 是生成这一组动作时的一次去噪更新。
LingBot-VA 的本项目 LIBERO 配置每个普通 chunk 至多执行 16 个环境动作，生成时做 50 次动作去噪更新。

| 参数 | 管理的尺度 | 例子 |
|---|---|---|
| `--cache-refresh-interval` | 跨 chunk 的刷新周期 | `2`：全算并存参考 → 复用 → 全算刷新 → 复用 |
| `--cache-start-step` / `--cache-end-step` | 一个 chunk 内可复用的去噪步索引，含两端 | `5..39`：索引 5～39，共 35 步 |

索引从 0 开始，因此 `5..39` 是第 6～40 次去噪更新。它不是 Transformer 层号，
也不是机器人环境动作步数。所有 50 次采样更新仍执行；缓存只代替某些步骤里的 Transformer 整栈计算。

`interval=1` 每个 chunk 全算；`2` 是 1 个全算、1 个复用；`4` 是 1 个全算、3 个复用。
`0` 的特殊含义是第一个普通 chunk 建立参考后不再周期刷新，并非关闭缓存。
LingBot-VA 的首个特殊 cold chunk 全算且不作为参考；它不计入普通 chunk 的刷新周期。
episode reset 会清空参考，不能跨 episode 复用。

[C³ache 原论文 §4.1](https://arxiv.org/html/2606.08962v1#S4.SS1) 在 Fast-WAM 的 **10 步**采样上，
主要比较 `0..6`（7 步）和 `0..7`（8 步），以及 interval `0、4、8`。
`0..7` 是复用 chunk 的前 8 步可用旧残差，最后 2 步全算；完整刷新 chunk 仍全部计算。

本项目原来的 `5..39 / interval=2` 是 **50 步 LingBot-VA 上未经质量验证的初始实验选择**，
不是论文的原始设置，也不是等价换算。LingBot-VA 还有每段 step 0 必须全算的 KV 保护。
默认固定窗口尚未通过质量验收；不要把它当作推荐部署参数。

实际动作 scheduler 使用 shift `0.05`，噪声 σ 非线性变化：

| 步索引 | σ 起点 → 这些更新后的 σ |
|---|---|
| `0..4` | 约 `1 → 0.3103` |
| `5..39` | 约 `0.3103 → 0.01235` |
| `40..49` | 约 `0.01235 → 0` |

因此不能只按步骤比例迁移论文窗口，也不能把“最后还有 10 步全算”视为充分纠偏的保证。

## 检测怎样保持完整策略

每一步只执行一次完整 Transformer，取得 `h0`、`hL` 和完整 velocity。
在假设可以复用的位置，用 `h0_current + R_reference` 额外运行一次当前输出头，获得旁路预测。
真正的采样器始终使用完整 velocity；真实历史与预测 KV 都只经过原生完整前向。
参考仍按指定 interval 刷新，不会因诊断每一步都全算而偷偷变成“每一步都刷新”。

另一个旁路用本次完整计算的 `h0 + (hL - h0)` 重建，检查相同状态下的 BF16 减法/加法误差。
这有助于区分纯数值重建误差与跨 chunk 复用带来的额外误差。

诊断记录包含：

- 原生 step、timestep、σ、Δσ、参考/当前 frame ID。
- residual、hidden、velocity 的相对 L2、RMSE、最大绝对误差，以及适用时的余弦相似度。
- `update_rmse`：FP32 计算的 `Δσ × (velocity_shadow - velocity_full)` 的 RMSE。
- `scaled_update_channel_rmse`：上述误差按原生动作反归一化尺度换算，逐通道报告。
- `same_state_roundtrip_velocity`：相同当前状态下 BF16 重建误差。
- 每 chunk 的实际全算次数、假设复用次数，以及完整动作数组和动作哈希。

velocity 先按原生 CFG 分支顺序融合，再仅取实际使用的动作通道；未使用的 23 个填充通道不混入动作误差。
LIBERO 的有效通道 0～2、3～5、6 分别对应平移、旋转、夹爪，应分别查看。
换算后的单位是原生动作命令单位，**不是直接的米、真实运动距离或最终抓取误差**。
`update_rmse` 是局部 Euler 更新差的浮点诊断量，不包含原生 BF16 采样乘加舍入和后续步骤的误差传播。

## 环境和更新

沿用已经跑通 baseline 的服务端和 LIBERO 客户端环境，以及同一个本地 checkpoint。
安装说明见 [README](../README.md) 和 [评测指南](EVALUATION.md)。新增诊断不需要额外 Python 包。
命令从仓库根目录执行。更新前先查看并保存自己的部署改动：

```bash
git status --short
git pull --ff-only
```

这版已包含 WebSocket `proxy=None`，本机连接不会经过代理。若之前本地修改过同一行，
先核对差异再合并，不要用重置命令丢弃其他部署改动。

## 推荐的一条命令

将下面 checkpoint、两个 Python 解释器路径换成现有机器上的实际路径。
先在命令末尾加 `--dry-run` 检查生成的命令；去掉后才会加载模型和运行仿真。

```bash
python script/run_c3ache_diagnostics.py \
  --checkpoint /models/lingbot-va-posttrain-libero-long \
  --out-dir outputs/diagnostics-01 \
  --gpu 0 \
  --server-python /path/to/server-env/bin/python \
  --client-python /path/to/libero-env/bin/python \
  --offload
```

默认在**一张已有 GPU 上串行**执行以下四组，不租机器、不启动训练或后续 full 队列：

| 输出目录 | 实际执行的策略 | 用途 |
|---|---|---|
| `baseline/` | 缓存关闭，全部计算 | 参考轨迹 |
| `baseline_repeat/` | 再跑同一 baseline | 判断运行/仿真本身的重复波动 |
| `full_refresh/` | 开启缓存但 interval=1，全部计算 | 检查缓存接入的全算路径 |
| `shadow/` | 全部计算，旁路测量旧残差 | 定位 step/σ/动作维度上的误差 |

默认 task 范围为 `[1,4)`，即 task 01、02、03，每任务只跑 episode 0。
base seed=0，对应 seed 100000、200000、300000。每组 3 回合，共 **12 回合**。
任务范围和回合数可以用 `--task-start`、`--task-end`、`--episodes` 调整。

默认诊断窗口是 **`1..49`、interval=2**，用于测量所有未被强制保护的步骤。
这和原先实际复用实验的 `5..39` 不同：它扩大的是观测范围，**不会让机器人执行这些近似动作**。
若只想检查原窗口，加 `--cache-start-step 5 --cache-end-step 39`，并使用新的输出目录。

默认保存视频和动作。只减少磁盘开销时可用 `--no-save-videos`；需要减少一组控制实验时，
可显式用 `--skip-baseline-repeat`，但这会失去直接的重复波动对照。
`--offload` 将 VAE/T5 放在 CPU，四组设置相同；24 GB 卡优先沿用之前已跑通的显存设置。
本脚本不进行两卡模型切分，也没有宣称已经测得新诊断的 GPU 峰值显存。

脚本会先检查环境资产、初始状态和 EGL 渲染，再启动服务端。每组完成或报错后只关闭自己启动的进程。
中断后可在代码、环境、参数都没变的条件下使用 `--resume`。已完成 episode 原子保存；
元数据变化则应新建输出目录。不同终端并行运行时必须使用不同 GPU、输出目录和两种端口。

## 查看和重新汇总

四组全部完成后自动生成：

- `diagnostics.json`：完整元数据、逐 step/逐任务统计和动作轨迹一致性检查。
- `diagnostics.csv`：按 step 排列的平均/P95 误差，便于绘图。
- 各组 `episodes/*.json`：逐 chunk、逐 step 原始标量和完整动作数组。
- 各组 `videos/*.mp4`：执行完整策略的机器人视频；不是旧 cached 失败轨迹的视频。
- 各组服务器日志和启动命令。

也可以在不占用 GPU 的情况下重新汇总，客户端环境即可：

```bash
python -m evaluation.libero.summarize_c3ache_diagnostics \
  --shadow outputs/diagnostics-01/shadow \
  --baseline outputs/diagnostics-01/baseline \
  --baseline-repeat outputs/diagnostics-01/baseline_repeat \
  --full-refresh outputs/diagnostics-01/full_refresh \
  --output outputs/diagnostics-01/diagnostics.json
```

如果跳过了 baseline repeat，去掉对应参数。单独运行 shadow 时可省略全部对照目录，
但报告不会因此声称全算轨迹已经通过一致性检查。未完整结束的目录不会冒充完整报告。

建议按下面顺序读结果：

1. `baseline_control_comparisons`：先看 baseline repeat 与 interval=1 对照是否一致。
2. `trace_comparisons.baseline`：shadow 与 baseline 的动作哈希、成功结果、环境步数是否一致。
   有差异时查看第一个不同的 chunk 和动作最大绝对差，并参照 baseline 自身的重复波动。
3. `hypothetical_reuse_samples` 必须大于零，且 `nonfinite_reuse_samples` 应为零。
   若任务在建立参考前已经结束，可能没有复用样本，不能当作“误差为零”。
4. 看逐 step 的 `update_rmse`、逐动作通道误差、逐任务变化，并对照 σ 和 Δσ。
5. 将跨 chunk 误差与 `same_state_roundtrip` 误差对照，再决定是否优先排查精度或残差失配。

不能只按余弦相似度或某个平均误差自动认定安全。当前没有预设“低于某个阈值就保证成功”的规则。
本报告是完整策略轨迹上的局部反事实检测，不能代替真正缓存策略的闭环评测，也不能证明成功率下降 ≤1 pp。
旁路模式实际 `reused_calls` 恒为 0；只有 `hypothetical_reused_calls` 计数。普通速度比较器会拒绝诊断结果。

## 本地验证范围

CPU 小模型测试覆盖完整输出、有效 KV、随机数状态、原生视频/动作采样循环的一致性，
并验证 CFG、动作尺度、参考刷新、BF16 重建误差、非有限值记录和报告计数检查。
新增诊断模式尚未在完整发布 checkpoint 的 CUDA/LIBERO 环境上验证，真实误差数值需要运行上述命令后获得。
