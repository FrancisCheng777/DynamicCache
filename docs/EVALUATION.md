# LIBERO 配对评测

## 前提与环境

1. 一台已有的 Linux NVIDIA GPU 机器，完整 CUDA/PyTorch 驱动环境。
2. 服务端使用上游 PyTorch 2.9.0、diffusers 0.36.0、transformers 4.55.2。
3. 使用 `robbyant/lingbot-va-posttrain-libero-long` 完整本地 snapshot，包括
   `transformer/`、`vae/`、`text_encoder/`、`tokenizer/`。
4. LIBERO 已安装并配置资产路径，`OffScreenRenderEnv` 可以用 EGL 渲染。

官方 LIBERO 的旧安装说明包含旧 Python/PyTorch。可以在 Python 3.10 的独立
客户端环境中保留已验证的 LIBERO / robosuite / MuJoCo / PyTorch 组合，安装
`requirements-eval-client.txt`，通过 `--client-python` 指定解释器。
新的客户端不会导入 `wan_va` 包的 eager model imports，通信格式仍使用上游
原有的 NumPy/MessagePack codec。

尤其注意：旧 LIBERO 调用 `torch.load(init_states_path)` 而没有显式设置
`weights_only`。在较新 PyTorch 上，如果官方初始状态资产触发兼容问题，应使用
兼容的客户端环境，不要为了跑通而替换初始状态或忽略加载失败。
自动配对脚本在加载大模型前会先检查初始状态、任务文件和真实仿真渲染。

目前未在真实 NVIDIA GPU 上验证本移植。不要从 CPU 测试推断 4090 是否能装下
完整模型。上游当前 `enable_offload=False`；需要时在同一对照里统一使用
`--offload`。增加独立 GPU 可以并行不同 episode/任务，但这版配对 launcher
有意按一张 GPU、每次一个 rollout 执行；它不自动进行多卡模型切分。

## 最短运行方式

```bash
python script/run_libero_pair.py \
  --checkpoint /models/lingbot-va-posttrain-libero-long \
  --out-dir outputs/smoke --preset smoke --gpu 0 \
  --client-python /path/to/libero-env/bin/python
```

| preset | 每组回合数 | 用途 |
|---|---:|---|
| `smoke` | 1 task × 2 = 2 | 环境、输出、缓存命中和整条流程是否跑通 |
| `pilot` | 10 tasks × 2 = 20 | 初步退化筛查和实际耗时估计 |
| `full` | 10 tasks × 50 = 500 | 固定参数后的正式对照，共 1000 个 rollout |

`--base-seed 10000` 可以作为调参后的另一组噪声种子。它不是新的任务集，也没有
更换官方初始状态集合；不要声称因此测到了新任务泛化。

默认视频保存关闭，两组一致；`--save-videos` 同时开启两组的录像。
上游逐 chunk 调试 tensor 保存在配对脚本中被关闭，避免大量磁盘 I/O。
这些设置被记录在 manifest 中。服务器单独运行时仍保留上游的调试保存默认。

脚本会记录命令和服务器日志，评测结束或出错时关闭自己启动的服务器。
`--dry-run` 不加载模型、不运行仿真。端口被占用或客户端环境检查失败时会提前退出。

## 单独启动 / 自定义窗口

全算基线：

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
  --nproc_per_node=1 --master_port 29061 wan_va/wan_va_server.py \
  --config-name libero --port 29056 \
  --checkpoint /models/lingbot-va-posttrain-libero-long \
  --profile-inference --no-save-debug --save_root outputs/server-debug
```

缓存组使用相同命令，增加：

```text
--c3ache --cache-start-step 5 --cache-end-step 39 --cache-refresh-interval 2
```

客户端：

```bash
python -m evaluation.libero.evaluate_c3ache \
  --host 127.0.0.1 --port 29056 --suite libero_10 \
  --task-start 0 --task-end 10 --episodes 50 --base-seed 10000 \
  --expected-mode baseline --out-dir outputs/full/baseline
```

缓存组改成 `--expected-mode cached --out-dir outputs/full/cached`。
只检查仿真环境时加 `--check-env`，不会连接模型服务端。

```bash
python -m evaluation.libero.compare_c3ache \
  --baseline outputs/full/baseline --cached outputs/full/cached \
  --margin-pp 1 --output outputs/full/comparison.json
```

## 保持的上游闭环协议

- LIBERO-Long 使用 `libero_10`；其他 suite 虽可手动指定，但没有据此宣称该 checkpoint
  在其余 suite 上已后训练或复现成功。
- 两个相机、128×128、图像垂直翻转。
- task 的第 i 个官方初始状态对应第 i 个 episode，不隐式重复初始状态。
- reset 后先做 5 个零动作。
- 原生 action 返回形状 `7 × 4 × 4`；首个 chunk 跳过首个固定 frame，执行 12 个动作；
  后续 chunk 执行 16 个动作。只使用返回的前 7 个有效动作维度，与上游一致。
- 成功时停止；否则提交观测序列及完整 action chunk，更新真实 KV。
- 保持上游 `env.env.timestep < 800` 的 **chunk 边界检查**。因此实际 timestep
  可能在最后一个 chunk 超过 800；没有悄悄改成每步截断或其他 VLA 的 horizon。
- 每个 episode 的种子为 `base_seed + task_id * 100000 + episode_id`，同时设置环境、
  NumPy/Python 和服务端 PyTorch。固定种子不等于跨硬件 CUDA bitwise determinism。

## 完整性、续跑与身份记录

每组有一个 `manifest.json`，和 `episodes/<task>_<episode>.json`。
每个 episode 原子写入并 fsync；错误记录不会被当成正常失败回合静默吞掉。
全部预定回合完成后才将 manifest 的 `complete` 设为 true。

继续运行使用原命令加 `--resume`。已成功记录的回合跳过，错误/未完成回合重新
执行；配置、代码、模型或 runtime 变化会拒绝续跑。比较器拒绝缺失、额外、
错误或 seed 不匹配的回合，也拒绝不一致的 checkpoint、原生设置和环境。

checkpoint 身份由本地路径及相对文件名/大小/mtime 生成指纹。
**这不是权重内容的 SHA256 校验**；适用于同一份未修改本地 snapshot 的配对运行。
换机器复制权重后可能因为 mtime 不同而拒绝比较，此时应重新进行同机配对实验。
发布正式研究结果时，额外保存下载时的 Hugging Face revision 和权重内容校验。

## 指标解释与验收

成功率用百分点：假如 baseline 为 96%，目标下界为 95%，不是相对乘以 0.99。
报告每个任务、配对 gain/loss、总差值以及 Bonferroni-Wilson 近似 95% 区间。
该区间按固定任务集合上的独立 episode pair 假设计算，是近似统计证据，不是保证。

- `supported`：区间下界 ≥ −1 pp。
- `exceeds_margin`：区间上界 < −1 pp。
- `not_established`：当前样本不能确立该边界，不能解读为通过。

样本点估计 ≥ −1 pp 和区间支持这个界限会分别记录。
20 回合完全相同也不会得到零宽度区间；小样本无法证明 1 pp 非劣。
500 回合不保证足以得出结论，尤其存在较多 gain/loss 时；需要更多预先设定的
重复种子，而不能反复增加到恰好“通过”为止。调参结果与确认结果应分开。

耗时单位均为毫秒：

| 指标 | 包含范围 |
|---|---|
| `server_action_loop_ms` | 原生动作循环，包括最后一次 `t=0` KV 提交 |
| `server_video_loop_ms` | 原生视频循环，包括视频 KV 提交 |
| `server_prepare_ms` | 首帧观测编码、noise/schedule 准备等 |
| `server_infer_ms` | `_infer` 内完整推理和 action postprocess |
| `infer_rpc_ms` | 客户端测量的一次 infer 请求/响应 |
| `history_rpc_ms` | 执行后真实观测编码及 KV 更新的请求/响应 |
| `policy_cycle_ms` | infer RPC + history RPC；终止 chunk 没有 history RPC |
| `episode_wall_ms` | reset/settling/推理/环境 step/反馈；不含 env 构造和录像文件编码 |

两组开启相同的 `--profile-inference`，在阶段边界同步 CUDA。阶段同步本身可能
改变实际吞吐，因此这里比较的是相同 profiling 设置下的耗时。也报告冷启动以外
的 infer 延迟。不同策略的闭环轨迹和 episode 长度可能不同，延迟中位数之比
不是在完全相同观测 trace 上测得的纯算子速度比。

先用 pilot 测出该机器上的回合耗时，再决定完整评测的租卡时长；本仓库没有
虚构的 4090/A800 耗时或缓存成功率。
