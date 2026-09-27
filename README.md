# DynamicCache

本项目从 [官方 LingBot-VA](https://github.com/Robbyant/lingbot-va/tree/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb) 的 `7c6ffa9` 提交派生，保留上游代码、历史和 Apache-2.0 许可证。C³ache 方法归属[原论文作者](https://arxiv.org/abs/2606.08962)；这是独立移植实现，不是论文官方代码。

**当前状态：固定窗口尚未通过质量验收；优先运行新增的[误差检测流程](docs/DIAGNOSTICS.md)。** 已收到小规模闭环评测中的动作效率与成功率退化反馈。误差检测始终执行完整策略，旁路测量残差复用误差；代码已通过 CPU 测试，新增诊断模式尚待完整 checkpoint 的 CUDA/LIBERO 验证。“成功率下降不超过 1 个百分点”仍是未被证明的目标。无需训练或修改模型权重，但需要使用已完成 LIBERO 后训练的 checkpoint。

## 做了什么

- 对动作分支，在相同去噪步缓存整栈残差 `R = h_L - h_0`。命中时用 `h_0_current + R_reference` 跳过全部 Transformer blocks，仍运行当前输出头。
- 视频生成、真实观测的 KV 更新、预测结果的 KV 提交保持全算。首个 cold chunk 全算；第一个普通 chunk 建立参考；每个 chunk 的动作第 0 步全算，以保持上游 KV 槽位淘汰行为。
- 默认关闭缓存。启用后的初始实验参数为 **0-based 第 5–39 步（含端点），每 2 个普通 chunk 刷新**。原生视频 20 步、动作 50 步、CFG、action horizon 和归一化配置不变。
- 提供同种子配对评测、逐回合原子保存、严格续跑检查、分阶段耗时和成功率统计区间。
- 提供 baseline 重复、interval=1 全算对照和不影响执行动作的旁路误差检测，记录逐 σ、逐动作通道误差与 BF16 重建误差。

`DynamicCache` 是项目名称。当前方法使用固定刷新间隔，没有实现额外的自适应刷新算法、训练头或 Next Forcing 模块。

## 安装和 checkpoint

先克隆仓库，所有下列命令从仓库根目录执行：

```bash
git clone https://github.com/FrancisCheng777/DynamicCache.git
cd DynamicCache
```

服务端使用上游的 Python 3.10 / PyTorch 2.9 / CUDA 12.6 环境配置：

```bash
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-inference.txt
```

服务端保留上游显式选择的 PyTorch SDPA attention。此路径不需要安装 FlashAttention；显式使用 `attn_mode="flashattn"` 时仍需要该库。

下载 [LIBERO-Long 官方 checkpoint](https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long)：

```bash
hf download robbyant/lingbot-va-posttrain-libero-long --local-dir checkpoints/lingbot-va-posttrain-libero-long
```

不要使用未适配 LIBERO 的基础 checkpoint 来判断缓存方法的成功率。评测无需下载训练示范数据，但必须安装 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 及其仿真依赖、任务资产和初始状态。

建议保留一个已验证能运行 LIBERO 的客户端环境。旧版 LIBERO 的 PyTorch 依赖可能与服务端冲突；使用 `--client-python /path/to/libero-env/bin/python` 分开运行。客户端需 Python ≥ 3.10，并安装 `requirements-eval-client.txt`。客户端不会导入 LingBot-VA 的模型依赖。详细说明见 [评测指南](docs/EVALUATION.md)。

## 先定位复用误差

沿用已跑通的服务端、LIBERO 客户端和 checkpoint，用一张已有 GPU：

```bash
python script/run_c3ache_diagnostics.py \
  --checkpoint checkpoints/lingbot-va-posttrain-libero-long \
  --out-dir outputs/diagnostics-01 --gpu 0 \
  --server-python /path/to/server-env/bin/python \
  --client-python /path/to/libero-env/bin/python --offload
```

先加 `--dry-run` 查看命令。默认 task 01～03，每组各 1 回合，共 4 组、12 回合，保存视频和动作，
最终生成 `diagnostics.json` 与 `diagnostics.csv`。实际动作始终来自完整计算。
诊断默认观察 `1..49`，不代表推荐实际复用该窗口。参数解释、运行步骤和结果解读见[完整教程](docs/DIAGNOSTICS.md)。

## 正式缓存行为的配对评测

下面会真正执行近似缓存策略；建议先完成误差诊断与小范围质量筛选。

```bash
python script/run_libero_pair.py \
  --checkpoint checkpoints/lingbot-va-posttrain-libero-long \
  --out-dir outputs/smoke \
  --preset smoke --gpu 0
```

该脚本在一张**现有 GPU**上依次启动 baseline 和 cached 服务端，各跑 1 个任务 × 2 个回合，最后停止自己启动的服务端。不会租用机器或启动训练。用 `--dry-run` 可先查看完整命令。

上游当前配置 `enable_offload=False`。脚本保持该默认值；显存不足时可以对两组同时加 `--offload`，将 VAE/T5 放在 CPU。CPU offload 可能显著拖慢评测，不能直接用另一组的 offload 设置比较速度。尚未验证特定 4090 / A800 配置的显存和耗时。

```bash
# 小规模 pilot：每组 10 个任务 × 2 回合
python script/run_libero_pair.py \
  --checkpoint checkpoints/lingbot-va-posttrain-libero-long \
  --out-dir outputs/pilot --preset pilot --gpu 0

# 参数固定后的正式对照：每组 10 个任务 × 50 回合
python script/run_libero_pair.py \
  --checkpoint checkpoints/lingbot-va-posttrain-libero-long \
  --out-dir outputs/full --preset full --gpu 0 --base-seed 10000
```

同一命令加 `--resume` 可以继续中断的评测；模型、代码、环境、初始状态或参数变化时会拒绝续跑。需要单独启动服务器或调整缓存窗口时，参见 [评测指南](docs/EVALUATION.md)。

## 看哪些结果

每次配对运行产生 `baseline/`、`cached/`、两份服务器日志以及 `comparison.json`：

- `success.difference_pp`：cached 成功率减 baseline，单位为**百分点**。
- `success.point_estimate_within_margin`：样本中的差值是否 ≥ −1 pp。
- `success.noninferiority`：统计区间是否支持该界限；smoke/pilot 通常是 `not_established`。
- `cached_stack_hits`：实际跳过整栈的次数；如果为 0，这次运行没有检验到复用路径。
- `server_action_loop_ms`、`server_video_loop_ms`、`infer_rpc_ms`、`policy_cycle_ms`：区分动作计算、视频计算、请求耗时和带观测反馈的策略周期。

不能把动作 Transformer 的理论省算比例当成整个 WAM 的实际加速比。视频分支继续运行，最终以完整闭环评测为准。

## 验证与实现位置

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests -q
```

- [残差缓存](wan_va/modules/c3ache.py)
- [旁路误差检测](wan_va/modules/c3ache_diagnostics.py) 和 [使用教程](docs/DIAGNOSTICS.md)
- [模型接入](wan_va/modules/model.py) 和 [服务器接入](wan_va/wan_va_server.py)
- [方法、边界和伪代码](docs/METHOD.md)
- [评测协议与运行命令](docs/EVALUATION.md)
- [实际验证范围](docs/VALIDATION.md)
- [上游原始 README](README_UPSTREAM.md) 和 [来源记录](UPSTREAM.md)

本项目遵循 [Apache-2.0](LICENSE.txt)。模型权重和 LIBERO 资产遵循各自发布方的许可。
