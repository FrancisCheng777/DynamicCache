# C³ache → LingBot-VA 移植

## 方法与边界

移植 [C³ache](https://arxiv.org/abs/2606.08962) 的整栈残差规则，不引入训练参数。

设 `h0(c,k)` 是第 c 个动作 chunk、第 k 个去噪步的 action embedding，
`hL(c,k)` 是所有 Transformer blocks 的输出，尚未经过最后的 norm、timestep
scale/shift 和 action projection。全算参考中保存：

```text
R(reference_chunk, k) = hL(reference_chunk, k) - h0(reference_chunk, k)
```

可复用的 chunk 中使用：

```text
hL_approx(current_chunk, k) = h0(current_chunk, k) + R(reference_chunk, k)
velocity = native_output_head(hL_approx, current_timestep_embedding)
```

缓存的是 `R`，不是旧 `hL`、旧 action、旧 velocity 或训练得到的预测头。
全算路径直接返回原始 `hL`，不会为了复用同一段代码先做减法再加法，以免引入
BF16 舍入差异。残差保存为当前 activation dtype 的 detached tensor。

LingBot-VA 的视频和动作共享 Transformer，但在服务端分开调用。
本项目只对**动作去噪调用**应用缓存，视频生成完整执行。当前 CFG 的两个 batch
分支原样保留，包括原生 LIBERO `video guidance=5`、`action guidance=1` 时的双分支。

## 必须全算的位置

| 位置 | 行为 | 原因 |
|---|---|---|
| `frame_st_id == 0` | 全算，不作为残差参考 | 首个 action frame 被钉为零，与普通 chunk 语义不同 |
| 第一个普通 chunk | 全算，指定窗口内建立残差参考 | 从正常 token 布局建立参考 |
| 每个普通 chunk 的第 0 步 | 全算 | 保留上游临时 attention KV 分配/淘汰 |
| 缓存窗口外 | 全算 | 保留早期更新和低噪声尾部计算 |
| 刷新 chunk | 全算，更新参考 | 控制跨 chunk 参考的年龄 |
| action 循环额外的 `t=0, update_cache=1` | 全算，不缓存残差 | 提交预测 action KV；不是第 51 个去噪积分步 |
| `update_cache=2` | 全算 | 提交真实视频/动作历史 |
| 视频调用 | 全算 | 本轮移植只修改动作分支 |
| 开启梯度或模型处于训练模式 | 全算 | 防止缓存介入训练/误用自动求导 |

上游 `WanAttention.forward(update_cache=0)` 并非纯读操作：它会写入临时 KV，
空间不足时先淘汰最旧槽位，最后只将新槽位的 mask 清空，不恢复淘汰内容。
同一 chunk 的 token 数固定，因此保留第 0 步全算可以先完成这些淘汰。
缓存命中时跳过的是后续步，不向 KV 池写近似的 hidden states。
测试比较了相同输入轨迹在 KV 池满时的有效 `mask/id/is_pred/K/V`；无效槽位中的
过期字节没有比较意义。**不同闭环动作导致不同历史是预期行为**，测试不声称
两种策略在真实 rollout 中保持相同的 KV 内容。

## 刷新与失效

普通 chunk 从 0 开始计数，cold chunk 不参与计数。

| refresh interval | 含义 |
|---|---|
| `0` | 第一个普通 chunk 建立参考，后续不周期刷新；仅作为消融选项 |
| `1` | 每个 chunk 全算；用于全算路径一致性诊断 |
| `2` | 全算、复用、全算、复用……（默认初始实验设置） |
| `N > 1` | 1 个全算 chunk + N−1 个复用 chunk |

缓存条目用去噪步索引区分，同时验证**完整原生 `(timestep, sigma)` 序列**以及
checkpoint 路径、CFG 分支顺序/scale、action token 布局、KV namespace、dtype
和 device。重复/倒退的 frame ID、episode reset、布局或 schedule 改变会使旧参考
失效。条目还检查 tensor 的 shape/dtype/device。不满足条件时执行完整计算。

服务器固定持有一个 checkpoint；更换权重需要重启服务器。每个 episode 必须
发送 reset；prompt 在 reset 时更新。不要在同一个有状态服务器上并发运行多个
rollout。当前观测改变不会清空残差，否则无法进行跨 chunk 复用。

命中结果永远不写回残差缓存，避免递归地积累近似误差。缓存没有序列化到模型
`state_dict`，也没有改动模型的参数名、权重或 checkpoint 格式。

## 实现伪代码

```python
on_episode_reset(prompt, seed):
    set_episode_seed(seed)
    clear_residual_cache()
    native_reset_video_and_action_kv(prompt)

infer_chunk(observation, frame_id):
    video, action_noise = native_initialization(observation)
    begin_residual_chunk(frame_id, native_action_schedule, layout_and_cfg)

    video = native_full_video_loop(video)  # 包含预测视频 KV 提交

    for k, t in enumerate(native_action_timesteps):
        h0, timestep_features = native_action_embedding(action_noise, t)
        if reuse_allowed(k) and compatible_reference_exists(k):
            hL = h0 + residual[k]
        else:
            hL = all_transformer_blocks(h0, current_kv, timestep_features)
            if eligible_reference_step(k):
                residual[k] = detach(hL - h0)
        velocity = native_output_head(hL, timestep_features)
        action_noise = native_scheduler_step(velocity, action_noise, t)
        native_restore_cold_frame_if_needed(action_noise)

    native_full_action_forward(t=0, update_cache=1)
    return native_action_postprocess(action_noise)

after_executing_actions(observations, actions):
    native_full_real_history_update(observations, actions, update_cache=2)
```

## 初始实验参数不等于已经验证的参数

默认窗口 `5..39`、interval `2` 是待检验的起点。论文在其他模型/去噪 schedule
上的经验不能直接证明 LingBot-VA 在此窗口内的残差足够稳定。
LingBot-VA 命中缓存的动作步不重新计算当前观测对整栈的影响，只能通过其他全算步、
新的动作状态以及后续刷新吸收反馈。因此保留全算边界并不能保证成功率不下降。

如果 pilot 明显退化，优先缩短窗口（例如 `5..24`），保持 interval `2`。
如果仍退化，再保留更少缓存步或回到 interval `1` 排查基线。
最终配置需要独立的正式配对评测，不能把调参集的最高分作为确认结果。

整套 WAM 的速度收益可能有限，因为视频分支、VAE、真实历史更新和输出头仍然
计算。是否值得使用取决于实测的成功率、全流程耗时和实际命中比例。
