# 最小 PPO 训练

本文记录 M3（PPO 冒烟训练）的实现方式、运行命令与验收结果。目标是验证
**环境 → 向量化 → PPO → 保存 → 回放** 这条数据链，不追求步态质量。

## 1. 代码结构

```text
src/reinforcelearning/
├── cli.py                    # argparse 入口：rl-train / rl-eval
└── training/
    ├── config.py             # TrainConfig：训练规模、种子、设备、输出路径
    ├── env_factory.py        # Duck-v0 + Monitor + DummyVecEnv
    └── runner.py             # build_model / train / evaluate
```

分层理由：

- `TrainConfig` 与 `DuckEnvConfig` 分离：前者描述一次实验，后者描述物理世界契约。
- `env_factory` 是环境创建的唯一入口，训练与评估共用同一套包装，避免观测/终止语义漂移。
- `runner` 只做编排，不解析命令行；`cli` 只解析参数，不含训练逻辑。

## 2. 核心调用

整条训练链在 SB3 里只有三行核心代码：

```python
model = PPO("MlpPolicy", vec_env, device="cpu", seed=0)
model.learn(total_timesteps=100_000)
model.save("models/duck_ppo.zip")
```

回放：

```python
model = PPO.load("models/duck_ppo", env=env, device="cpu")
action, _ = model.predict(obs, deterministic=True)
```

## 3. 运行命令

```bash
# 训练（默认 100000 步、4 个并行环境、CPU）
uv run rl-train

# 小规模冒烟：只跑一次 PPO 更新
uv run rl-train --total-timesteps 4096 --n-envs 2 --model-path models/smoke_ppo

# 评估：确定性策略回放
uv run rl-eval models/duck_ppo --episodes 3

# 打开 MuJoCo 窗口实时观看
uv run rl-eval models/duck_ppo --episodes 1 --render

# 查看训练曲线（runs/ 下是 TensorBoard event 文件）
uv run tensorboard --logdir runs
```

## 4. 关键决策

| 决策 | 原因 |
| --- | --- |
| `DummyVecEnv` 而非 `SubprocVecEnv` | MuJoCo 单步开销小，同进程顺序执行更易调试；吞吐不足再换 |
| 默认 `device="cpu"` | MLP 策略在 GPU 上反而更慢：实测 4096 步从 228 fps 提升到 1621 fps |
| 暂不接 `VecNormalize` | 保持最小可读版本；归一化是下一阶段的第一优先级 |
| 暂不写自定义 callback | SB3 的 `Monitor` 已提供 `ep_rew_mean` / `ep_len_mean` |

## 5. 验收记录（2026-11-14）

| 运行 | 结果 | 解读 |
| --- | --- | --- |
| 4096 步（2 envs） | 回放 1000 步全部 `timeout`，`mean_vx ≈ 0.00` | 策略学会"站着不动"：`upright + height + alive` 白拿 0.8 分 |
| 100000 步（4 envs） | 每回合 34~97 步后摔倒，`mean_vx ≈ 0.2~0.4 m/s` | 开始前扑，但尚未形成步态 |

结论：**训练链路正确，学习效果不足**。这是奖励结构与训练规模的正常结果，不是代码错误。
下一步按优先级：

1. 接 `VecNormalize` 并保存统计量；`command_speed` 维度用固定缩放，避免课程切换时统计量漂移。
2. 把 `info["reward_terms"]` 扁平化后交给 `Monitor(info_keywords=...)`，让 TensorBoard 能看到分项曲线。
3. 评估指标补充平均前进速度、摔倒率、左右脚接触比例。
4. 加长训练并引入速度课程（`0.0~1.0` → `2.0~3.5 m/s`），再判断是否需要调奖励权重。

## 6. 测试

`tests/test_training.py` 覆盖训练层的契约，而不是训练效果：

- `TrainConfig` 的非法输入拒绝与 `.zip` 路径补全；
- `DummyVecEnv` 的批处理形状 `(n_envs, 34)` / `(n_envs, 8)`；
- 相同种子下 `make_env` 的重置结果可复现。
