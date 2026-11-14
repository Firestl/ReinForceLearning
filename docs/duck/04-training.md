# 最小 PPO 训练与归一化

本文记录 M3（PPO 冒烟训练）与 M4 首项（`VecNormalize`）的实现方式、运行命令与
验收结果。目标是验证 **环境 → 向量化/归一化 → PPO → 保存 → 回放** 这条数据链，
并让优化过程稳定下来；步态质量仍待后续调优。

## 1. 代码结构

```text
src/reinforcelearning/
├── cli.py                    # argparse 入口：rl-train / rl-eval
└── training/
    ├── config.py             # TrainConfig：训练规模、种子、设备、gamma、归一化开关
    ├── env_factory.py        # Duck-v0 + Monitor + DummyVecEnv + DuckVecNormalize
    ├── vec_normalize.py      # 只归一化非 command 维度的 VecNormalize 子类
    └── runner.py             # build_model / train / evaluate
```

分层理由：

- `TrainConfig` 与 `DuckEnvConfig` 分离：前者描述一次实验，后者描述物理世界契约。
- `env_factory` 是环境创建的唯一入口，训练与评估共用同一套包装，避免观测/终止语义漂移。
- `DuckVecNormalize` 单独成文件：归一化规则（尤其 command 例外）是被保存进 `.pkl` 的契约。
- `runner` 只做编排，不解析命令行；`cli` 只解析参数，不含训练逻辑。

## 2. 核心调用

```python
vec_env = make_vec_env(n_envs=4, seed=0)           # DummyVecEnv + DuckVecNormalize
model = PPO("MlpPolicy", vec_env, device="cpu", gamma=0.99)
model.learn(total_timesteps=100_000)
model.save("models/duck_ppo.zip")
vec_env.save("models/duck_ppo_vecnormalize.pkl")    # 统计量与模型成对保存
```

回放时必须加载同一份统计量并冻结：

```python
vec_env = make_vec_env(n_envs=1, seed=0, normalize=False)
vec_env = DuckVecNormalize.load("models/duck_ppo_vecnormalize.pkl", vec_env)
vec_env.training = False       # 冻结 obs 统计量
vec_env.norm_reward = False    # 返回原始奖励
model = PPO.load("models/duck_ppo", env=vec_env, device="cpu")
```

## 3. 运行命令

```bash
# 训练（默认 100000 步、4 个并行环境、CPU、开启归一化）
uv run rl-train

# 小规模冒烟：只跑一次 PPO 更新
uv run rl-train --total-timesteps 4096 --n-envs 2 --model-path models/smoke_ppo

# 关闭归一化（对照组）
uv run rl-train --no-normalize --model-path models/duck_ppo_raw

# 评估：自动加载模型旁的 *_vecnormalize.pkl
uv run rl-eval models/duck_ppo --episodes 3

# 忽略统计量，用原始观测评估
uv run rl-eval models/duck_ppo --episodes 3 --raw

# 打开 MuJoCo 窗口实时观看
uv run rl-eval models/duck_ppo --episodes 1 --render

# 查看训练曲线（runs/ 下是 TensorBoard event 文件）
uv run tensorboard --logdir runs
```

模型与统计量命名规则：`<model>.zip` 与 `<model>_vecnormalize.pkl` 同目录。丢失 `.pkl`
后策略看到的观测分布会改变，这是"训练正常、加载变傻"的最常见原因。

## 4. 关键决策

| 决策 | 原因 |
| --- | --- |
| `DummyVecEnv` 而非 `SubprocVecEnv` | MuJoCo 单步开销小，同进程顺序执行更易调试；吞吐不足再换 |
| 默认 `device="cpu"` | MLP 策略在 GPU 上反而更慢，实测见下方"设备与吞吐" |
| `DuckVecNormalize` 不归一化 `command_speed` | 指令范围会随课程变化；若参与运行统计，策略对同一指令的理解会在训练中途漂移 |
| 奖励归一化用折扣回报标准差 | SB3 的实现；`gamma` 必须与 PPO 一致，否则回报统计的时间尺度错误 |
| `Monitor` 在里层、`VecNormalize` 在外层 | TensorBoard 的 `ep_rew_mean` 保持原始奖励单位，便于解释 |
| 暂不写自定义 callback | SB3 的 `Monitor` 已提供 `ep_rew_mean` / `ep_len_mean` |

### 设备与吞吐实测

环境：RTX 4060 Ti；16384 步、4 个环境、归一化开启、PPO 默认超参。

| 配置 | 实测吞吐 |
| --- | --- |
| CPU，`n_envs=4` | ≈2313 fps |
| CUDA，`n_envs=4` | ≈1137 fps |
| CPU，`n_envs=8` | ≈2603 fps |


## 5. 归一化细节

观测（最后一维 command 除外）：

$$
\hat{o}_i = \text{clip}\left(\frac{o_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}},\ -10,\ 10\right)
$$

奖励：

$$
\hat{r}_t = \text{clip}\left(\frac{r_t}{\sqrt{\text{Var}(R_t) + \epsilon}},\ -10,\ 10\right),
\quad R_t = \sum_k \gamma^k r_{t+k}
$$

- 统计量在线更新，只来自训练数据；评估时 `training=False` 冻结。
- 观测是 34 维扁平 `Box`，SB3 的 `norm_obs_keys` 无法排除单维，因此用
  `DuckVecNormalize.normalize_obs` 在归一化后把最后一维还原为原始 `command_speed`。
- `normalize_obs` 同时覆盖 reset、step 和 `terminal_observation`，三条路径语义一致。

## 6. 验收记录

| 运行 | 回放结果 | 训练指标 | 解读 |
| --- | --- | --- | --- |
| 4096 步，无归一化 | 1000 步全部 `timeout`，`mean_vx ≈ 0.00` | — | 学会"站着不动"：静态奖励约 0.8/步 |
| 100000 步，无归一化 | 34~46 步摔倒，`mean_vx ≈ 0.2~0.4`，平均回报 43.65 | `value_loss ≈ 28`，`explained_variance ≈ 0.52` | 开始前扑，critic 拟合困难 |
| 100000 步，有归一化 | 41~58 步摔倒，`mean_vx ≈ 0.22~0.28`，平均回报 55.17 | `value_loss ≈ 0.28`，`explained_variance ≈ 0.58` | 优化明显更稳，存活更长，但仍未形成步态 |