# 最小 PPO 训练与归一化规范

本文记录 PPO 基线训练、向量化与 `DuckVecNormalize` 归一化数据链的实现细节、运行命令与实测基准。

## 1. 代码分层架构

```text
src/reinforcelearning/
├── cli.py                    # 命令行入口：rl-train 与 rl-eval
└── training/
    ├── config.py             # TrainConfig：实验规模、种子、设备、归一化开关
    ├── env_factory.py        # 环境工厂：Monitor + DummyVecEnv + DuckVecNormalize
    ├── vec_normalize.py      # 特化归一化：保留 command 维原始物理尺度
    └── runner.py             # 纯编排层：build_model / train / evaluate
```

- **职责边界**：`DuckEnvConfig` 负责物理世界契约，`TrainConfig` 负责算法实验参数。
- **环境一致性**：`env_factory` 是环境创建唯一入口，训练与评估共用相同包装器栈。

## 2. 核心调用逻辑

训练并保存模型与统计量对：

```python
vec_env = make_vec_env(n_envs=4, seed=0)
model = PPO("MlpPolicy", vec_env, device="cpu", gamma=0.99)
model.learn(total_timesteps=100_000)
model.save("models/duck_ppo.zip")
vec_env.save("models/duck_ppo_vecnormalize.pkl")
```

评估必须加载同一份统计量并冻结更新：

```python
vec_env = make_vec_env(n_envs=1, seed=0, normalize=False)
vec_env = DuckVecNormalize.load("models/duck_ppo_vecnormalize.pkl", vec_env)
vec_env.training = False       # 冻结 obs 统计均值与方差
vec_env.norm_reward = False    # 回放保持原始物理奖励
model = PPO.load("models/duck_ppo", env=vec_env, device="cpu")
```

## 3. 运行命令参考

```bash
# 标准训练（默认 100k 步、4 环境、CPU、开启归一化）
uv run rl-train

# 快速冒烟测试（单次更新）
uv run rl-train --total-timesteps 4096 --n-envs 2 --model-path models/smoke_ppo

# 对照实验（关闭归一化）
uv run rl-train --no-normalize --model-path models/duck_ppo_raw

# 策略评估（自动配对加载 *_vecnormalize.pkl）
uv run rl-eval models/duck_ppo --episodes 3

# 跳过统计量，使用原始观测评估
uv run rl-eval models/duck_ppo --episodes 3 --raw

# 窗口实时回放
uv run rl-eval models/duck_ppo --episodes 1 --render

# 监控训练面板
uv run tensorboard --logdir runs
```

> **配对约定**：`<model>.zip` 与 `<model>_vecnormalize.pkl` 必须同目录同名。遗失统计量文件将导致输入分布偏移（Covariate Shift）。

## 4. 架构决策与实测基准

| 决策点 | 选型 | 理由与实测依据 |
| :--- | :--- | :--- |
| **向量化环境** | `DummyVecEnv` | MuJoCo 单步耗时极短，单进程顺序执行易调试且避免 IPC 开销 |
| **计算设备** | `device="cpu"` | 浅层 MLP 策略在 CPU 上吞吐显著高于 GPU（见下方基准） |
| **指令归一化** | 排除 `command_speed` | 指令随课程学习扩展分布；若参与均值方差统计，会导致策略对同数值指令的语义理解漂移 |
| **奖励归一化** | 折扣回报标准差缩放 | 匹配 SB3 算法设计；`gamma` 必须与 PPO 严格同步 |
| **包装顺序** | `Monitor` 内层，`VecNormalize` 外层 | 确保 TensorBoard 记录的 `ep_rew_mean` 反映真实物理回报 |

### 吞吐基准（RTX 4060 Ti + 现代多核 CPU，16k 步测试）

| 运行配置 | 实测采样吞吐 | 结论 |
| :--- | :---: | :--- |
| **CPU, `n_envs=4`** | $\approx 2313\text{ fps}$ | **默认推荐**：吞吐高且系统负载均衡 |
| **CUDA, `n_envs=4`** | $\approx 1137\text{ fps}$ | 小批次下张量在 GPU/CPU 间反复拷贝成为瓶颈 |
| **CPU, `n_envs=8`** | $\approx 2603\text{ fps}$ | 边际提升有限，4 进程已接近瓶颈 |

## 5. 归一化数学形式

- **观测标准化**（除最后一维 command 保持原值外）：
  $$\hat{o}_i = \text{clip}\left(\frac{o_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}},\ -10,\ 10\right)$$

- **奖励标准化**：
  $$\hat{r}_t = \text{clip}\left(\frac{r_t}{\sqrt{\text{Var}(R_t) + \epsilon}},\ -10,\ 10\right),\quad R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

## 6. 阶段验收记录

| 运行配置 | 回放表现 | 训练收敛指标 | 现象解读 |
| :--- | :--- | :--- | :--- |
| **4k 步（无归一化）** | 1000 步超时不跌倒，$\bar{v}_x \approx 0.00$ | — | 策略陷入静态平衡局部最优（静态奖励约 $0.8/\text{step}$） |
| **100k 步（无归一化）** | 34~46 步摔倒，$\bar{v}_x \approx 0.2\sim 0.4$ | `value_loss` $\approx 28$, `expl_var` $\approx 0.52$ | 开始尝试前冲，但 Critic 拟合剧烈震荡 |
| **100k 步（有归一化）** | 41~58 步摔倒，$\bar{v}_x \approx 0.22\sim 0.28$ | `value_loss` $\approx 0.28$, `expl_var` $\approx 0.58$ | Critic 拟合稳定 100 倍，存活步数增加，步态探索更平稳 |

## 7. 待解决瓶颈

1. **步态未收敛**：100k 步仍以微前倾或静态站立为主，需加大训练量与速度课程。
2. **多指标可观测性**：奖励分项仅写入 `info`，待接入 TensorBoard 分项标量图。
3. **指令维度限制**：当前仅支持一维前进速度，尚未引入偏航转向。
4. **训练规模**：单机 CPU 训练，尚未做并行调参或超参搜索。
