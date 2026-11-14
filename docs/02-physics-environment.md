# 物理环境接口设计

本文定义第一版 `DuckEnv` 的 Gymnasium 接口。实现代码应严格遵守本规范；任何维度或语义变化都需要同步更新文档和测试。

## 1. 环境职责

`DuckEnv` 负责：

1. 加载 `assets/duck.xml` 并持有 `mujoco.MjModel` / `mujoco.MjData`。
2. 将归一化动作映射为关节目标角。
3. 以固定控制频率推进物理仿真。
4. 构造观测、计算奖励、判断终止和重置。
5. 向训练代码提供固定形状和数值范围的接口。

环境不负责算法实现，也不在训练循环中打开可视化窗口。

## 2. 控制循环与时间参数

| 参数 | 初始值 | 说明 |
| --- | --- | --- |
| `sim_dt` | 0.002 s | 物理步长 |
| `frame_skip` | 10 | 每个控制步内执行 10 次 `mj_step` |
| `control_dt` | 0.02 s | 控制频率 50 Hz |
| `episode_seconds` | 20.0 s | 每回合 1000 个控制步 |
| `max_episode_steps` | 1000 | 超时使用 `truncated=True` |

每个 `step(action)` 的流程：

```text
clip action 到 [-1, 1]
q_target = q_default + action_scale * action
data.ctrl = clip(q_target, joint_limits_margin)
for _ in range(frame_skip):
    mujoco.mj_step(model, data)
计算观测、奖励、终止标志
返回 obs, reward, terminated, truncated, info
```

## 3. 观测空间 Observation

第一版使用本体感知状态，不使用图像。观测向量为 `float32`，维度 **34**：

| 索引区间 | 维度 | 内容 | 说明 |
| --- | --- | --- | --- |
| `0:3` | 3 | 身体坐标系下的基座线速度 | `base_lin_vel` |
| `3:6` | 3 | 身体坐标系下的基座角速度 | `base_ang_vel` |
| `6:9` | 3 | 重力方向在身体坐标系的投影 | `projected_gravity` |
| `9:17` | 8 | 8 个关节角 | 顺序固定见模型文档 |
| `17:25` | 8 | 8 个关节角速度 | 与关节角同序 |
| `25:33` | 8 | 上一步归一化动作 | 用于平滑和可观测性 |
| `33:34` | 1 | 目标前进速度 `v_x_cmd` | 单位 m/s |

固定关节顺序：

```text
left_hip_roll, left_hip_pitch, left_knee_pitch, left_ankle_pitch,
right_hip_roll, right_hip_pitch, right_knee_pitch, right_ankle_pitch
```

设计注意事项：

- 不把世界坐标 `x, y` 放入观测，减少对起点和地图位置的过拟合。
- 初始阶段不使用历史帧堆叠；MLP 能处理马尔可夫状态。
- 观测中的速度、关节角等应在训练脚本中通过 `VecNormalize` 做归一化。
- 后续若加入足底接触、步态相位或转向指令，应升维并更新本章节。

`observation_space` 建议：

```python
spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(34,),
    dtype=np.float32,
)
```

实际实现时应逐项检查 `np.isfinite(obs).all()`，避免关节限位或接触爆炸导致 NaN 传入网络。

## 4. 动作空间 Action

```python
spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
```

- 动作是目标关节角的增量，不是力矩，也不是关节速度。
- 默认姿态 `q_default` 来自模型文档。
- 每个关节使用独立 `action_scale`。
- 映射后的目标角度必须裁剪到关节限位内，并保留安全边界。
- 动作通过 `data.ctrl` 交给 MuJoCo 位置执行器。

第一版不加入随机动作噪声；探索由 PPO 的高斯策略负责。

## 5. 速度指令 Command

第一版为平地直线任务，每回合采样一个目标前进速度：

```text
v_x_cmd ~ Uniform(0.0, 2.5) m/s
```

之后可以通过课程学习逐步扩大范围。低速区间对应“行走”，高速区间对应“奔跑”，但步态是否真正变为奔跑需要通过视频和足底接触指标检查。

转向暂不实现。原因是没有髋 yaw 关节，且观测中只保留一维速度指令。后续若要转向，需要扩展模型关节和 command 维度。

## 6. 奖励函数 Reward

奖励函数应该由多个可解释项组成，并在 `info` 中返回各分项，不能只返回一个总数。

第一版建议形式：

```text
speed_error = v_x_body - v_x_cmd
tracking    = exp(-(speed_error / 0.5)^2)

reward =
    1.0  * tracking                         # 速度跟踪
  + 0.5  * upright                          # 身体直立
  + 0.2  * height_reward                    # 维持合理站立高度
  + 0.1  * alive                            # 存活
  - 0.05 * v_y^2                            # 减少侧向漂移
  - 0.05 * omega_z^2                        # 减少无意义自转
  - 0.005 * sum(tau_i^2)                    # 能量消耗
  - 0.01 * sum((a_t - a_{t-1})^2)           # 动作变化率
  - 0.01 * sum((q_target - q)^2)            # 关节跟踪误差
```

其中：

- `v_x_body` 是身体坐标系下的前进速度。
- `body_up_z = -projected_gravity[2]`；`upright = clip(body_up_z, 0.0, 1.0)`，或使用躯干 `xmat` 的 z 轴分量。
- 倾角可由 `tilt_angle = arccos(clip(-projected_gravity[2], -1.0, 1.0))` 计算。
- `height_reward` 应使用一个目标站立高度附近的平滑函数。
- 终止时是否额外加摔倒惩罚需要单独实验；建议先不要给太大惩罚。
- 所有奖励权重集中放在 `RewardConfig`，禁止散落在 `step()` 中。

### 奔跑阶段的变化

训练奔跑时优先调整目标速度和奖励，而不是改动作接口：

1. 扩大 `v_x_cmd` 范围到约 `2.0~3.5 m/s`。
2. 降低能量惩罚，允许更大的关节力矩和动作幅度。
3. 加入足底接触/腾空阶段信息，帮助区分走和跑。
4. 使用课程学习，不要一开始只采样高速指令。
5. 记录平均速度、速度误差、腾空时间比例和摔倒率。

## 7. 终止与超时

### `terminated=True`

发生以下任一情况时结束本回合：

- 躯干或头部与地面碰撞。
- 躯干倾角超过阈值，初始建议 60°。
- 基座高度低于安全阈值，初始建议 0.18 m。
- 物理状态出现 NaN 或 Inf。

### `truncated=True`

- 达到 `max_episode_steps`。

终止原因写入 `info["termination_reason"]`，便于统计：

```text
fall_contact / excessive_tilt / too_low / non_finite / timeout
```

## 8. Reset 规则

每次 `reset(seed=None, options=None)`：

1. 清空 `MjData`，保持模型参数不变。
2. 根节点放在平地上方约 0.32 m。
3. 初始姿态直立，可随机绕 z 轴旋转 ±π。
4. 8 个关节设置为默认姿态并加入小幅噪声。
5. 基座速度和关节速度置零，可加入极小噪声。
6. 采样新的速度指令。
7. 不进行长时间“稳定化预仿真”，避免训练和评估的初始条件不一致；如需稳定化，单独记录步数。

建议用 `env.reset(seed=...)` 和 `env.action_space.seed(...)` 保证可复现。

## 9. `info` 字典

每次 `step()` 至少返回：

```python
{
    "reward_terms": {
        "tracking": ...,
        "upright": ...,
        "alive": ...,
        "energy": ...,
        "action_rate": ...,
    },
    "forward_speed": ...,
    "lateral_speed": ...,
    "body_height": ...,
    "tilt_angle": ...,
    "command_speed": ...,
    "left_foot_contact": bool,
    "right_foot_contact": bool,
    "termination_reason": None,
}
```

日志字段和奖励分项应使用英文键，便于 TensorBoard 和后续脚本复用。

## 10. 渲染与调试

- 训练时默认不渲染，避免大幅降低吞吐量。
- 评估时提供 `render_mode="human"`，通过 `mujoco.viewer.launch_passive` 或 Gymnasium 封装播放。
- 保存训练视频时使用离屏渲染，而不是在主训练进程中打开窗口。
- 调试工具应能显示关节目标角、实际角、足底接触和机身高度。

## 11. 环境实现结构建议

```text
src/reinforcelearning/envs/
├── __init__.py
├── config.py          # DuckEnvConfig, RewardConfig
├── duck_env.py        # DuckEnv(gym.Env)
└── mujoco_utils.py    # 坐标变换、接触读取等纯函数（可选）
```

`DuckEnvConfig` 至少包含：

```text
model_path
sim_dt / frame_skip / max_episode_steps
action_scale
default_joint_pos
joint_lower / joint_upper
command_range
reward_weights
fall_height / max_tilt
```

## 12. 测试清单

- [ ] `mujoco.MjModel.from_xml_path()` 加载成功且无警告。
- [ ] `reset()` 返回数值有限、形状为 `(34,)`、dtype 为 `float32` 的观测。
- [ ] `step(np.zeros(8, dtype=np.float32))` 返回正确形状和类型。
- [ ] 连续执行 1000 次随机动作，观测、奖励和物理状态均不出现 NaN/Inf。
- [ ] 相同 seed 下 reset 结果可复现。
- [ ] 超时只设置 `truncated=True`，摔倒只设置 `terminated=True`。
- [ ] `info` 中包含全部奖励分项和终止原因。
- [ ] 8 个关节的动作映射到正确的关节目标角，顺序不混乱。
- [ ] 通过 viewer 能看到模型默认站立、腿部和脚掌方向正确。

## 13. 环境规范版本

当前版本：**v0.1**

升级规范时需要同时更新：

1. 本文档中的观测/动作维度。
2. `DuckEnv.step()` 的返回结构。
3. 训练脚本的输入维度。
4. 对应测试用例。
