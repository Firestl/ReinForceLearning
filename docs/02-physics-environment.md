# 物理环境接口设计

本文定义第一版 `DuckEnv` 的 Gymnasium 接口。实现代码应严格遵守本规范；任何维度或语义变化都需要同步更新文档和测试。

## 1. 环境职责

`DuckEnv` 负责加载 `assets/duck.xml`、维护 `MjModel`/`MjData`、映射动作、按固定频率推进仿真、构造观测、计算奖励、判断终止和重置，并提供固定形状与数值范围的接口。环境不实现算法，训练时不打开可视化窗口。

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

动作空间顺序与上述关节顺序一致；`q_default`/`action_scale` 映射见模型文档 §6。

MuJoCo 索引约定（实现时按名称查，不硬编码）：

- `freejoint` 占 `qpos[0:7]`、`qvel[0:6]`。
- 8 个主动关节占 `qpos[7:15]`、`qvel[6:14]`，顺序与上表一致。

设计注意事项：

- 不把世界坐标 `x, y` 放入观测，减少对起点和地图位置的过拟合。
- 初始阶段不使用历史帧堆叠；后续加入足底接触、步态相位或转向指令时，应升维并更新本章节。
- `base_lin_vel`/`base_ang_vel` 从 freejoint 的 `qvel` 转换到身体坐标系；实现时用已知姿态测试符号和参考系。
- 速度、关节角等用 `VecNormalize` 归一化；command 的归一化见 §5。

`observation_space` 建议：

```python
spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(34,),
    dtype=np.float32,
)
```

实际实现时应逐项检查 `np.isfinite(obs).all()`，避免接触爆炸导致 NaN 传入网络。`Box(-inf, inf)` 可能触发 `check_env` 的 unbounded 警告；配合 `VecNormalize` 可接受。若要去掉警告，可按实际范围给有限上下界。

## 4. 动作空间 Action

```python
spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
```

- 动作是目标关节角，不是力矩，也不是关节速度。
- `q_default`/`action_scale` 来自模型文档 §6，动作顺序与 §3 固定关节顺序一致；actuator 定义顺序也必须一致。
- 映射后的目标角必须裁剪到关节限位内并保留安全边界，再通过 `data.ctrl` 交给位置执行器。
- 第一版不加入随机动作噪声；探索由 PPO 的高斯策略负责。

## 5. 速度指令 Command

第一版为平地直线任务，每回合采样一个目标前进速度。初始阶段只采样行走速度，后续用课程学习逐步扩大范围：

| 阶段 | `v_x_cmd` 范围 | 目标 |
| --- | --- | --- |
| 行走（初始） | `Uniform(0.0, 1.0) m/s` | 站立和低速行走 |
| 过渡 | `Uniform(0.5, 2.0) m/s` | 提高步频和步幅 |
| 奔跑 | `Uniform(2.0, 3.5) m/s` | 高速、腾空阶段 |

走/跑不是仅按速度硬切分，最终以足底接触和腾空比例判断。转向暂不实现：没有髋 yaw 关节，且指令只有一维。

训练时建议对 command 使用固定范围缩放（例如除以最大指令速度），或把它排除在 `VecNormalize` 之外；否则课程切换会改变归一化统计量。

## 6. 奖励函数 Reward

奖励函数应该由多个可解释项组成，并在 `info` 中返回各分项，不能只返回一个总数。

第一版建议形式：

```text
speed_error   = v_x_body - v_x_cmd
tracking      = exp(-(speed_error / 0.5)^2)
height_reward = exp(-((body_height - h_target) / 0.05)^2)

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
- `tilt_angle = arccos(clip(-projected_gravity[2], -1.0, 1.0))`。
- `body_height` 取 `root` 的 z 坐标；`height_reward = exp(-((body_height - h_target) / 0.05)^2)`，`h_target` 取默认站立时的 root 高度约 `0.233 m`。
- 若 `upright` 与 `height_reward` 相关性过强，训练时可降低 `height_reward` 权重或只保留一个。
- 摔倒惩罚需要单独实验，建议先不要给太大惩罚。
- 所有奖励权重集中放在 `RewardConfig`，禁止散落在 `step()` 中。
- 上述权重下单步奖励量级约 1~2；1000 步回合的未折扣回报可达约 1000~1800，训练时考虑 reward normalization 或整体缩放。

### 奔跑阶段的变化

1. 从行走范围 `0.0~1.0 m/s` 开始，按课程逐步扩大 `v_x_cmd` 到约 `2.0~3.5 m/s`。
2. 降低能量惩罚，允许更大的力矩和动作幅度。
3. 加入足底接触/腾空阶段信息，并记录平均速度、误差、腾空比例和摔倒率。

## 7. 终止与超时

### `terminated=True`

发生以下任一情况时结束本回合：

- 躯干或头部与地面碰撞。
- 躯干倾角超过阈值，初始建议 60°。
- 基座高度低于安全阈值，初始建议 0.15 m（约比默认站立高度低 0.08 m）。
- 物理状态出现 NaN 或 Inf。

### `truncated=True`

- 达到 `max_episode_steps`。

终止原因写入 `info["termination_reason"]`，便于统计：

```text
fall_contact / excessive_tilt / too_low / non_finite / timeout
```

## 8. Reset 规则

每次 `reset(seed=None, options=None)` 返回 `(obs, info)`：

1. 清空 `MjData`，模型参数保持不变。
2. 根节点放在平地上方约 0.233 m（默认姿态下脚底接近 z=0），初始姿态直立，可随机绕 z 轴旋转 ±π。
3. 8 个关节设为默认姿态并加入小幅噪声。
4. 基座速度和关节速度置零，可加入极小噪声；采样新的速度指令。
5. 不做长时间“稳定化预仿真”，避免训练和评估初始条件不一致。

用 `env.reset(seed=...)` 和 `env.action_space.seed(...)` 保证可复现。

## 9. `info` 字典

每次 `step()` 至少返回：

```python
{
    "reward_terms": {
        "tracking": ...,
        "upright": ...,
        "height_reward": ...,
        "alive": ...,
        "lateral": ...,
        "yaw": ...,
        "energy": ...,
        "action_rate": ...,
        "joint_tracking": ...,
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

`reward_terms` 中每项是乘以权重后的实际贡献，惩罚项为负；9 项之和等于 `reward`。其中 `lateral`/`yaw` 分别对应 `v_y^2`/`omega_z^2` 惩罚，`height_reward` 对应高度项。日志字段和奖励分项应使用英文键，便于 TensorBoard 和后续脚本复用。

## 10. 渲染与调试

- 训练默认不渲染；评估时用 `render_mode="human"`，或通过 `mujoco.viewer.launch_passive` 调试。
- 视频用 `render_mode="rgb_array"` 离屏渲染保存。
- `metadata` 至少包含 `render_modes=["human", "rgb_array"]` 和 `render_fps=50`。
- 实现 `close()` 释放 viewer 和渲染资源。
- 调试工具显示关节目标角/实际角、足底接触和机身高度。

## 11. 配置项

`DuckEnvConfig` 至少包含：

```text
model_path
obs_dim / action_dim
sim_dt / frame_skip / max_episode_steps
action_scale
default_joint_pos
joint_lower / joint_upper
command_range
reward_weights / height_target
fall_height / max_tilt
```

`model_path` 建议通过 `importlib.resources.as_file` 或基于 `__file__` 解析，避免依赖当前工作目录；`MjModel.from_xml_path` 需要真实文件路径。

## 12. 测试清单

- [ ] `mujoco.MjModel.from_xml_path()` 加载成功且无警告。
- [ ] `reset()` 返回 `(obs, info)`，`obs` 为有限值、形状 `(34,)`、dtype `float32`。
- [ ] `step(np.zeros(8, dtype=np.float32))` 返回 `(obs, reward, terminated, truncated, info)`，形状和类型正确。
- [ ] `gymnasium.utils.env_checker.check_env(env, skip_render_check=True)` 通过。
- [ ] 跨回合累计执行 1000 次随机动作（`terminated`/`truncated` 时 `reset()`），观测、奖励和物理状态无 NaN/Inf。
- [ ] 用已知 `qpos`/`qvel` 验证 34 维观测的索引布局和 body-frame 速度。
- [ ] 相同 seed 下 reset 结果可复现。
- [ ] 超时只设置 `truncated=True`，摔倒只设置 `terminated=True`。
- [ ] 超时测试通过临时调小 `max_episode_steps` 或稳定站姿触发，避免随机动作提前摔倒。
- [ ] `info` 包含全部 9 个奖励分项和终止原因。
- [ ] 8 个动作映射到正确的关节目标角，顺序不混乱，并裁剪到关节限位内。
- [ ] 头部/躯干触地时 `terminated=True`，`termination_reason="fall_contact"`。
- [ ] 鸭子-鸭子无接触，脚/躯干/头部-地面接触可被检测。
- [ ] `render_mode="rgb_array"` 能返回帧，`close()` 不报错。
- [ ] viewer 中默认站姿、头部、腿部和脚掌方向正确。

规范升级时同步更新观测/动作维度、`step()` 返回结构、训练输入维度和对应测试。
