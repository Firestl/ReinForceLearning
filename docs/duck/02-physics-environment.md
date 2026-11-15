# 物理环境接口规范

本文定义 `DuckEnv` 的 Gymnasium 接口契约。实现代码严格遵守本规范；接口维度与动力学语义变化均需同步测试。

## 1. 控制循环与时间契约

物理步长以 `DuckEnvConfig.sim_dt` 为单一事实源，环境初始化时覆盖 XML 中的 `timestep`。

| 参数 | 契约值 | 物理含义与约束 |
| :--- | :---: | :--- |
| `sim_dt` | $0.002\text{ s}$ | 物理引擎单步积分时间步长 |
| `frame_skip` | $10$ | 单个控制步内的物理子步数（Sub-stepping） |
| `control_dt` | $0.02\text{ s}$ | 控制周期 $50\text{ Hz}$（$= \text{sim\_dt} \times \text{frame\_skip}$） |
| `max_episode_steps` | $1000$ | 单回合最大步数（$= 20\text{ s}$ 仿真时长），超限置 `truncated=True` |
| `render_fps` | $50$ | 渲染帧率，由 $1 / \text{control\_dt}$ 动态推导 |

## 2. 观测空间 Observation

观测空间为本体感知向量 $o_t \in \mathbb{R}^{34}$，`dtype=np.float32`，不包含全局世界坐标 $(x, y)$。

| 索引区间 | 维度 | 物理量 | 参考系 | 计算来源 / 变换公式 | 物理意图 |
| :---: | :---: | :--- | :---: | :--- | :--- |
| `0:3` | 3 | $v_{\text{base}}$ | Body | $\mathbf{R}^T \cdot \dot{p}_{\text{world}}$（世界线速度投影） | 前进/侧移速度感知 |
| `3:6` | 3 | $\omega_{\text{base}}$ | Body | $qvel_{[3:6]}$（MuJoCo freejoint 原生即为 Body 系） | 翻滚/俯仰/偏航角速度 |
| `6:9` | 3 | $g_{\text{proj}}$ | Body | $-\mathbf{R}_{[:, 2]}$（重力方向在身体系的投影） | 重心姿态与倾角感知 |
| `9:17` | 8 | $q_{\text{joint}}$ | Local | $qpos_{[\text{adr}]}$（8 关节位置） | 当前关节构型 |
| `17:25` | 8 | $\dot{q}_{\text{joint}}$ | Local | $qvel_{[\text{adr}]}$（8 关节速度） | 关节运动趋势 |
| `25:33` | 8 | $a_{t-1}$ | — | 上一步执行的归一化动作 | 动作连贯性与平滑评估 |
| `33:34` | 1 | $v_{\text{cmd}}$ | Body-X | 目标前进速度指令 (m/s) | 条件控制目标输入 |

> **关键约束**：除末维 $v_{\text{cmd}}$ 保留物理量纲外，其余 33 维在训练/回放时均通过 `DuckVecNormalize` 标准化。

## 3. 动作空间 Action

动作空间为连续向量 $a \in [-1.0, 1.0]^8$，`dtype=np.float32`。

- 动作语义为**目标关节角偏置**，非力矩控制。
- 映射公式与单腿拓扑限位严格遵循[模型设计规范](01-duck-model.md)的「关节与执行器契约」节。
- 环境拒绝 NaN/Inf 动作，并在写入 `data.ctrl` 前完成关节限位裁剪。

## 4. 速度指令 Command

平地直线任务在回合重置时采样单维前进速度指令 $v_{\text{cmd}}$：

| 训练阶段 | 采样分布 | 训练重点 |
| :--- | :---: | :--- |
| **行走（当前）** | $\text{Uniform}(0.0, 1.0)\text{ m/s}$ | 站立稳定与低速移动 |
| **过渡** | $\text{Uniform}(0.5, 2.0)\text{ m/s}$ | 提高步频与单腿支撑 |
| **奔跑** | $\text{Uniform}(2.0, 3.5)\text{ m/s}$ | 双足腾空与高速平衡 |

*注：$v_{\text{cmd}}$ 排除在 `VecNormalize` 的运行统计之外，避免课程切换时破坏网络对速度尺度的既有表征。*

## 5. 奖励函数 Reward

单步奖励由 9 项加权分项构成，总奖励 $r_t = \sum_k w_k r_k$。权重与形状超参以 `RewardConfig` 为唯一事实源。

$$
\begin{aligned}
\text{tracking} &= \exp\left(-\left(\frac{v_{x}^{\text{body}} - v_{\text{cmd}}}{\sigma_v}\right)^2\right), &
\text{upright} &= \text{clip}\left(-g_{\text{proj}, z},\ 0.0,\ 1.0\right) \\
\text{height} &= \exp\left(-\left(\frac{z_{\text{root}} - h_{\text{target}}}{\sigma_h}\right)^2\right), &
\text{alive} &= 1.0
\end{aligned}
$$

$$
\begin{aligned}
r_t = &+ 1.0 \cdot \text{tracking} + 0.5 \cdot \text{upright} + 0.2 \cdot \text{height} + 0.1 \cdot \text{alive} \\
      &- 0.05 \cdot (v_y^{\text{body}})^2 - 0.05 \cdot (\omega_z^{\text{body}})^2 \\
      &- 0.005 \cdot \|\tau\|_2^2 - 0.01 \cdot \|a_t - a_{t-1}\|_2^2 - 0.01 \cdot \|q^* - q\|_2^2
\end{aligned}
$$

- 速度与高度基准：$\sigma_v = 0.5\text{ m/s}$，$\sigma_h = 0.05\text{ m}$，$h_{\text{target}} = 0.233\text{ m}$。
- 所有分项的加权实际贡献在单步 `info["reward_terms"]` 中完整返回。

## 6. 终止与超时条件

| 终止判定 | 触发条件 | `termination_reason` |
| :--- | :--- | :--- |
| **触地摔倒** | 躯干 (`torso_geom`) 或头部 (`head_geom`) 与地面发生接触 | `fall_contact` |
| **过度倾斜** | 躯干倾角 $\theta_{\text{tilt}} = \arccos(\text{clip}(-g_{\text{proj}, z}, -1, 1)) > 60^\circ$ | `excessive_tilt` |
| **机身过低** | 躯干根节点高度 $z_{\text{root}} < 0.15\text{ m}$ | `too_low` |
| **数值异常** | 状态出现 NaN/Inf 或 MuJoCo 警告计数增加 | `non_finite` |
| **回合截断** | 控制步数达到 `max_episode_steps`（非终止，`truncated=True`） | `timeout` |

## 7. 重置协议 Reset

每次调用 `reset(seed=None)` 执行以下确定性重置序列：

1. 清空 `MjData`，设置根节点位置 $(0, 0, \text{spawn\_height})$，初始朝向随机绕 $z$ 轴旋转 $\pm\pi$。
2. 8 关节赋默认姿态 $q_{\text{default}}$ 并施加 $\pm 0.01\text{ rad}$ 均匀噪声；速度赋微小噪声。
3. 从 `command_range` 重新采样速度指令 $v_{\text{cmd}}$。
4. 调用 `mj_forward` 计算初始快照并返回 `(obs, info)`。
5. 不进行长时间稳定化预仿真，避免训练与评估初始条件不一致。

## 8. 调试与环境配置

- **运行时元数据**：`metadata["render_fps"]` 与控制步长联动，避免离屏录像时间缩放失真。
- **渲染接口**：`render_mode` 支持 `"human"` 与 `"rgb_array"`；`metadata` 含 `render_modes`；`close()` 释放 viewer 与渲染器资源。
- **诊断字段**：`step()` 的 `info` 提供 `forward_speed`、`lateral_speed`、`body_height`、`tilt_angle`、`command_speed`、左右脚接触标志与 `termination_reason`；`reward_terms` 为 9 项加权贡献。
- **环境配置解耦**：
  - `spawn_height`（重置位置 $0.233\text{ m}$）与奖励 $h_{\text{target}}$ 显式解耦，互不干扰。
  - `command_range` 是扩展速度范围的唯一配置点；扩大范围无需修改归一化代码。
  - 资产路径通过 `__file__` 解析，不依赖执行脚本时的工作目录。
