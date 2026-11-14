# 鸭子模型与物理环境实施计划

> **归档状态（2026-11-14）**：阶段 A-C 已全部完成，本文作为历史记录保留，不再维护。
> 当前有效规范见 [鸭子模型设计](../01-duck-model.md) 与 [物理环境接口](../02-physics-environment.md)。

本文把“准备建立鸭子和物理环境”拆分为可执行任务。当前阶段只做模型和环境，不训练最终策略。

## 1. 阶段 A：环境与项目脚手架

- [x] 确认 Python 3.12 与依赖安装完成。
- [x] 创建 `src/reinforcelearning/assets/`、`src/reinforcelearning/envs/` 和 `tests/`。
- [x] 确认 `assets/duck.xml` 被打包进 wheel/editable 安装，运行时用 `importlib.resources` 或 `__file__` 解析。（采用 `__file__` 解析）
- [x] 建立 `DuckEnvConfig` 与 `RewardConfig`，配置值集中管理；构造期校验尺寸、限位、时间参数和奖励权重，`spawn_height` 与奖励的 `height_target` 独立配置。
- [x] 运行 `uv sync`；如未安装 pytest，执行 `uv add --dev pytest`；确认导入路径正常。（pytest 已在 dev 依赖组）

验收：能 import `mujoco`、`gymnasium`，并运行一个空测试。（已通过脚本验证 import 与模型加载）

## 2. 阶段 B：鸭子 MJCF 模型

- [x] 按 `01-duck-model.md` 建立躯干、头部、双腿和脚掌。
- [x] 添加 8 个主动关节及对应位置执行器，并确认执行器定义顺序与关节/动作顺序一致。
- [x] 设置默认站立姿态、关节限位、执行器力限。
- [x] 添加地面、摩擦参数、足底/头部碰撞体，并配置 `contype`/`conaffinity` 关闭鸭子自碰撞。
- [x] 给根节点、关键几何体和执行器命名。
- [x] 使用 viewer 检查默认姿态、朝向和左右对称性。（离屏渲染多机位帧目视检查）
- [x] 检查默认类生效：主动关节的 `damping`/`armature`/`frictionloss` 正确，`freejoint` 保持 0；执行器 `ctrlrange` 位于关节限位内且覆盖动作目标范围，避免 MuJoCo 二次裁剪改变动作映射。
- [x] 校验模型契约：命名实体可解析，关节顺序/类型/轴与配置一致；执行器必须是 joint transmission 上的 position actuator（fixed gain + affine bias），不能误用 motor。
- [x] 检查接触掩码：鸭子-地面可接触，鸭子-鸭子无接触，头部/躯干触地可被识别。
- [x] 编写 `test_duck_model.py`：加载成功、状态有限、默认姿态下 root≈0.233 m 且脚底最低点 `|z| <= 0.005 m`，执行器/关节顺序和接触掩码正确。

验收：M1 完成，默认姿态脚底接地（实测 `z = +0.001 m`），模型可稳定仿真至少 2 秒（关节误差 ≤0.005 rad）。

## 3. 阶段 C：Gymnasium 环境

- [x] 实现 `DuckEnv(gym.Env)` 的 `reset()`、`step()`、`render()`、`close()`；`reset()` 返回 `(obs, info)`，`step()` 返回 5 元组，且每步只计算一次状态与接触快照供观测、奖励和终止共享。
- [x] 实现动作到 `data.ctrl` 的映射和裁剪，拒绝维度错误或含 NaN/Inf 的动作。
- [x] 实现 34 维观测构造：线速度由世界系经 `R^T` 转到机体系，free joint 角速度直接读取 `qvel[3:6]`（本身已在机体系），并用已知 `qpos`/`qvel` 验证索引布局。
- [x] 以 `config.sim_dt` 作为物理步长唯一来源写入 `model.opt.timestep`，`render_fps` 按 `1 / (sim_dt × frame_skip)` 推导。
- [x] 实现速度指令采样和奖励分项。
- [x] 实现倾角/高度/碰撞/数值异常的终止判断。（NaN/Inf 检测需监视 MuJoCo 警告计数：`mj_step` 会在数值异常时自动重置 `qpos`，仅检查有限值会漏报）
- [x] 实现超时截断和包含全部奖励分项的 `info` 字典。
- [x] 编写 `test_duck_env.py`：覆盖 reset/step 接口、34 维观测布局与机体系速度、动作映射与裁剪、奖励分项、终止/截断、渲染与 `check_env`。（1000 步随机动作无 NaN 由测试覆盖）
- [x] 注册环境 ID，例如 `Duck-v0`，供训练脚本使用。

验收：M2 完成，`check_env` 通过（仅余 `Box(-inf, inf)` 的预期警告），跨回合累计 1000 步随机动作无 NaN。
