# 鸭形机器人强化学习

本项目以 MuJoCo 为动力学仿真底座，基于 Gymnasium 与 Stable-Baselines3 (PPO) 训练双足鸭形机器人在平地直线行走。当前已打通**模型规范 → 仿真环境 → 向量化/归一化 → PPO 训练与回放**全链路。

> **当前状态**：M1（模型契约）、M2（物理环境）、M3（PPO 训练闭环）已完成；M4（训练调优，含特化 `VecNormalize`）进行中。最后更新：2026-11-15。

## 技术栈与选型理由

| 选型 | 核心优势 |
| :--- | :--- |
| **MuJoCo** | 接触动力学求解快速稳定，原生 Python API 丰富，支持确定性碰撞检测 |
| **Gymnasium** | 标准强化学习环境契约，可无缝对接主流算法库 |
| **Stable-Baselines3 (PPO)** | 连续控制任务的工业级基线算法，兼顾稳定性与样本效率 |

## 文档地图

- [模型设计规范](01-duck-model.md)：坐标系统、几何与质量参数、关节与执行器契约、MJCF 约定。
- [物理环境接口规范](02-physics-environment.md)：控制周期、观测空间（34 维）、动作空间、速度指令、奖励函数与终止准则。
- [训练与归一化规范](04-training.md)：分层架构、运行 CLI 命令、VecNormalize 策略、吞吐基准与验收记录。
- [已归档的实施计划](archive/03-implementation-plan.md)：M1/M2 阶段历史交付记录。

## 里程碑进展

$$\text{M0 文档规范} \longrightarrow \text{M1 模型契约} \longrightarrow \text{M2 仿真环境} \longrightarrow \text{M3 训练闭环} \longrightarrow \mathbf{M4\ 训练调优（进行中）}$$

## 维护约定

1. **唯一事实源 (SSOT)**：物理参数以 `DuckEnvConfig`、`RewardConfig` 与 `duck.xml` 为准，文档提供物理意义与数学推导。
2. **时态分离**：已完成的里程碑任务移入 `archive/` 归档，规范文档只描述当前系统生效的契约。
3. **数据一致性**：实验指标与基准只写在 `04-training.md`，不在其他文档重复复制。
