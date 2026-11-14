# 鸭形机器人强化学习：设计与实现文档

本目录记录“训练鸭形机器人学会行走和奔跑”项目的设计决策、模型规范、环境接口与训练实现。当前已完成模型、环境与最小 PPO 训练闭环，进入训练调优阶段。

> 状态：M1（模型可加载）、M2（环境可运行）、M3（PPO 冒烟训练）已实现；M4 进行中，`VecNormalize` 已完成并接入训练/评估。`tests/` 覆盖模型、环境与训练层。步态质量待后续调优。最后更新：2026-11-14。

## 1. 本阶段目标

建立一套可复现、可测试的最小平坦地面物理环境：

1. 用 MuJoCo 定义一只双足、8 个主动关节的鸭形机器人。
2. 提供符合 Gymnasium 接口的观测、动作、奖励、终止和重置。
3. 通过无训练策略完成模型加载、随机动作仿真和数值稳定性测试。
4. 为后续 PPO 训练提供固定接口，并已接入最小 PPO 训练闭环。

暂不包含：复杂地形、视觉输入、sim-to-real、真实电机建模和最终步态调优。

## 2. 关键设计决策

| 项目 | 初始选择 | 说明 |
| --- | --- | --- |
| 仿真器 | MuJoCo | 接触仿真稳定，Python API 简单 |
| 环境接口 | Gymnasium | 可直接接入 Stable-Baselines3 |
| 训练算法 | PPO | 连续控制的可靠基线；后续可比较 SAC |
| 模型格式 | MJCF (`.xml`) | MuJoCo 原生格式，便于分层定义 |
| 鸭子形态 | 双足、8 个主动关节 | 每条腿：髋外摆、髋俯仰、膝俯仰、踝俯仰 |
| 初始训练任务 | 平地直线行走 | 一维向前速度指令；转向后续加入 |

坐标系、步长、尺寸、关节限位和接口维度见后续文档。表中的初始值需通过模型测试和训练结果迭代。

## 3. 代码结构

```text
src/reinforcelearning/
├── assets/
│   └── duck.xml              # MJCF 模型
├── envs/
│   ├── __init__.py
│   ├── config.py             # 环境与模型参数
│   ├── duck_env.py           # Gymnasium 环境
│   └── model_contract.py     # 模型契约校验
├── training/
│   ├── config.py             # 训练实验参数
│   ├── env_factory.py        # 向量化环境构造
│   ├── vec_normalize.py      # command 维不归一化的 VecNormalize
│   └── runner.py             # PPO 训练/评估编排
└── cli.py                    # rl-train / rl-eval 入口
tests/
├── test_duck_model.py
├── test_duck_env.py
├── test_training.py
└── test_vec_normalize.py
```

## 4. 文档索引

- [鸭子模型设计](01-duck-model.md)：坐标系、尺寸、关节、执行器、碰撞体和 MJCF 结构。
- [物理环境接口](02-physics-environment.md)：观测空间、动作空间、奖励、终止条件、控制循环与渲染调试。
- [最小 PPO 训练与归一化](04-training.md)：训练层结构、运行命令、`VecNormalize` 设计与验收记录。

历史文档：[已归档的实施计划](archive/03-implementation-plan.md)（阶段 A-C 已完成，仅作记录）。

## 5. 里程碑

M0 文档就绪 → M1 模型可加载 → M2 环境可运行 → M3 PPO 冒烟训练（已完成）→ M4 训练调优（进行中：`VecNormalize` 已完成；奖励分项日志、速度课程待做）。验收记录见[已归档的实施计划](archive/03-implementation-plan.md)与[最小 PPO 训练与归一化](04-training.md)。

## 6. 维护约定

- 模型尺寸、关节名、观测/动作维度是代码与测试的“契约”，改动时同步更新文档和测试。
- 修改模型或接口契约时，同步更新顶部状态行的日期。
- 奖励权重必须配合监控指标解释，不能只看总奖励判断策略好坏。
