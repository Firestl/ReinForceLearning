# 项目文档

本仓库的文档按主题分目录保存。

## 鸭子机器人强化学习

- [鸭子文档索引](duck/README.md)：设计决策、里程碑与维护约定。
- [鸭子模型设计](duck/01-duck-model.md)：坐标系、尺寸、关节、执行器与 MJCF 结构。
- [物理环境接口](duck/02-physics-environment.md)：观测、动作、奖励、终止、控制循环与渲染。
- [最小 PPO 训练](duck/04-training.md)：训练层结构、运行命令、验收记录与已知局限。
- [实施计划（已归档）](duck/archive/03-implementation-plan.md)：M1/M2 的历史实施与验收记录。

## 维护约定

- 新主题在 `docs/` 下新建独立目录，并在此处登记。
- 已完成的阶段性计划移入对应主题目录的 `archive/`，不再作为活跃文档维护。
