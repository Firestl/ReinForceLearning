# ReinForceLearning

用 MuJoCo + Gymnasium + Stable-Baselines3 学习强化学习的实验仓库。当前主题是双足鸭形机器人的平地直线行走。

## 快速开始

```bash
uv sync
uv run rl-train                 # 默认 100k 步、4 环境、CPU、开启归一化
uv run rl-eval models/duck_ppo --episodes 3
uv run pytest                   # 模型/环境/训练层测试
uv run tensorboard --logdir runs
```

## 文档

- [项目文档索引](docs/README.md)
- [鸭子机器人：设计与实现](docs/duck/README.md)

## 技术栈

MuJoCo · Gymnasium · Stable-Baselines3 (PPO) · PyTorch · uv
