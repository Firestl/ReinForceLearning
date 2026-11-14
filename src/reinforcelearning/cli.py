"""命令行入口：``rl-train`` 与 ``rl-eval``。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from reinforcelearning.training.config import (
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_PATH,
    TrainConfig,
)
from reinforcelearning.training.runner import evaluate, train


def train_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rl-train", description="用 PPO 训练 Duck-v0")
    parser.add_argument(
        "--total-timesteps", type=int, default=100_000, help="总环境步数（默认 100000）"
    )
    parser.add_argument("--n-envs", type=int, default=4, help="并行环境数（默认 4）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="模型保存路径（自动补 .zip）"
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="TensorBoard 目录")
    parser.add_argument("--no-tensorboard", action="store_true", help="关闭 TensorBoard 日志")
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch 设备（默认 cpu；MLP 策略在 GPU 上通常更慢）",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="观测/奖励归一化（默认开启）",
    )
    parser.add_argument(
        "--norm-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="归一化奖励（默认开启；仅在 --normalize 时生效）",
    )
    args = parser.parse_args(argv)

    config = TrainConfig(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        model_path=args.model_path,
        log_dir=None if args.no_tensorboard else args.log_dir,
        device=args.device,
        normalize=args.normalize,
        norm_reward=args.norm_reward,
    )
    model_file = train(config)
    print(f"\n模型已保存: {model_file}")
    return 0


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rl-eval", description="回放训练好的 PPO 策略")
    parser.add_argument("model_path", type=Path, help="模型路径（可省略 .zip）")
    parser.add_argument("--episodes", type=int, default=3, help="回放回合数（默认 3）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument("--render", action="store_true", help="打开 MuJoCo 窗口实时回放")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="忽略模型旁的 VecNormalize 统计量，用原始观测评估",
    )
    args = parser.parse_args(argv)

    returns = evaluate(
        args.model_path,
        episodes=args.episodes,
        render=args.render,
        seed=args.seed,
        load_normalization=not args.raw,
    )
    print(f"\n平均回报: {sum(returns) / len(returns):.2f}")
    return 0
