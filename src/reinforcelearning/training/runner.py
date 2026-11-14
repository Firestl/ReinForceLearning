"""PPO 训练与评估的编排层：不解析命令行参数，也不构造环境参数。

对应 SB3 的最小用法：``PPO("MlpPolicy", vec_env).learn(...)``。
在这里显式写出超参数，便于初学者逐项对照文档调整。
"""

from __future__ import annotations

import time
from pathlib import Path

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

import reinforcelearning.envs  # noqa: F401  导入即注册 Duck-v0
from reinforcelearning.training.config import TrainConfig
from reinforcelearning.training.env_factory import ENV_ID, make_vec_env


def build_model(vec_env: VecEnv, config: TrainConfig) -> PPO:
    """在已构造的向量化环境上创建 PPO 模型；环境生命周期由调用方管理。"""
    return PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,  # 每个环境每次更新采集的步数
        batch_size=64,
        gamma=0.99,
        verbose=1,
        seed=config.seed,
        device=config.device,
        tensorboard_log=str(config.log_dir) if config.log_dir is not None else None,
    )


def train(config: TrainConfig) -> Path:
    """训练 PPO 并保存模型，返回模型文件路径。"""
    vec_env = make_vec_env(n_envs=config.n_envs, seed=config.seed)
    try:
        model = build_model(vec_env, config)
        model.learn(total_timesteps=config.total_timesteps)
        config.model_file.parent.mkdir(parents=True, exist_ok=True)
        model.save(config.model_file)
    finally:
        vec_env.close()
    return config.model_file


def evaluate(
    model_path: Path,
    *,
    episodes: int = 3,
    render: bool = False,
    seed: int = 0,
) -> list[float]:
    """用确定性策略跑若干回合，返回每回合总回报。

    评估使用单环境循环，便于逐回合统计；``PPO.load`` 会把传入的 Gymnasium
    环境自动包装成 ``Monitor`` + ``DummyVecEnv``，但 ``predict`` 仍接受单条观测。
    """
    if episodes < 1:
        raise ValueError(f"episodes 必须为 >= 1 的整数，实际 {episodes}")

    env = gymnasium.make(ENV_ID, render_mode="human" if render else None)
    try:
        model = PPO.load(model_path, env=env, device="cpu")
        frame_period = 1.0 / float(env.unwrapped.metadata["render_fps"])
        returns: list[float] = []
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            total_reward = 0.0
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                if render:
                    env.render()
                    time.sleep(frame_period)
                if terminated or truncated:
                    break
            returns.append(total_reward)
            print(
                f"[episode {episode + 1}/{episodes}] "
                f"return={total_reward:.2f} steps={steps} "
                f"reason={info.get('termination_reason')}"
            )
        return returns
    finally:
        env.close()
