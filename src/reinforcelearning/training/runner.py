"""PPO 训练与评估的编排层：不解析命令行参数，也不构造环境参数。

对应 SB3 的最小用法：``PPO("MlpPolicy", vec_env).learn(...)``。
在这里显式写出超参数，便于初学者逐项对照文档调整。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

import reinforcelearning.envs  # noqa: F401  导入即注册 Duck-v0
from reinforcelearning.training.config import TrainConfig, vecnormalize_file_for
from reinforcelearning.training.env_factory import make_vec_env
from reinforcelearning.training.vec_normalize import DuckVecNormalize


def build_model(vec_env: VecEnv, config: TrainConfig) -> PPO:
    """在已构造的向量化环境上创建 PPO 模型；环境生命周期由调用方管理。"""
    return PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,  # 每个环境每次更新采集的步数
        batch_size=64,
        gamma=config.gamma,
        verbose=1,
        seed=config.seed,
        device=config.device,
        tensorboard_log=str(config.log_dir) if config.log_dir is not None else None,
    )


def train(config: TrainConfig) -> Path:
    """训练 PPO 并保存模型；启用归一化时同时保存统计量。"""
    vec_env = make_vec_env(
        n_envs=config.n_envs,
        seed=config.seed,
        normalize=config.normalize,
        norm_reward=config.norm_reward,
        gamma=config.gamma,
    )
    try:
        model = build_model(vec_env, config)
        model.learn(total_timesteps=config.total_timesteps)
        config.model_file.parent.mkdir(parents=True, exist_ok=True)
        model.save(config.model_file)
        if config.normalize:
            if not isinstance(vec_env, DuckVecNormalize):
                raise RuntimeError("normalize=True 但环境未套 DuckVecNormalize")
            vec_env.save(str(config.vecnormalize_file))
    finally:
        vec_env.close()
    return config.model_file


def evaluate(
    model_path: Path,
    *,
    episodes: int = 3,
    render: bool = False,
    seed: int = 0,
    load_normalization: bool = True,
) -> list[float]:
    """用确定性策略跑若干回合，返回每回合总回报。

    模型旁若存在 ``*_vecnormalize.pkl`` 且 ``load_normalization=True``，会自动
    加载并冻结统计量；否则按原始观测评估。评估循环基于单环境 ``VecEnv``，
    以便复用训练时的包装，并支持 ``--render`` 实时观看。
    """
    if episodes < 1:
        raise ValueError(f"episodes 必须为 >= 1 的整数，实际 {episodes}")

    vec_env = make_vec_env(
        n_envs=1,
        seed=seed,
        render_mode="human" if render else None,
        normalize=False,
    )
    try:
        normalize_file = vecnormalize_file_for(model_path)
        if load_normalization and normalize_file.exists():
            normalized_env = DuckVecNormalize.load(str(normalize_file), vec_env)
            normalized_env.training = False
            normalized_env.norm_reward = False
            vec_env = normalized_env
            print(f"加载归一化统计量: {normalize_file}")

        model = PPO.load(model_path, env=vec_env, device="cpu")
        frame_period = 1.0 / float(vec_env.get_attr("metadata")[0]["render_fps"]) if render else 0.0

        returns: list[float] = []
        obs = vec_env.reset()
        assert isinstance(obs, np.ndarray)
        current_return = 0.0
        steps = 0
        while len(returns) < episodes:
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(actions)
            assert isinstance(obs, np.ndarray)
            current_return += float(rewards[0])
            steps += 1
            if render:
                vec_env.env_method("render")
                time.sleep(frame_period)
            if bool(dones[0]):
                info = infos[0]
                episode_stats = info.get("episode")
                total_return = (
                    float(episode_stats["r"])
                    if isinstance(episode_stats, dict) and "r" in episode_stats
                    else current_return
                )
                returns.append(total_return)
                print(
                    f"[episode {len(returns)}/{episodes}] "
                    f"return={total_return:.2f} steps={steps} "
                    f"reason={info.get('termination_reason')}"
                )
                current_return = 0.0
                steps = 0
        return returns
    finally:
        vec_env.close()
