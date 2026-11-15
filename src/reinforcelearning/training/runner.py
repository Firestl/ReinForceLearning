"""PPO 训练执行与策略评估编排层。

专注训练控制流，不侵入命令行解析与底层环境构造。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

import reinforcelearning.envs  # noqa: F401
from reinforcelearning.training.config import TrainConfig, vecnormalize_file_for
from reinforcelearning.training.env_factory import make_vec_env
from reinforcelearning.training.vec_normalize import DuckVecNormalize


def build_model(vec_env: VecEnv, config: TrainConfig) -> PPO:
    """基于向量化环境初始化 PPO 模型；超参显式声明。"""
    return PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,  # 单个环境每次更新采样的步数
        batch_size=64,
        gamma=config.gamma,
        verbose=1,
        seed=config.seed,
        device=config.device,
        tensorboard_log=str(config.log_dir) if config.log_dir is not None else None,
    )


def train(config: TrainConfig) -> Path:
    """执行 PPO 训练循环并成对持久化模型权重与归一化统计量。"""
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
                raise RuntimeError("normalize=True 但环境未挂载 DuckVecNormalize")
            vec_env.save(str(config.vecnormalize_file))
    finally:
        vec_env.close()
    return config.model_file


def _make_eval_env(
    model_path: Path,
    *,
    seed: int,
    render: bool,
    load_normalization: bool,
) -> VecEnv:
    """装配单进程评估环境；存在统计量时加载并冻结更新。"""
    vec_env = make_vec_env(
        n_envs=1,
        seed=seed,
        render_mode="human" if render else None,
        normalize=False,
    )
    if not load_normalization:
        return vec_env

    normalize_file = vecnormalize_file_for(model_path)
    if not normalize_file.exists():
        return vec_env

    try:
        normalized_env = DuckVecNormalize.load(str(normalize_file), vec_env)
    except Exception:
        vec_env.close()
        raise

    normalized_env.training = False     # 冻结统计量
    normalized_env.norm_reward = False  # 汇报原始真实物理奖励
    print(f"加载归一化统计量: {normalize_file}")
    return normalized_env


def evaluate(
    model_path: Path,
    *,
    episodes: int = 3,
    render: bool = False,
    seed: int = 0,
    load_normalization: bool = True,
) -> list[float]:
    """回放确定性策略并统计各回合总回报。"""
    if episodes < 1:
        raise ValueError(f"episodes 必须为 >= 1 的整数，实际 {episodes}")

    vec_env = _make_eval_env(
        model_path,
        seed=seed,
        render=render,
        load_normalization=load_normalization,
    )
    try:
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
