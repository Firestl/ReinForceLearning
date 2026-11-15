"""Duck-v0 训练与评估环境构造工厂。

环境创建的单一入口，确保训练与回放阶段的包装器堆叠（Monitor / VecEnv / Normalizer）一致。
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

import reinforcelearning.envs  # noqa: F401  导入以触发 Duck-v0 环境注册
from reinforcelearning.envs.config import DuckEnvConfig
from reinforcelearning.training.vec_normalize import DuckVecNormalize

ENV_ID = "Duck-v0"


def make_env(
    *,
    seed: int,
    env_config: DuckEnvConfig | None = None,
    render_mode: str | None = None,
) -> Monitor:
    """构造单个被 Monitor 包装的 Gymnasium 环境实例。"""
    env = gymnasium.make(ENV_ID, config=env_config, render_mode=render_mode)
    env = Monitor(env)
    env.reset(seed=seed)  # 确定初始种子，保障首回合确定性
    return env


def make_vec_env(
    *,
    n_envs: int,
    seed: int,
    env_config: DuckEnvConfig | None = None,
    render_mode: str | None = None,
    normalize: bool = True,
    norm_reward: bool = True,
    gamma: float = 0.99,
) -> VecEnv:
    """构造 DummyVecEnv 向量化环境，可选套载 DuckVecNormalize。"""
    if n_envs < 1:
        raise ValueError(f"n_envs 必须为 >= 1 的整数，实际 {n_envs}")

    thunks: list[Callable[[], gymnasium.Env]] = [
        lambda rank=rank: make_env(seed=seed + rank, env_config=env_config, render_mode=render_mode)
        for rank in range(n_envs)
    ]
    vec_env: VecEnv = DummyVecEnv(thunks)
    if not normalize:
        return vec_env

    config = env_config if env_config is not None else DuckEnvConfig()
    return DuckVecNormalize(
        vec_env,
        command_obs_index=config.obs_dim - 1,
        norm_reward=norm_reward,
        gamma=gamma,
    )
