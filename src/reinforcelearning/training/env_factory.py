"""构造 ``Duck-v0`` 的训练/评估环境：环境创建的唯一入口。

训练与评估必须共用同一套包装，否则观测和终止语义会不一致。
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

import reinforcelearning.envs  # noqa: F401  导入即注册 Duck-v0
from reinforcelearning.envs.config import DuckEnvConfig
from reinforcelearning.training.vec_normalize import DuckVecNormalize

ENV_ID = "Duck-v0"


def make_env(
    *,
    seed: int,
    env_config: DuckEnvConfig | None = None,
    render_mode: str | None = None,
) -> Monitor:
    """创建单个带回合统计的 Gymnasium 环境。"""
    env = gymnasium.make(ENV_ID, config=env_config, render_mode=render_mode)
    env = Monitor(env)
    # VecEnv 之后还会 reset，但这里的种子保证首个回合也可复现
    env.reset(seed=seed)
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
    """创建 ``n_envs`` 个并行环境；``normalize=True`` 时套上 ``DuckVecNormalize``。

    第一版用 ``DummyVecEnv``（同进程顺序执行）：MuJoCo 单步开销小，
    多进程带来的序列化与调试成本不划算。需要吞吐时再换 ``SubprocVecEnv``。
    """
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
