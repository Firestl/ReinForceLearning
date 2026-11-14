"""Duck 专用的 ``VecNormalize``：归一化除速度指令外的所有观测维度。

``command_speed`` 的物理范围会随课程学习变化，若参与运行统计，策略对同一
指令的理解会在训练中途漂移。这里保留该维原始值，其余维度交给 SB3 的
``VecNormalize`` 在线统计。
"""

from __future__ import annotations

import gymnasium
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecNormalize


class DuckVecNormalize(VecNormalize):
    """只归一化 Box 观测中非 command 维度的 ``VecNormalize``。"""

    def __init__(
        self,
        venv: VecEnv,
        *,
        command_obs_index: int = -1,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(
            venv,
            training=training,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=gamma,
            epsilon=epsilon,
        )
        if not isinstance(venv.observation_space, gymnasium.spaces.Box):
            raise TypeError("DuckVecNormalize 仅支持 Box 观测")
        shape = venv.observation_space.shape
        assert shape is not None
        obs_dim = int(shape[0])
        index = command_obs_index if command_obs_index >= 0 else obs_dim + command_obs_index
        if not 0 <= index < obs_dim:
            raise ValueError(f"command_obs_index={command_obs_index} 超出观测维度 {obs_dim}")
        self.command_obs_index = index

    @staticmethod
    def load(load_path: str, venv: VecEnv) -> DuckVecNormalize:
        """加载统计量，并校验文件确实由 ``DuckVecNormalize`` 保存。"""
        loaded = VecNormalize.load(load_path, venv)
        if not isinstance(loaded, DuckVecNormalize):
            raise TypeError(f"{load_path} 不是 DuckVecNormalize 统计量文件")
        return loaded

    def normalize_obs(
        self, obs: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray | dict[str, np.ndarray]:
        if not isinstance(obs, np.ndarray):
            raise TypeError("DuckVecNormalize 仅支持 ndarray 观测")
        normalized = super().normalize_obs(obs)
        assert isinstance(normalized, np.ndarray)
        normalized[..., self.command_obs_index] = obs[..., self.command_obs_index]
        return normalized

    def unnormalize_obs(
        self, obs: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray | dict[str, np.ndarray]:
        if not isinstance(obs, np.ndarray):
            raise TypeError("DuckVecNormalize 仅支持 ndarray 观测")
        unnormalized = super().unnormalize_obs(obs)
        assert isinstance(unnormalized, np.ndarray)
        unnormalized[..., self.command_obs_index] = obs[..., self.command_obs_index]
        return unnormalized
