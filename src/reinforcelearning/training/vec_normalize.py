"""Duck 专用向量化归一化包装器。

保留 command_speed 的物理量纲，仅对动力学感知观测进行在线统计与标准化。
若 command 参与动态均值方差统计，随着课程学习扩展指令范围，网络对已有指令的
语义表征会发生灾难性漂移。
"""

from __future__ import annotations

import gymnasium
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecNormalize


class DuckVecNormalize(VecNormalize):
    """仅对 Box 观测中非 command 维度执行标准化的包装器。"""

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
            raise TypeError("DuckVecNormalize 仅支持 Box 观测空间")

        shape = venv.observation_space.shape
        assert shape is not None
        obs_dim = int(shape[0])
        index = command_obs_index if command_obs_index >= 0 else obs_dim + command_obs_index
        if not 0 <= index < obs_dim:
            raise ValueError(f"command_obs_index={command_obs_index} 超出观测维度 {obs_dim}")
        self.command_obs_index = index

    @staticmethod
    def load(load_path: str, venv: VecEnv) -> DuckVecNormalize:
        """从持久化文件反序列化，并严格校验类型契约。"""
        loaded = VecNormalize.load(load_path, venv)
        if not isinstance(loaded, DuckVecNormalize):
            raise TypeError(f"{load_path} 非 DuckVecNormalize 统计量文件")
        return loaded

    def normalize_obs(
        self, obs: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray | dict[str, np.ndarray]:
        """对输入观测执行标准化，并在末尾还原原始 command 维度。"""
        if not isinstance(obs, np.ndarray):
            raise TypeError("DuckVecNormalize 仅支持 ndarray 观测")
        normalized = super().normalize_obs(obs)
        assert isinstance(normalized, np.ndarray)
        normalized[..., self.command_obs_index] = obs[..., self.command_obs_index]
        return normalized

    def unnormalize_obs(
        self, obs: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray | dict[str, np.ndarray]:
        """还原标准化观测，保持 command 维度不变。"""
        if not isinstance(obs, np.ndarray):
            raise TypeError("DuckVecNormalize 仅支持 ndarray 观测")
        unnormalized = super().unnormalize_obs(obs)
        assert isinstance(unnormalized, np.ndarray)
        unnormalized[..., self.command_obs_index] = obs[..., self.command_obs_index]
        return unnormalized
