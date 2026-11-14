"""训练层测试：配置校验与向量化环境契约。

训练层不改变环境语义，因此这里只验证"包装正确"：形状、批处理、种子可复现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reinforcelearning.training.config import TrainConfig
from reinforcelearning.training.env_factory import make_env, make_vec_env

OBS_DIM = 34
ACTION_DIM = 8


def test_model_file_appends_zip_suffix(tmp_path: Path) -> None:
    config = TrainConfig(model_path=tmp_path / "duck")
    assert config.model_file == tmp_path / "duck.zip"


def test_model_file_keeps_existing_zip_suffix(tmp_path: Path) -> None:
    config = TrainConfig(model_path=tmp_path / "duck.zip")
    assert config.model_file == tmp_path / "duck.zip"


@pytest.mark.parametrize(
    "overrides",
    [
        {"total_timesteps": 0},
        {"total_timesteps": -1},
        {"n_envs": 0},
        {"seed": -1},
        {"device": ""},
    ],
)
def test_train_config_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TrainConfig(**overrides)


def test_vec_env_batches_duck_contract() -> None:
    vec_env = make_vec_env(n_envs=2, seed=0)
    try:
        obs = vec_env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2, OBS_DIM)
        assert obs.dtype == np.float32

        actions = np.zeros((2, ACTION_DIM), dtype=np.float32)
        obs, rewards, dones, infos = vec_env.step(actions)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2, OBS_DIM)
        assert rewards.shape == (2,)
        assert dones.dtype == np.bool_
        assert len(infos) == 2
    finally:
        vec_env.close()


def test_make_env_is_reproducible_for_same_seed() -> None:
    env_a = make_env(seed=123)
    env_b = make_env(seed=123)
    try:
        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()
        np.testing.assert_allclose(obs_a, obs_b)
    finally:
        env_a.close()
        env_b.close()
