"""``DuckVecNormalize`` 契约测试：command 维例外、保存/加载与类型守卫。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecNormalize

from reinforcelearning.training.env_factory import make_vec_env
from reinforcelearning.training.vec_normalize import DuckVecNormalize

OBS_DIM = 34
ACTION_DIM = 8


def test_normalize_obs_standardizes_non_command_dims() -> None:
    env = make_vec_env(n_envs=1, seed=0)
    assert isinstance(env, DuckVecNormalize)
    try:
        obs_rms = env.obs_rms
        assert isinstance(obs_rms, RunningMeanStd)
        # 注入非平凡统计量，避免首帧 count=1 时归一化退化为 0
        obs_rms.mean = np.linspace(-1.0, 1.0, OBS_DIM)
        obs_rms.var = np.linspace(0.5, 2.0, OBS_DIM)

        raw = np.linspace(-1.0, 1.0, OBS_DIM, dtype=np.float32)[None, :]
        normalized = env.normalize_obs(raw)
        assert isinstance(normalized, np.ndarray)
        expected = (raw - obs_rms.mean) / np.sqrt(obs_rms.var + env.epsilon)

        # 值域在 clip_obs=10 以内，可直接对照 SB3 的归一化公式
        np.testing.assert_allclose(normalized[:, :-1], expected[:, :-1], rtol=1e-5)
        np.testing.assert_allclose(normalized[:, -1], raw[:, -1])

        restored = env.unnormalize_obs(normalized)
        assert isinstance(restored, np.ndarray)
        np.testing.assert_allclose(restored[:, :-1], raw[:, :-1], rtol=1e-5)
        np.testing.assert_allclose(restored[:, -1], raw[:, -1])
    finally:
        env.close()


def test_normalize_preserves_command_dimension_through_reset_and_step() -> None:
    raw_env = make_vec_env(n_envs=2, seed=7, normalize=False)
    normalized_env = make_vec_env(n_envs=2, seed=7)
    assert isinstance(normalized_env, DuckVecNormalize)
    try:
        raw_obs = raw_env.reset()
        normalized_obs = normalized_env.reset()
        assert isinstance(raw_obs, np.ndarray)
        assert isinstance(normalized_obs, np.ndarray)
        np.testing.assert_allclose(normalized_obs[:, -1], raw_obs[:, -1])

        actions = np.zeros((2, ACTION_DIM), dtype=np.float32)
        raw_obs, _, _, _ = raw_env.step(actions)
        normalized_obs, _, _, _ = normalized_env.step(actions)
        assert isinstance(raw_obs, np.ndarray)
        assert isinstance(normalized_obs, np.ndarray)
        np.testing.assert_allclose(normalized_obs[:, -1], raw_obs[:, -1])
    finally:
        raw_env.close()
        normalized_env.close()


def test_save_load_roundtrip_restores_statistics_and_command_rule(tmp_path: Path) -> None:
    stats_path = tmp_path / "duck_ppo_vecnormalize.pkl"
    source_env = make_vec_env(n_envs=1, seed=0)
    assert isinstance(source_env, DuckVecNormalize)
    try:
        source_env.reset()
        source_env.step(np.zeros((1, ACTION_DIM), dtype=np.float32))
        source_env.save(str(stats_path))
        source_rms = source_env.obs_rms
        assert isinstance(source_rms, RunningMeanStd)
        expected_mean = source_rms.mean.copy()
    finally:
        source_env.close()

    raw_env = make_vec_env(n_envs=1, seed=0, normalize=False)
    try:
        loaded_env = DuckVecNormalize.load(str(stats_path), raw_env)
        assert loaded_env.command_obs_index == OBS_DIM - 1
        loaded_rms = loaded_env.obs_rms
        assert isinstance(loaded_rms, RunningMeanStd)
        np.testing.assert_allclose(loaded_rms.mean, expected_mean)

        probe = np.zeros((1, OBS_DIM), dtype=np.float32)
        probe[0, -1] = 0.42
        normalized = loaded_env.normalize_obs(probe)
        assert isinstance(normalized, np.ndarray)
        assert normalized[0, -1] == pytest.approx(0.42)
    finally:
        raw_env.close()


def test_load_rejects_plain_vecnormalize_statistics(tmp_path: Path) -> None:
    stats_path = tmp_path / "plain_vecnormalize.pkl"
    source_env = make_vec_env(n_envs=1, seed=0, normalize=False)
    try:
        VecNormalize(source_env).save(str(stats_path))
    finally:
        source_env.close()

    raw_env = make_vec_env(n_envs=1, seed=0, normalize=False)
    try:
        with pytest.raises(TypeError):
            DuckVecNormalize.load(str(stats_path), raw_env)
    finally:
        raw_env.close()


@pytest.mark.parametrize("command_obs_index", [OBS_DIM, -OBS_DIM - 1])
def test_init_rejects_out_of_range_command_index(command_obs_index: int) -> None:
    raw_env = make_vec_env(n_envs=1, seed=0, normalize=False)
    try:
        with pytest.raises(ValueError):
            DuckVecNormalize(raw_env, command_obs_index=command_obs_index)
    finally:
        raw_env.close()
