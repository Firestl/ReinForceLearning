"""训练层测试：配置校验、向量化环境工厂与评估环境装配。

训练层不改变环境语义，因此这里只验证"包装正确"：路径推导、批处理形状、
种子可复现，以及 ``evaluate`` 正确加载/冻结归一化统计量。
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from stable_baselines3.common.running_mean_std import RunningMeanStd

from reinforcelearning.envs.config import DuckEnvConfig
from reinforcelearning.training.config import TrainConfig, vecnormalize_file_for
from reinforcelearning.training.env_factory import make_env, make_vec_env
from reinforcelearning.training.runner import _make_eval_env, build_model, evaluate
from reinforcelearning.training.vec_normalize import DuckVecNormalize

OBS_DIM = 34
ACTION_DIM = 8


@pytest.mark.parametrize(
    ("model_path", "model_file", "stats_file"),
    [
        ("duck", "duck.zip", "duck_vecnormalize.pkl"),
        ("duck.zip", "duck.zip", "duck_vecnormalize.pkl"),
    ],
)
def test_output_paths_append_derived_suffixes(
    tmp_path: Path,
    model_path: str,
    model_file: str,
    stats_file: str,
) -> None:
    config = TrainConfig(model_path=tmp_path / model_path)
    assert config.model_file == tmp_path / model_file
    assert config.vecnormalize_file == tmp_path / stats_file


@pytest.mark.parametrize(
    "overrides",
    [
        {"total_timesteps": 0},
        {"n_envs": 0},
        {"seed": -1},
        {"device": ""},
        {"gamma": 0.0},
        {"gamma": float("nan")},
    ],
)
def test_train_config_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TrainConfig(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"total_timesteps": 1.5},
        {"n_envs": True},  # bool 是 int 的子类，必须显式拒绝
        {"seed": 1.0},
        {"device": 123},
        {"gamma": "0.99"},
        {"normalize": "yes"},
        {"norm_reward": None},
    ],
)
def test_train_config_rejects_invalid_types(overrides: dict[str, Any]) -> None:
    with pytest.raises(TypeError):
        TrainConfig(**overrides)


def test_vec_env_batches_duck_contract() -> None:
    vec_env = make_vec_env(n_envs=2, seed=0)
    try:
        assert vec_env.num_envs == 2
        obs = vec_env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2, OBS_DIM)
        assert obs.dtype == np.float32

        actions = np.zeros((2, ACTION_DIM), dtype=np.float32)
        obs, _, _, _ = vec_env.step(actions)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2, OBS_DIM)
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


def test_eval_env_loads_and_freezes_saved_statistics(tmp_path: Path) -> None:
    model_path = tmp_path / "duck_ppo.zip"
    stats_path = vecnormalize_file_for(model_path)

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

    eval_env = _make_eval_env(model_path, seed=0, render=False, load_normalization=True)
    assert isinstance(eval_env, DuckVecNormalize)
    try:
        assert eval_env.training is False
        assert eval_env.norm_reward is False
        eval_rms = eval_env.obs_rms
        assert isinstance(eval_rms, RunningMeanStd)
        np.testing.assert_allclose(eval_rms.mean, expected_mean)
    finally:
        eval_env.close()


@pytest.mark.parametrize(
    ("save_statistics", "load_normalization"),
    [
        (True, False),  # --raw：文件存在但显式跳过
        (False, True),  # 默认行为：文件缺失时退回原始观测
    ],
)
def test_eval_env_skips_statistics_when_disabled_or_missing(
    tmp_path: Path,
    save_statistics: bool,
    load_normalization: bool,
) -> None:
    model_path = tmp_path / "duck_ppo.zip"
    if save_statistics:
        source_env = make_vec_env(n_envs=1, seed=0)
        assert isinstance(source_env, DuckVecNormalize)
        try:
            source_env.save(str(vecnormalize_file_for(model_path)))
        finally:
            source_env.close()

    eval_env = _make_eval_env(
        model_path,
        seed=0,
        render=False,
        load_normalization=load_normalization,
    )
    try:
        assert not isinstance(eval_env, DuckVecNormalize)
    finally:
        eval_env.close()


def test_evaluate_rejects_non_positive_episodes() -> None:
    with pytest.raises(ValueError):
        evaluate(Path("unused.zip"), episodes=0)


def test_evaluate_runs_requested_episodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fast_env_config = DuckEnvConfig(max_episode_steps=8)
    monkeypatch.setattr(
        "reinforcelearning.training.runner.make_vec_env",
        functools.partial(make_vec_env, env_config=fast_env_config),
    )

    config = TrainConfig(model_path=tmp_path / "duck_ppo", log_dir=None)
    train_env = make_vec_env(n_envs=1, seed=0, env_config=fast_env_config, normalize=False)
    try:
        model = build_model(train_env, config)
        model.save(config.model_file)
    finally:
        train_env.close()

    returns = evaluate(config.model_file, episodes=2, seed=0, load_normalization=False)
    assert len(returns) == 2
    assert all(isinstance(value, float) for value in returns)
