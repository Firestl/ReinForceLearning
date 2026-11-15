"""DuckEnv 行为测试。

期望值来自 docs/duck/01（模型与动作映射）和 docs/duck/02（环境接口）的规范，
以及真实物理行为；不通过复述实现细节来构造断言。少数需要注入精确物理状态或
单独验证奖励公式的用例会有意调用 ``_current_state``/``_get_obs``/``_compute_reward``，
这是对这些内部契约的白盒单测；期望值仍由文档公式或四元数解析计算得出。
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import gymnasium
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import reinforcelearning.envs  # noqa: F401  导入即注册 Duck-v0
from reinforcelearning.envs.config import JOINT_ORDER, DuckEnvConfig
from reinforcelearning.envs.duck_env import DuckEnv

OBS_DIM = 34
ACTION_DIM = 8

# 契约镜像：默认站姿与名义目标范围（q_default ± action_scale）；刻意硬编码，不从实现导入
DEFAULT_JOINT_POS = np.array((0.0, -0.30, 0.60, -0.30) * 2)
TARGET_LOW = np.array((-0.20, -0.65, 0.10, -0.55) * 2)
TARGET_HIGH = np.array((0.20, 0.05, 1.10, -0.05) * 2)

# 契约镜像：奖励分项与初始权重；刻意硬编码，避免用实现自证
REWARD_TERM_NAMES = (
    "tracking",
    "upright",
    "height_reward",
    "alive",
    "lateral",
    "yaw",
    "energy",
    "action_rate",
    "joint_tracking",
)
REWARD_WEIGHTS = {
    "tracking": 1.0,
    "upright": 0.5,
    "height_reward": 0.2,
    "alive": 0.1,
    "lateral": 0.05,
    "yaw": 0.05,
    "energy": 0.005,
    "action_rate": 0.01,
    "joint_tracking": 0.01,
}


@pytest.fixture
def env_factory() -> Iterator[Callable[..., DuckEnv]]:
    created: list[DuckEnv] = []

    def make(**overrides: Any) -> DuckEnv:
        env = DuckEnv(DuckEnvConfig(**overrides))
        created.append(env)
        return env

    yield make
    for env in created:
        env.close()


def _joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_ORDER]
    return (
        np.array([model.jnt_qposadr[joint_id] for joint_id in joint_ids]),
        np.array([model.jnt_dofadr[joint_id] for joint_id in joint_ids]),
    )


def _quat_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _quat_pitch(pitch: float) -> tuple[float, float, float, float]:
    return (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)


def _place(
    env: DuckEnv,
    *,
    z: float = 0.233,
    yaw: float = 0.0,
    pitch: float = 0.0,
    world_lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    body_ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """把鸭子放到指定物理状态，便于验证观测、终止等行为。"""
    if pitch and yaw:
        raise ValueError("_place 不支持同时设置 yaw 和 pitch")
    env.reset(seed=0)
    qpos_adr, dof_adr = _joint_addresses(env.model)
    env.data.qpos[0:3] = (0.0, 0.0, z)
    env.data.qpos[3:7] = _quat_pitch(pitch) if pitch else _quat_yaw(yaw)
    env.data.qpos[qpos_adr] = DEFAULT_JOINT_POS
    env.data.qvel[:] = 0.0
    env.data.qvel[0:3] = world_lin_vel
    env.data.qvel[3:6] = body_ang_vel
    env.data.qvel[dof_adr] = 0.0
    mujoco.mj_forward(env.model, env.data)


def test_spaces_and_step_api_follow_documented_contract(
    env_factory: Callable[..., DuckEnv],
) -> None:
    env = env_factory()

    assert env.observation_space.shape == (OBS_DIM,)
    assert env.observation_space.dtype == np.float32
    action_space = env.action_space
    assert isinstance(action_space, gymnasium.spaces.Box)
    assert action_space.shape == (ACTION_DIM,)
    assert action_space.dtype == np.float32
    assert np.all(action_space.low == -1.0)
    assert np.all(action_space.high == 1.0)

    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()

    obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_control_step_advances_by_configured_dt(
    env_factory: Callable[..., DuckEnv],
) -> None:
    """sim_dt 覆盖模型 timestep，每次 step 推进 frame_skip 个物理步。"""
    env = env_factory()
    config = env.config
    assert env.model.opt.timestep == pytest.approx(config.sim_dt)

    env.reset(seed=0)
    assert env.data.time == pytest.approx(0.0)
    env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert env.data.time == pytest.approx(config.control_dt)


def test_reset_state_matches_episode_start_rules(
    env_factory: Callable[..., DuckEnv],
) -> None:
    env = env_factory()
    config = env.config
    obs, info = env.reset(seed=0)

    qpos_adr, _ = _joint_addresses(env.model)
    assert env.data.qpos[2] == pytest.approx(config.spawn_height)

    joint_error = np.abs(env.data.qpos[qpos_adr] - DEFAULT_JOINT_POS)
    assert np.all(joint_error <= config.reset_joint_noise + 1e-12)
    assert np.all(np.abs(env.data.qvel) <= config.reset_velocity_noise + 1e-12)

    # 只绕 z 轴随机偏航，四元数应为单位四元数且 roll/pitch 为 0
    quat = env.data.qpos[3:7]
    assert np.linalg.norm(quat) == pytest.approx(1.0)
    assert quat[1] == pytest.approx(0.0, abs=1e-12)
    assert quat[2] == pytest.approx(0.0, abs=1e-12)

    low, high = config.command_range
    assert low <= info["command_speed"] <= high
    assert obs[33] == pytest.approx(info["command_speed"])


def test_reset_is_reproducible_for_same_seed(
    env_factory: Callable[..., DuckEnv],
) -> None:
    env_a = env_factory()
    env_b = env_factory()

    obs_a, info_a = env_a.reset(seed=123)
    obs_b, info_b = env_b.reset(seed=123)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert info_a["command_speed"] == info_b["command_speed"]

    env_a.step(np.zeros(ACTION_DIM, dtype=np.float32))
    obs_again, _ = env_a.reset(seed=123)
    np.testing.assert_array_equal(obs_a, obs_again)


def test_command_is_held_within_episode_and_sampled_per_reset(
    env_factory: Callable[..., DuckEnv],
) -> None:
    env = env_factory()
    low, high = env.config.command_range
    commands: list[float] = []

    for seed in range(5):
        obs, info = env.reset(seed=seed)
        command = info["command_speed"]
        assert low <= command <= high
        assert obs[33] == pytest.approx(command)

        for _ in range(3):
            obs, _, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
            assert not terminated and not truncated
            assert info["command_speed"] == command
            assert obs[33] == pytest.approx(command)

        commands.append(command)

    assert len(set(commands)) > 1


def test_zero_action_holds_standing_posture(
    env_factory: Callable[..., DuckEnv],
) -> None:
    """零动作映射到默认目标角，位置执行器应让鸭子保持站立（真实物理行为）。"""
    env = env_factory()
    env.reset(seed=0)

    for _ in range(100):  # 100 × 20 ms = 2 s
        obs, reward, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
        assert not terminated and not truncated
        assert np.isfinite(obs).all()
        assert math.isfinite(reward)

    qpos_adr, _ = _joint_addresses(env.model)
    joint_error = np.abs(env.data.qpos[qpos_adr] - DEFAULT_JOINT_POS)
    assert float(joint_error.max()) <= 0.01
    assert env.data.qpos[2] == pytest.approx(env.config.spawn_height, abs=0.01)
    assert info["left_foot_contact"] is True
    assert info["right_foot_contact"] is True


@pytest.mark.parametrize(
    ("action_value", "expected_ctrl"),
    [(0.0, DEFAULT_JOINT_POS), (1.0, TARGET_HIGH), (-1.0, TARGET_LOW)],
)
def test_action_maps_to_documented_joint_targets(
    env_factory: Callable[..., DuckEnv],
    action_value: float,
    expected_ctrl: np.ndarray,
) -> None:
    env = env_factory()
    _place(env)
    env.step(np.full(ACTION_DIM, action_value, dtype=np.float32))
    np.testing.assert_allclose(env.data.ctrl, expected_ctrl, atol=1e-6)


def test_each_action_dimension_drives_its_own_joint(
    env_factory: Callable[..., DuckEnv],
) -> None:
    env = env_factory()
    for index in range(ACTION_DIM):
        _place(env)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[index] = 1.0
        env.step(action)

        expected = DEFAULT_JOINT_POS.copy()
        expected[index] = TARGET_HIGH[index]
        np.testing.assert_allclose(env.data.ctrl, expected, atol=1e-6)


@pytest.mark.parametrize(("raw_value", "expected_ctrl"), [(5.0, TARGET_HIGH), (-5.0, TARGET_LOW)])
def test_actions_outside_unit_range_are_clipped(
    env_factory: Callable[..., DuckEnv],
    raw_value: float,
    expected_ctrl: np.ndarray,
) -> None:
    env = env_factory()
    _place(env)
    env.step(np.full(ACTION_DIM, raw_value, dtype=np.float32))
    np.testing.assert_allclose(env.data.ctrl, expected_ctrl, atol=1e-6)


def test_action_targets_are_clipped_to_joint_limits_with_margin(
    env_factory: Callable[..., DuckEnv],
) -> None:
    """动作尺度放大到超出限位时，目标角应裁剪到 [q_min+margin, q_max-margin]。"""
    env = env_factory(action_scale=(1.0,) * ACTION_DIM)
    expected_high = np.array((0.33, 0.70, 1.60, 0.58) * 2)
    expected_low = np.array((-0.33, -1.28, 0.02, -0.88) * 2)

    _place(env)
    env.step(np.ones(ACTION_DIM, dtype=np.float32))
    np.testing.assert_allclose(env.data.ctrl, expected_high, atol=1e-6)

    env.step(-np.ones(ACTION_DIM, dtype=np.float32))
    np.testing.assert_allclose(env.data.ctrl, expected_low, atol=1e-6)


@pytest.mark.parametrize(
    "bad_action",
    [
        np.zeros(ACTION_DIM - 1, dtype=np.float32),
        np.full(ACTION_DIM, np.nan, dtype=np.float32),
        np.full(ACTION_DIM, np.inf, dtype=np.float32),
    ],
)
def test_invalid_actions_are_rejected_without_advancing_simulation(
    env_factory: Callable[..., DuckEnv],
    bad_action: np.ndarray,
) -> None:
    env = env_factory()
    env.reset(seed=0)
    time_before = float(env.data.time)

    with pytest.raises(ValueError):
        env.step(bad_action)

    assert float(env.data.time) == time_before


def test_reset_observation_layout_matches_spec(
    env_factory: Callable[..., DuckEnv],
) -> None:
    """无噪声、零指令下，34 维观测应完全可预测。"""
    env = env_factory(reset_joint_noise=0.0, reset_velocity_noise=0.0, command_range=(0.0, 0.0))
    obs, _ = env.reset(seed=0)

    np.testing.assert_allclose(obs[0:3], 0.0, atol=1e-7)  # 基座线速度
    np.testing.assert_allclose(obs[3:6], 0.0, atol=1e-7)  # 基座角速度
    np.testing.assert_allclose(obs[6:9], (0.0, 0.0, -1.0), atol=1e-6)  # 重力投影
    np.testing.assert_allclose(obs[9:17], DEFAULT_JOINT_POS, atol=1e-6)  # 关节角
    np.testing.assert_allclose(obs[17:25], 0.0, atol=1e-7)  # 关节角速度
    np.testing.assert_allclose(obs[25:33], 0.0, atol=1e-7)  # 上一步动作
    assert obs[33] == pytest.approx(0.0)  # 速度指令


def test_observation_joint_angles_and_velocities_follow_contract_order(
    env_factory: Callable[..., DuckEnv],
) -> None:
    """注入互不相同的关节状态，验证左右腿与同腿内关节顺序未被置换。"""
    env = env_factory()
    env.reset(seed=0)
    qpos_adr, dof_adr = _joint_addresses(env.model)
    joint_pos = np.array((0.01, -0.21, 0.51, -0.41, -0.02, -0.22, 0.52, -0.42))
    joint_vel = np.array((0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18))

    env.data.qpos[qpos_adr] = joint_pos
    env.data.qvel[:] = 0.0
    env.data.qvel[dof_adr] = joint_vel
    mujoco.mj_forward(env.model, env.data)

    obs = env._get_obs(env._current_state())
    np.testing.assert_allclose(obs[9:17], joint_pos, atol=1e-6)
    np.testing.assert_allclose(obs[17:25], joint_vel, atol=1e-6)


def test_observation_uses_body_frame_and_excludes_world_position(
    env_factory: Callable[..., DuckEnv],
) -> None:
    """偏航 90°、世界系 +x 速度 1 m/s 时，机体系速度应为 (0, -1, 0)。"""
    env = env_factory(reset_joint_noise=0.0, reset_velocity_noise=0.0, command_range=(0.0, 0.0))
    _place(
        env,
        yaw=math.pi / 2,
        world_lin_vel=(1.0, 0.0, 0.0),
        body_ang_vel=(0.1, 0.2, 0.3),
    )

    # 直接注入状态后读取观测，避免 step 推进改变待验证的速度；期望值由四元数解析给出
    obs = env._get_obs(env._current_state())
    np.testing.assert_allclose(obs[0:3], (0.0, -1.0, 0.0), atol=1e-5)
    np.testing.assert_allclose(obs[3:6], (0.1, 0.2, 0.3), atol=1e-6)
    np.testing.assert_allclose(obs[6:9], (0.0, 0.0, -1.0), atol=1e-6)
    np.testing.assert_allclose(obs[9:17], DEFAULT_JOINT_POS, atol=1e-6)
    np.testing.assert_allclose(obs[17:25], 0.0, atol=1e-7)
    np.testing.assert_allclose(obs[25:33], 0.0, atol=1e-7)

    # 世界系 x,y 不应进入观测
    env.data.qpos[0:2] = (5.0, 3.0)
    mujoco.mj_forward(env.model, env.data)
    np.testing.assert_array_equal(obs, env._get_obs(env._current_state()))


def test_observation_reports_previous_action(
    env_factory: Callable[..., DuckEnv],
) -> None:
    env = env_factory()
    env.reset(seed=0)
    action = np.array((0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8), dtype=np.float32)

    obs, *_ = env.step(action)
    np.testing.assert_array_equal(obs[25:33], action)


def test_reward_terms_follow_documented_formula(
    env_factory: Callable[..., DuckEnv],
) -> None:
    env = env_factory(command_range=(0.5, 0.5))
    env.reset(seed=0)
    env.step(np.full(ACTION_DIM, 0.1))  # 设置上一步动作，用于验证 action_rate
    action = np.full(ACTION_DIM, -0.2)

    state = dataclasses.replace(
        env._current_state(),
        lin_vel=np.array((0.0, 0.2, 0.0)),
        ang_vel=np.array((0.0, 0.0, 0.3)),
        projected_gravity=np.array((0.0, 0.0, -0.9)),
        body_up_z=0.9,
        tilt_angle=math.acos(0.9),
        body_height=0.283,
        joint_pos=DEFAULT_JOINT_POS + 0.1,
        joint_vel=np.zeros(ACTION_DIM),
        ctrl=DEFAULT_JOINT_POS.copy(),
        actuator_force=np.array((1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0)),
    )
    reward, terms = env._compute_reward(action, state)

    # 环境文档「奖励函数」的公式；期望值独立计算，非零误差锁住两个 sigma
    values = {
        "tracking": math.exp(-(((0.0 - 0.5) / 0.5) ** 2)),
        "upright": 0.9,
        "height_reward": math.exp(-(((0.283 - 0.233) / 0.05) ** 2)),
        "alive": 1.0,
        "lateral": -(0.2**2),
        "yaw": -(0.3**2),
        "energy": -(1.0**2 + (-2.0) ** 2 + 0.5**2),
        "action_rate": -(ACTION_DIM * (0.1 - (-0.2)) ** 2),
        "joint_tracking": -(ACTION_DIM * 0.1**2),
    }
    assert set(terms) == set(REWARD_TERM_NAMES)
    for name in REWARD_TERM_NAMES:
        assert terms[name] == pytest.approx(REWARD_WEIGHTS[name] * values[name])
    assert reward == pytest.approx(sum(REWARD_WEIGHTS[name] * values[name] for name in values))

    # 偏离指令速度应降低跟踪奖励，并锁住 speed_sigma 的解析值
    far_state = dataclasses.replace(state, lin_vel=np.array((1.5, 0.0, 0.0)))
    _, far_terms = env._compute_reward(action, far_state)
    assert far_terms["tracking"] == pytest.approx(REWARD_WEIGHTS["tracking"] * math.exp(-4.0))
    assert far_terms["tracking"] < terms["tracking"]


def test_step_info_reports_consistent_reward_terms(
    env_factory: Callable[..., DuckEnv],
) -> None:
    env = env_factory()
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))

    assert not terminated and not truncated
    assert info["termination_reason"] is None
    assert set(info["reward_terms"]) == set(REWARD_TERM_NAMES)
    assert sum(info["reward_terms"].values()) == pytest.approx(reward)


def test_info_reports_physical_diagnostics(
    env_factory: Callable[..., DuckEnv],
) -> None:
    env = env_factory()
    env.reset(seed=0)
    obs, _, _, _, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))

    required = {
        "forward_speed",
        "lateral_speed",
        "body_height",
        "tilt_angle",
        "command_speed",
        "left_foot_contact",
        "right_foot_contact",
        "termination_reason",
        "reward_terms",
    }
    assert required <= info.keys()

    torso = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    rotation = env.data.xmat[torso].reshape(3, 3)
    forward_speed = float((rotation.T @ env.data.qvel[0:3])[0])
    lateral_speed = float((rotation.T @ env.data.qvel[0:3])[1])
    assert info["forward_speed"] == pytest.approx(forward_speed, abs=1e-6)
    assert info["lateral_speed"] == pytest.approx(lateral_speed, abs=1e-6)
    assert info["body_height"] == pytest.approx(float(env.data.qpos[2]))
    assert info["tilt_angle"] == pytest.approx(
        math.acos(np.clip(rotation[2, 2], -1.0, 1.0)), abs=1e-6
    )
    assert info["command_speed"] == pytest.approx(obs[33])
    assert isinstance(info["left_foot_contact"], bool)
    assert isinstance(info["right_foot_contact"], bool)


def test_excessive_tilt_terminates_episode(env_factory: Callable[..., DuckEnv]) -> None:
    env = env_factory()
    _place(env, pitch=math.radians(70))  # 超过环境文档「终止与超时」的 60° 阈值

    _, _, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "excessive_tilt"
    assert info["tilt_angle"] > env.config.max_tilt


def test_torso_floor_contact_terminates_episode(
    env_factory: Callable[..., DuckEnv],
) -> None:
    env = env_factory()
    _place(env, z=-0.02)  # 躯干压入地面

    _, _, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "fall_contact"


def test_too_low_body_terminates_episode(env_factory: Callable[..., DuckEnv]) -> None:
    env = env_factory()
    # 低于环境文档「终止与超时」的 0.15 m 阈值，但躯干尚未触地，隔离 too_low 分支
    _place(env, z=0.10)

    _, _, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "too_low"
    assert info["body_height"] < env.config.fall_height


def test_non_finite_physics_terminates_episode(
    env_factory: Callable[..., DuckEnv],
) -> None:
    env = env_factory()
    _place(env)
    env.data.qvel[0] = np.nan  # 模拟数值爆炸

    _, _, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "non_finite"


def test_timeout_sets_truncated_only(env_factory: Callable[..., DuckEnv]) -> None:
    env = env_factory(max_episode_steps=1)
    env.reset(seed=0)

    _, _, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))

    assert terminated is False
    assert truncated is True
    assert info["termination_reason"] == "timeout"


def test_termination_wins_over_truncation(env_factory: Callable[..., DuckEnv]) -> None:
    env = env_factory(max_episode_steps=1)
    _place(env, pitch=math.radians(70))

    _, _, terminated, truncated, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "excessive_tilt"


def test_registered_env_passes_gymnasium_checker() -> None:
    assert "Duck-v0" in gymnasium.registry
    env = gymnasium.make("Duck-v0")
    try:
        with warnings.catch_warnings():
            # 已知：Box(-inf, inf) 会触发 unbounded 警告；只忽略这一类，保留其他告警
            warnings.filterwarnings(
                "ignore", message=".*Box observation space.*", category=UserWarning
            )
            check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


def test_thousand_random_steps_stay_finite(env_factory: Callable[..., DuckEnv]) -> None:
    env = env_factory()
    rng = np.random.default_rng(2026)
    obs, _ = env.reset(seed=0)

    for step in range(1000):
        obs, reward, terminated, truncated, _ = env.step(
            rng.uniform(-1.0, 1.0, ACTION_DIM).astype(np.float32)
        )
        assert np.isfinite(obs).all()
        assert math.isfinite(reward)
        assert np.isfinite(env.data.qpos).all()
        assert np.isfinite(env.data.qvel).all()

        if terminated or truncated:
            obs, _ = env.reset(seed=step)


def test_render_contract() -> None:
    env = DuckEnv(render_mode=None)
    try:
        assert env.metadata["render_modes"] == ["human", "rgb_array"]
        assert env.metadata["render_fps"] == round(1.0 / env.config.control_dt)
        env.reset(seed=0)
        assert env.render() is None
    finally:
        env.close()

    with pytest.raises(ValueError):
        DuckEnv(render_mode="invalid-mode")

    env = DuckEnv(render_mode="rgb_array")
    try:
        env.reset(seed=0)
        env.step(np.zeros(ACTION_DIM, dtype=np.float32))
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
    finally:
        env.close()
