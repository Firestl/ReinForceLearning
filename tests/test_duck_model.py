"""duck.xml 模型契约测试。

覆盖阶段 B 验收项：加载成功、默认姿态 root 高度与脚底接地、状态有限性、
执行器/关节顺序以及接触掩码。
"""

from __future__ import annotations

import itertools

import mujoco
import numpy as np
import pytest

from reinforcelearning.envs.config import JOINT_ORDER, DuckEnvConfig

# MuJoCo 索引约定：freejoint 占 qpos[0:7] 与 qvel[0:6]
FREE_JOINT_QPOS = 7
FREE_JOINT_QVEL = 6

# 契约镜像：鸭子 geom 统一 contype=2/conaffinity=1，与地面(1/1)接触、鸭子之间不接触
DUCK_CONTYPE = 2
DUCK_CONAFFINITY = 1


@pytest.fixture
def config() -> DuckEnvConfig:
    return DuckEnvConfig()


@pytest.fixture
def model(config: DuckEnvConfig) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(config.model_path))


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert geom_id >= 0, f"模型缺少必需 geom: {name}"
    return geom_id


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id >= 0, f"模型缺少必需 joint: {name}"
    return joint_id


def _default_pose_data(model: mujoco.MjModel, config: DuckEnvConfig) -> mujoco.MjData:
    """构造与 DuckEnv.reset() 一致的默认站立姿态（无噪声、零速度）。"""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = (0.0, 0.0, config.spawn_height)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)  # 单位四元数 (w, x, y, z)
    qpos_adr = np.array([model.jnt_qposadr[_joint_id(model, name)] for name in JOINT_ORDER])
    data.qpos[qpos_adr] = config.default_joint_pos
    data.ctrl[:] = config.default_joint_pos
    mujoco.mj_forward(model, data)
    return data


def _box_lowest_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """box geom 的 8 个角点经当前位姿变换后的最低 z 坐标。"""
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
    corners = np.array(list(itertools.product(*[(-s, s) for s in model.geom_size[geom_id]])))
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    world = data.geom_xpos[geom_id] + corners @ rotation.T
    return float(world[:, 2].min())


def _can_collide(model: mujoco.MjModel, geom_a: int, geom_b: int) -> bool:
    """MuJoCo 的广相接触判定规则。"""
    return bool(
        (model.geom_contype[geom_a] & model.geom_conaffinity[geom_b])
        or (model.geom_contype[geom_b] & model.geom_conaffinity[geom_a])
    )


def test_model_loads_with_expected_dimensions(model: mujoco.MjModel, config: DuckEnvConfig) -> None:
    root = _joint_id(model, "root")
    assert model.jnt_type[root] == mujoco.mjtJoint.mjJNT_FREE
    assert model.jnt_qposadr[root] == 0
    assert model.jnt_dofadr[root] == 0

    assert model.nq == FREE_JOINT_QPOS + config.action_dim
    assert model.nv == FREE_JOINT_QVEL + config.action_dim
    assert model.nu == config.action_dim
    assert model.opt.timestep > 0


def test_default_pose_root_height_and_foot_ground_contact(
    model: mujoco.MjModel, config: DuckEnvConfig
) -> None:
    data = _default_pose_data(model, config)

    assert float(data.qpos[2]) == pytest.approx(0.233, abs=1e-3)
    for name in ("left_foot_geom", "right_foot_geom"):
        lowest = _box_lowest_z(model, data, _geom_id(model, name))
        assert abs(lowest) <= 0.005, f"{name} 最低点 z={lowest:.5f} 超出 ±0.005 m"


def test_default_pose_is_numerically_stable(model: mujoco.MjModel, config: DuckEnvConfig) -> None:
    model.opt.timestep = config.sim_dt
    data = _default_pose_data(model, config)

    for _ in range(round(2.0 / config.sim_dt)):
        mujoco.mj_step(model, data)

    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    qpos_adr = np.array([model.jnt_qposadr[_joint_id(model, name)] for name in JOINT_ORDER])
    joint_error = np.abs(data.qpos[qpos_adr] - np.array(config.default_joint_pos))
    assert float(joint_error.max()) <= 0.005
    assert float(data.qpos[2]) == pytest.approx(config.spawn_height, abs=0.005)


def test_joint_and_actuator_order_matches_contract(
    model: mujoco.MjModel, config: DuckEnvConfig
) -> None:
    for i, name in enumerate(JOINT_ORDER):
        joint_id = _joint_id(model, name)
        assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        assert model.jnt_qposadr[joint_id] == FREE_JOINT_QPOS + i
        assert model.jnt_dofadr[joint_id] == FREE_JOINT_QVEL + i
        np.testing.assert_allclose(
            model.jnt_range[joint_id], [config.joint_lower[i], config.joint_upper[i]]
        )

        # 第 i 个执行器必须驱动第 i 个关节，且是 position actuator（fixed gain + affine bias）
        assert model.actuator_trnid[i, 0] == joint_id
        assert model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
        assert model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED
        assert model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE
        # 模型文档「执行器」允许 ctrlrange 与关节限位一致或略小；只需保证不越界，
        # 且覆盖环境可能写入的动作目标区间（与环境启动校验的条件一致）
        ctrl_low, ctrl_high = model.actuator_ctrlrange[i]
        assert ctrl_low >= config.joint_lower[i] - 1e-9
        assert ctrl_high <= config.joint_upper[i] + 1e-9
        assert ctrl_low <= config.action_target_low[i] + 1e-9
        assert ctrl_high >= config.action_target_high[i] - 1e-9


def test_contact_masks_allow_floor_and_block_self_collision(model: mujoco.MjModel) -> None:
    floor = _geom_id(model, "floor")
    # 只检查参与碰撞的 geom；后续视觉 geom 若 contype/conaffinity 为 0，不会被误判
    duck_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if geom_id != floor
        and (model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0)
    ]
    assert duck_geoms, "模型应至少包含一个鸭子碰撞 geom"
    for name in ("torso_geom", "head_geom", "left_foot_geom", "right_foot_geom"):
        assert _geom_id(model, name) in duck_geoms

    for geom_id in duck_geoms:
        assert model.geom_contype[geom_id] == DUCK_CONTYPE
        assert model.geom_conaffinity[geom_id] == DUCK_CONAFFINITY
        assert _can_collide(model, geom_id, floor)

    for geom_a, geom_b in itertools.combinations(duck_geoms, 2):
        assert not _can_collide(model, geom_a, geom_b)


def test_contact_detection_at_low_root_only_touches_floor(
    model: mujoco.MjModel, config: DuckEnvConfig
) -> None:
    data = _default_pose_data(model, config)
    data.qpos[2] = 0.05  # 躯干与脚掌压入地面，强制产生接触
    mujoco.mj_forward(model, data)

    assert data.ncon > 0
    floor = _geom_id(model, "floor")
    contacted: set[int] = set()
    for i in range(data.ncon):
        contact = data.contact[i]
        assert floor in (contact.geom1, contact.geom2), "鸭子自碰撞未被接触掩码屏蔽"
        contacted.update((int(contact.geom1), int(contact.geom2)))

    for name in ("torso_geom", "left_foot_geom", "right_foot_geom"):
        assert _geom_id(model, name) in contacted
