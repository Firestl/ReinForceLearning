"""duck.xml 模型契约：实体名称解析、拓扑索引映射与启动前严格校验。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from reinforcelearning.envs.config import JOINT_AXES, JOINT_ORDER, DuckEnvConfig

# 接触动力学与姿态解算绑定的关键命名实体
FLOOR_GEOM_NAME = "floor"
TORSO_GEOM_NAME = "torso_geom"
HEAD_GEOM_NAME = "head_geom"
LEFT_FOOT_GEOM_NAME = "left_foot_geom"
RIGHT_FOOT_GEOM_NAME = "right_foot_geom"
TORSO_BODY_NAME = "torso"


@dataclass(frozen=True, slots=True)
class DuckModelContract:
    """已通过严格校验的模型拓扑索引缓存对象。"""

    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    floor_geom: int
    torso_geom: int
    head_geom: int
    left_foot_geom: int
    right_foot_geom: int
    torso_body: int

    @classmethod
    def from_model(cls, model: mujoco.MjModel, config: DuckEnvConfig) -> DuckModelContract:
        """从 MjModel 编译契约并校验；若有不符汇总抛出单个 ValueError。"""
        errors: list[str] = []
        n = config.action_dim
        target_low = config.action_target_low
        target_high = config.action_target_high

        def resolve_id(obj_type: int, name: str, kind: str) -> int:
            obj_id = mujoco.mj_name2id(model, obj_type, name)
            if obj_id < 0:
                errors.append(f"模型缺少必需 {kind}: {name}")
            return obj_id

        # 1. 广义坐标与控制维度校验 (nq=7+n, nv=6+n, nu=n)
        if model.nq != 7 + n or model.nv != 6 + n or model.nu != n:
            errors.append(
                f"维度契约不符: nq={model.nq}, nv={model.nv}, nu={model.nu} (期望 {7+n}, {6+n}, {n})"
            )

        # 2. 关节拓扑与限位校验
        joint_ids = [resolve_id(mujoco.mjtObj.mjOBJ_JOINT, name, "joint") for name in JOINT_ORDER]

        for i, (name, jid) in enumerate(zip(JOINT_ORDER, joint_ids, strict=True)):
            if jid < 0:
                continue

            if model.jnt_qposadr[jid] != 7 + i or model.jnt_dofadr[jid] != 6 + i:
                errors.append(f"关节存储顺序不符: {name} qposadr={model.jnt_qposadr[jid]}, 期望 {7+i}")

            jnt_range = model.jnt_range[jid]
            if not (
                np.isclose(jnt_range[0], config.joint_lower[i], atol=1e-9)
                and np.isclose(jnt_range[1], config.joint_upper[i], atol=1e-9)
            ):
                errors.append(
                    f"关节限位不符: {name} model={jnt_range}, config=({config.joint_lower[i]}, {config.joint_upper[i]})"
                )

            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                errors.append(f"关节类型错误: {name} 应为 hinge")

            if not np.allclose(model.jnt_axis[jid], JOINT_AXES[i], atol=1e-9):
                errors.append(f"关节旋转轴不符: {name} model={model.jnt_axis[jid]}, 期望 {JOINT_AXES[i]}")

            # 3. 执行器物理语义校验
            if model.nu > i:
                if model.actuator_trnid[i, 0] != jid:
                    errors.append(f"执行器映射错误: 第 {i} 个执行器未绑定 {name}")
                if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
                    errors.append(f"执行器传输类型错误: 第 {i} 个执行器非 joint transmission")

                # 确保是位置执行器 (fixed gain + affine bias)，避免误配为 motor 导致控制语义偏移为力矩
                if model.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED:
                    errors.append(f"执行器 gain 类型错误: 第 {i} 个执行器非 fixed gain")
                if model.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_AFFINE:
                    errors.append(f"执行器 bias 类型错误: 第 {i} 个执行器非 affine bias (position actuator)")

                # ctrlrange 必须完全覆盖动作名义范围，避免 MuJoCo 内部静默再次截断导致映射失真
                ctrl_lo, ctrl_hi = model.actuator_ctrlrange[i]
                if not (ctrl_lo <= target_low[i] and ctrl_hi >= target_high[i]):
                    errors.append(
                        f"执行器 ctrlrange 未覆盖动作目标范围: {name} model=[{ctrl_lo}, {ctrl_hi}], "
                        f"需包含 [{target_low[i]:.3f}, {target_high[i]:.3f}]"
                    )
                if not (ctrl_lo >= config.joint_lower[i] and ctrl_hi <= config.joint_upper[i]):
                    errors.append(
                        f"执行器 ctrlrange 超出关节物理限位: {name} model=[{ctrl_lo}, {ctrl_hi}], "
                        f"限位=({config.joint_lower[i]}, {config.joint_upper[i]})"
                    )

        # 4. 关键碰撞几何体与刚体解析
        floor_geom = resolve_id(mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME, "geom")
        torso_geom = resolve_id(mujoco.mjtObj.mjOBJ_GEOM, TORSO_GEOM_NAME, "geom")
        head_geom = resolve_id(mujoco.mjtObj.mjOBJ_GEOM, HEAD_GEOM_NAME, "geom")
        left_foot_geom = resolve_id(mujoco.mjtObj.mjOBJ_GEOM, LEFT_FOOT_GEOM_NAME, "geom")
        right_foot_geom = resolve_id(mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FOOT_GEOM_NAME, "geom")
        torso_body = resolve_id(mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME, "body")

        if errors:
            raise ValueError("duck.xml 与 DuckEnvConfig 契约不一致:\n" + "\n".join(errors))

        return cls(
            qpos_adr=model.jnt_qposadr[joint_ids],
            dof_adr=model.jnt_dofadr[joint_ids],
            floor_geom=floor_geom,
            torso_geom=torso_geom,
            head_geom=head_geom,
            left_foot_geom=left_foot_geom,
            right_foot_geom=right_foot_geom,
            torso_body=torso_body,
        )
