"""duck.xml 模型契约：名称、索引解析与启动校验的唯一归属。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from reinforcelearning.envs.config import JOINT_AXES, JOINT_ORDER, DuckEnvConfig

# 接触检测与躯干姿态依赖的命名实体；与 assets/duck.xml 绑定
FLOOR_GEOM_NAME = "floor"
TORSO_GEOM_NAME = "torso_geom"
HEAD_GEOM_NAME = "head_geom"
LEFT_FOOT_GEOM_NAME = "left_foot_geom"
RIGHT_FOOT_GEOM_NAME = "right_foot_geom"
TORSO_BODY_NAME = "torso"


@dataclass(frozen=True, slots=True)
class DuckModelContract:
    """通过校验的模型索引；环境只通过此对象访问名称对应的 id 与关节地址。"""

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
        """解析并校验模型；任何契约不符都汇总为一次 ValueError。"""
        errors: list[str] = []
        n = config.action_dim
        target_low = config.action_target_low
        target_high = config.action_target_high

        def resolve_id(obj_type: int, name: str, kind: str) -> int:
            obj_id = mujoco.mj_name2id(model, obj_type, name)
            if obj_id < 0:
                errors.append(f"模型缺少必需 {kind}: {name}")
            return obj_id

        if model.nq != 7 + n or model.nv != 6 + n or model.nu != n:
            errors.append(
                f"维度不符: nq={model.nq} nv={model.nv} nu={model.nu}，"
                f"期望 nq={7 + n} nv={6 + n} nu={n}"
            )

        joint_ids = [resolve_id(mujoco.mjtObj.mjOBJ_JOINT, name, "joint") for name in JOINT_ORDER]

        for i, (name, jid) in enumerate(zip(JOINT_ORDER, joint_ids, strict=True)):
            if jid < 0:
                continue
            if model.jnt_qposadr[jid] != 7 + i or model.jnt_dofadr[jid] != 6 + i:
                errors.append(
                    f"关节顺序不符: {name} qposadr={model.jnt_qposadr[jid]}，期望 {7 + i}"
                )
            jnt_range = model.jnt_range[jid]
            if not (
                np.isclose(jnt_range[0], config.joint_lower[i], atol=1e-9)
                and np.isclose(jnt_range[1], config.joint_upper[i], atol=1e-9)
            ):
                errors.append(
                    f"关节限位不符: {name} model={jnt_range}，"
                    f"config=({config.joint_lower[i]}, {config.joint_upper[i]})"
                )
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                errors.append(f"关节类型不符: {name} 应为 hinge")
            if not np.allclose(model.jnt_axis[jid], JOINT_AXES[i], atol=1e-9):
                errors.append(
                    f"关节轴不符: {name} model={model.jnt_axis[jid]}，期望 {JOINT_AXES[i]}"
                )
            if model.nu > i:
                act_joint = model.actuator_trnid[i, 0]
                if act_joint != jid:
                    errors.append(f"执行器顺序不符: 第 {i} 个执行器未驱动 {name}")
                # 必须是 joint transmission 上的 position actuator（fixed gain + affine bias）；
                # 例如误用 motor 时 ctrl 语义会从目标角变成力矩
                if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
                    errors.append(f"执行器类型不符: 第 {i} 个执行器不是 joint transmission")
                if model.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED:
                    errors.append(f"执行器 gain 类型不符: 第 {i} 个执行器应为 fixed gain")
                if model.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_AFFINE:
                    errors.append(
                        f"执行器 bias 类型不符: 第 {i} 个执行器应为 affine bias（position actuator）"
                    )
                # ctrlrange 不得改变动作语义（docs/duck/01 §6：与关节限位一致或略小）：
                # 不超出关节限位，且覆盖环境可能写入的目标角范围
                # （default ± action_scale 与限位内 margin 区间的交集），
                # 否则 MuJoCo 会再次 clamp，动作映射被静默修改
                ctrl_lo, ctrl_hi = model.actuator_ctrlrange[i]
                if not (ctrl_lo <= target_low[i] and ctrl_hi >= target_high[i]):
                    errors.append(
                        f"执行器 ctrlrange 不覆盖动作目标范围: {name} "
                        f"model=[{ctrl_lo}, {ctrl_hi}]，"
                        f"需包含 [{target_low[i]:.3f}, {target_high[i]:.3f}]"
                    )
                if not (ctrl_lo >= config.joint_lower[i] and ctrl_hi <= config.joint_upper[i]):
                    errors.append(
                        f"执行器 ctrlrange 超出关节限位: {name} "
                        f"model=[{ctrl_lo}, {ctrl_hi}]，"
                        f"限位=({config.joint_lower[i]}, {config.joint_upper[i]})"
                    )

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
