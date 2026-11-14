"""鸭形机器人 Gymnasium 环境，实现规范见 docs/02-physics-environment.md。"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces

from reinforcelearning.envs.config import JOINT_ORDER, DuckEnvConfig

# 表示物理状态非有限的 MuJoCo 警告；数值异常时 MuJoCo 会自动重置 qpos，
# 因此事后检查 qpos 的有限性不够，需要监视警告计数
_PHYSICS_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADCTRL,
)

# info["reward_terms"] 的固定键顺序，惩罚项为负，九项之和等于 reward
_REWARD_TERM_KEYS = (
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


class DuckEnv(gymnasium.Env):
    """平地直线行走的鸭形双足环境（位置执行器、本体感知观测）。"""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: DuckEnvConfig | None = None,
        *,
        render_mode: str | None = None,
    ) -> None:
        self.config = config if config is not None else DuckEnvConfig()
        cfg = self.config

        # 实例级 metadata：render_fps 按实际控制周期推导，避免与 frame_skip/sim_dt 脱节
        self.metadata = {**type(self).metadata, "render_fps": round(1.0 / cfg.control_dt)}
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"不支持的 render_mode={render_mode!r}，可选 {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(cfg.model_path))
        # sim_dt 是物理步长的唯一来源，覆盖 XML 中的 timestep
        self.model.opt.timestep = cfg.sim_dt
        self.data = mujoco.MjData(self.model)
        self._validate_model()

        # 按名称建立索引，不硬编码下标
        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_ORDER
        ]
        self._qpos_adr = self.model.jnt_qposadr[joint_ids]
        self._dof_adr = self.model.jnt_dofadr[joint_ids]

        self._floor_geom = self._geom_id("floor")
        self._torso_geom = self._geom_id("torso_geom")
        self._head_geom = self._geom_id("head_geom")
        self._left_foot_geom = self._geom_id("left_foot_geom")
        self._right_foot_geom = self._geom_id("right_foot_geom")
        self._torso_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(cfg.action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(cfg.obs_dim,), dtype=np.float32
        )

        # 动作目标角的安全裁剪范围：关节限位内收缩 margin
        self._ctrl_low = np.asarray(cfg.joint_lower) + cfg.joint_limits_margin
        self._ctrl_high = np.asarray(cfg.joint_upper) - cfg.joint_limits_margin

        self._step_count = 0
        self._command = 0.0
        self._last_action = np.zeros(cfg.action_dim, dtype=np.float64)
        self._warning_count = self._physics_warning_count()
        self._viewer: Any = None
        self._renderer: Any = None

    # ------------------------------------------------------------------ 核心接口

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        cfg = self.config
        _ = options  # 第一版不使用 reset 选项

        mujoco.mj_resetData(self.model, self.data)

        # 根节点：默认站姿高度 + 随机朝向（绕 z 轴 ±pi）
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[0:3] = 0.0
        self.data.qpos[2] = cfg.reward.height_target
        self.data.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))

        # 关节默认姿态 + 小幅噪声；速度置零 + 极小噪声
        self.data.qpos[self._qpos_adr] = cfg.default_joint_pos
        self.data.qpos[self._qpos_adr] += self.np_random.uniform(
            -cfg.reset_joint_noise, cfg.reset_joint_noise, cfg.action_dim
        )
        self.data.qvel[:] = self.np_random.uniform(
            -cfg.reset_velocity_noise, cfg.reset_velocity_noise, self.model.nv
        )

        self._command = float(self.np_random.uniform(*cfg.command_range))
        self._step_count = 0
        self._last_action = np.zeros(cfg.action_dim, dtype=np.float64)
        self._warning_count = self._physics_warning_count()

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._build_info(termination_reason=None)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cfg = self.config
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (cfg.action_dim,):
            raise ValueError(f"动作维度应为 {cfg.action_dim}，实际 shape={action.shape}")
        action = np.clip(action, -1.0, 1.0)

        # 归一化动作 -> 目标关节角 -> 限位内裁剪 -> 位置执行器
        q_target = cfg.default_joint_pos + cfg.action_scale * action
        self.data.ctrl[:] = np.clip(q_target, self._ctrl_low, self._ctrl_high)

        for _ in range(cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        state = self._current_state()
        reward, reward_terms = self._compute_reward(action, state)
        terminated, reason = self._check_termination(state)
        truncated = (not terminated) and self._step_count >= cfg.max_episode_steps
        if truncated:
            reason = "timeout"

        self._last_action = action
        obs = self._get_obs(state)
        info = self._build_info(reward_terms=reward_terms, termination_reason=reason)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        # human
        if self._viewer is None:
            from mujoco import viewer as viewer_module

            self._viewer = viewer_module.launch_passive(self.model, self.data)
        else:
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ 状态与观测

    def _current_state(self) -> dict[str, Any]:
        """从 qvel/xmat 提取身体坐标系下的运动量。"""
        rotation = self.data.xmat[self._torso_body].reshape(3, 3)
        lin_vel = rotation.T @ self.data.qvel[0:3]  # 世界系 -> 身体系
        ang_vel = self.data.qvel[3:6].copy()  # free joint 角速度本身就是 body 系，不可再旋转
        projected_gravity = -rotation[2, :]  # R^T @ (0, 0, -1)
        body_up_z = -projected_gravity[2]
        tilt = float(np.arccos(np.clip(body_up_z, -1.0, 1.0)))
        return {
            "lin_vel": lin_vel,
            "ang_vel": ang_vel,
            "projected_gravity": projected_gravity,
            "body_up_z": float(body_up_z),
            "tilt_angle": tilt,
            "body_height": float(self.data.qpos[2]),
        }

    def _get_obs(self, state: dict[str, Any] | None = None) -> np.ndarray:
        state = self._current_state() if state is None else state
        return np.concatenate(
            [
                state["lin_vel"],
                state["ang_vel"],
                state["projected_gravity"],
                self.data.qpos[self._qpos_adr],
                self.data.qvel[self._dof_adr],
                self._last_action,
                np.array([self._command]),
            ]
        ).astype(np.float32)

    # ------------------------------------------------------------------ 奖励与终止

    def _compute_reward(
        self, action: np.ndarray, state: dict[str, Any]
    ) -> tuple[float, dict[str, float]]:
        cfg = self.config
        rc = cfg.reward

        speed_error = float(state["lin_vel"][0]) - self._command
        tracking = float(np.exp(-((speed_error / rc.speed_sigma) ** 2)))
        upright = float(np.clip(state["body_up_z"], 0.0, 1.0))
        height_term = float(
            np.exp(-(((state["body_height"] - rc.height_target) / rc.height_sigma) ** 2))
        )
        joint_tracking_error = self.data.qpos[self._qpos_adr] - self.data.ctrl

        terms = {
            "tracking": rc.tracking_weight * tracking,
            "upright": rc.upright_weight * upright,
            "height_reward": rc.height_weight * height_term,
            "alive": rc.alive_weight,
            "lateral": -rc.lateral_weight * float(state["lin_vel"][1] ** 2),
            "yaw": -rc.yaw_weight * float(state["ang_vel"][2] ** 2),
            "energy": -rc.energy_weight * float(np.sum(np.square(self.data.actuator_force))),
            "action_rate": -rc.action_rate_weight
            * float(np.sum(np.square(action - self._last_action))),
            "joint_tracking": -rc.joint_tracking_weight
            * float(np.sum(np.square(joint_tracking_error))),
        }
        return float(sum(terms.values())), terms

    def _check_termination(self, state: dict[str, Any]) -> tuple[bool, str | None]:
        if self._physics_warning_count() > self._warning_count or not (
            np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        ):
            return True, "non_finite"
        _, _, fall = self._contact_flags()
        if fall:
            return True, "fall_contact"
        if state["tilt_angle"] > self.config.max_tilt:
            return True, "excessive_tilt"
        if state["body_height"] < self.config.fall_height:
            return True, "too_low"
        return False, None

    # ------------------------------------------------------------------ 接触与 info

    def _physics_warning_count(self) -> int:
        """累计的物理状态警告数（BADQPOS/BADQVEL/BADQACC/BADCTRL）。"""
        return int(sum(self.data.warning[w].number for w in _PHYSICS_WARNINGS))

    def _contact_flags(self) -> tuple[bool, bool, bool]:
        """返回 (左脚触地, 右脚触地, 躯干/头部触地)。"""
        left = right = fall = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {contact.geom1, contact.geom2}
            if self._floor_geom not in pair:
                continue  # 自碰撞已通过 contype/conaffinity 关闭
            other = next(g for g in pair if g != self._floor_geom)
            if other == self._left_foot_geom:
                left = True
            elif other == self._right_foot_geom:
                right = True
            elif other in (self._torso_geom, self._head_geom):
                fall = True
        return left, right, fall

    def _build_info(
        self,
        reward_terms: dict[str, float] | None = None,
        termination_reason: str | None = None,
    ) -> dict[str, Any]:
        state = self._current_state()
        left_contact, right_contact, _ = self._contact_flags()
        info: dict[str, Any] = {
            "forward_speed": float(state["lin_vel"][0]),
            "lateral_speed": float(state["lin_vel"][1]),
            "body_height": state["body_height"],
            "tilt_angle": state["tilt_angle"],
            "command_speed": self._command,
            "left_foot_contact": bool(left_contact),
            "right_foot_contact": bool(right_contact),
            "termination_reason": termination_reason,
        }
        if reward_terms is not None:
            info["reward_terms"] = {key: float(reward_terms[key]) for key in _REWARD_TERM_KEYS}
        return info

    # ------------------------------------------------------------------ 模型校验与工具

    def _validate_model(self) -> None:
        """启动时校验模型与配置契约一致，避免静默的接口漂移。"""
        cfg = self.config
        n = cfg.action_dim
        errors: list[str] = []

        if self.model.nq != 7 + n or self.model.nv != 6 + n or self.model.nu != n:
            errors.append(
                f"维度不符: nq={self.model.nq} nv={self.model.nv} nu={self.model.nu}，"
                f"期望 nq={7 + n} nv={6 + n} nu={n}"
            )

        for i, name in enumerate(JOINT_ORDER):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                errors.append(f"缺少关节 {name}")
                continue
            if self.model.jnt_qposadr[jid] != 7 + i or self.model.jnt_dofadr[jid] != 6 + i:
                errors.append(
                    f"关节顺序不符: {name} qposadr={self.model.jnt_qposadr[jid]}，期望 {7 + i}"
                )
            jnt_range = self.model.jnt_range[jid]
            if not (
                np.isclose(jnt_range[0], cfg.joint_lower[i], atol=1e-9)
                and np.isclose(jnt_range[1], cfg.joint_upper[i], atol=1e-9)
            ):
                errors.append(
                    f"关节限位不符: {name} model={jnt_range}，config=({cfg.joint_lower[i]}, {cfg.joint_upper[i]})"
                )
            if self.model.nu > i:
                act_joint = self.model.actuator_trnid[i, 0]
                if act_joint != jid:
                    errors.append(f"执行器顺序不符: 第 {i} 个执行器未驱动 {name}")

        if errors:
            raise ValueError("duck.xml 与 DuckEnvConfig 契约不一致:\n" + "\n".join(errors))

    def _geom_id(self, name: str) -> int:
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"模型缺少必需 geom: {name}")
        return gid
