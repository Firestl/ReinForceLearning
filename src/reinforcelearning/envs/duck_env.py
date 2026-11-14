"""鸭形机器人 Gymnasium 环境，实现规范见 docs/duck/02-physics-environment.md。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import gymnasium
import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from reinforcelearning.envs.config import (
    REWARD_TERM_KEYS,
    DuckEnvConfig,
    RewardTerm,
    RewardTerms,
)
from reinforcelearning.envs.model_contract import DuckModelContract

# 表示物理状态非有限的 MuJoCo 警告；数值异常时 MuJoCo 会自动重置 qpos，
# 因此事后检查 qpos 的有限性不够，需要监视警告计数
_PHYSICS_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADCTRL,
)


@dataclass(frozen=True, slots=True)
class DuckState:
    """单步物理状态快照；观测、奖励与终止都消费同一份数据。"""

    lin_vel: np.ndarray
    ang_vel: np.ndarray
    projected_gravity: np.ndarray
    body_up_z: float
    tilt_angle: float
    body_height: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    ctrl: np.ndarray
    actuator_force: np.ndarray
    physics_finite: bool
    warning_count: int


class ContactFlags(NamedTuple):
    """接触检测结果（左脚、右脚、躯干/头部触地）。"""

    left: bool
    right: bool
    fall: bool


class DuckEnv(gymnasium.Env[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    """平地直线行走的鸭形双足环境（位置执行器、本体感知观测）。"""

    # 类级模板，__init__ 中合并为实例副本；模板本身从不被原地修改。
    # gymnasium.Env 基类契约即如此声明（core.py 的 metadata），故豁免 RUF012。
    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}  # noqa: RUF012

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

        # 名称解析、维度/关节/执行器校验与接触 id 统一由模型契约负责
        self._model_contract = DuckModelContract.from_model(self.model, cfg)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(cfg.action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(cfg.obs_dim,), dtype=np.float32
        )

        # 动作映射后的可达目标区间；也是写入 ctrl 前的裁剪边界
        self._ctrl_low = np.asarray(cfg.action_target_low)
        self._ctrl_high = np.asarray(cfg.action_target_high)

        self._step_count = 0
        self._command = 0.0
        self._last_action = np.zeros(cfg.action_dim, dtype=np.float64)
        self._warning_count = self._physics_warning_count()
        self._viewer: Any = None
        self._renderer: Any = None

    # ------------------------------------------------------------------ 核心接口

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        cfg = self.config
        _ = options  # 第一版不使用 reset 选项

        mujoco.mj_resetData(self.model, self.data)

        # 根节点：初始站立高度 + 随机朝向（绕 z 轴 ±pi）
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[0:3] = 0.0
        self.data.qpos[2] = cfg.spawn_height
        self.data.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))

        # 关节默认姿态 + 小幅噪声；速度置零 + 极小噪声
        qpos_adr = self._model_contract.qpos_adr
        self.data.qpos[qpos_adr] = cfg.default_joint_pos
        self.data.qpos[qpos_adr] += self.np_random.uniform(
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
        state = self._current_state()
        contacts = self._contact_flags()
        info = self._build_info(state=state, contacts=contacts)
        return self._get_obs(state), info

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        cfg = self.config
        # 动作空间契约是 float32；内部统一转为 float64 平面向量（MuJoCo 数据为 float64）
        action_vec = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_vec.shape != (cfg.action_dim,):
            raise ValueError(f"动作维度应为 {cfg.action_dim}，实际 shape={action_vec.shape}")
        if not np.isfinite(action_vec).all():
            # fail-fast：NaN/Inf 进入 ctrl 会产生 NaN 奖励/观测并污染训练 buffer
            raise ValueError("action 含 NaN/Inf，拒绝执行")
        action_vec = np.clip(action_vec, -1.0, 1.0)

        # 归一化动作 -> 目标关节角 -> 限位内裁剪 -> 位置执行器
        q_target = cfg.default_joint_pos + cfg.action_scale * action_vec
        self.data.ctrl[:] = np.clip(q_target, self._ctrl_low, self._ctrl_high)

        for _ in range(cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        state = self._current_state()
        contacts = self._contact_flags()
        reward, reward_terms = self._compute_reward(action_vec, state)
        terminated, reason = self._check_termination(state, contacts)
        truncated = (not terminated) and self._step_count >= cfg.max_episode_steps
        if truncated:
            reason = "timeout"

        self._last_action = action_vec
        obs = self._get_obs(state)
        info = self._build_info(
            state=state, contacts=contacts, reward_terms=reward_terms, termination_reason=reason
        )
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

    def _current_state(self) -> DuckState:
        """从 MjData 提取单步快照；观测、奖励与终止只消费这一份数据。"""
        rotation = self.data.xmat[self._model_contract.torso_body].reshape(3, 3)
        lin_vel = rotation.T @ self.data.qvel[0:3]  # 世界系 -> 身体系
        ang_vel = self.data.qvel[3:6].copy()  # free joint 角速度本身就是 body 系，不可再旋转
        projected_gravity = -rotation[2, :]  # R^T @ (0, 0, -1)
        body_up_z = -projected_gravity[2]
        tilt = float(np.arccos(np.clip(body_up_z, -1.0, 1.0)))
        return DuckState(
            lin_vel=lin_vel,
            ang_vel=ang_vel,
            projected_gravity=projected_gravity,
            body_up_z=float(body_up_z),
            tilt_angle=tilt,
            body_height=float(self.data.qpos[2]),
            joint_pos=self.data.qpos[self._model_contract.qpos_adr].copy(),
            joint_vel=self.data.qvel[self._model_contract.dof_adr].copy(),
            ctrl=self.data.ctrl.copy(),
            actuator_force=self.data.actuator_force.copy(),
            physics_finite=bool(
                np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
            ),
            warning_count=self._physics_warning_count(),
        )

    def _get_obs(self, state: DuckState) -> npt.NDArray[np.float32]:
        return np.concatenate(
            [
                state.lin_vel,
                state.ang_vel,
                state.projected_gravity,
                state.joint_pos,
                state.joint_vel,
                self._last_action,
                np.array([self._command]),
            ]
        ).astype(np.float32)

    # ------------------------------------------------------------------ 奖励与终止

    def _compute_reward(
        self, action: npt.NDArray[np.float64], state: DuckState
    ) -> tuple[float, RewardTerms]:
        cfg = self.config
        rc = cfg.reward

        speed_error = float(state.lin_vel[0]) - self._command
        tracking = float(np.exp(-((speed_error / rc.speed_sigma) ** 2)))
        upright = float(np.clip(state.body_up_z, 0.0, 1.0))
        height_term = float(
            np.exp(-(((state.body_height - rc.height_target) / rc.height_sigma) ** 2))
        )
        joint_tracking_error = state.joint_pos - state.ctrl

        # 未加权的分项值；键集与权重字段名均由 REWARD_TERM_KEYS 派生
        reward_values: dict[RewardTerm, float] = {
            "tracking": tracking,
            "upright": upright,
            "height_reward": height_term,
            "alive": 1.0,
            "lateral": -float(state.lin_vel[1] ** 2),
            "yaw": -float(state.ang_vel[2] ** 2),
            "energy": -float(np.sum(np.square(state.actuator_force))),
            "action_rate": -float(np.sum(np.square(action - self._last_action))),
            "joint_tracking": -float(np.sum(np.square(joint_tracking_error))),
        }
        terms: RewardTerms = {
            term: float(getattr(rc, f"{term}_weight")) * reward_values[term]
            for term in REWARD_TERM_KEYS
        }
        return float(sum(terms.values())), terms

    def _check_termination(
        self, state: DuckState, contacts: ContactFlags
    ) -> tuple[bool, str | None]:
        if state.warning_count > self._warning_count or not state.physics_finite:
            return True, "non_finite"
        if contacts.fall:
            return True, "fall_contact"
        if state.tilt_angle > self.config.max_tilt:
            return True, "excessive_tilt"
        if state.body_height < self.config.fall_height:
            return True, "too_low"
        return False, None

    # ------------------------------------------------------------------ 接触与 info

    def _physics_warning_count(self) -> int:
        """累计的物理状态警告数（BADQPOS/BADQVEL/BADQACC/BADCTRL）。"""
        return int(sum(self.data.warning[w].number for w in _PHYSICS_WARNINGS))

    def _contact_flags(self) -> ContactFlags:
        """返回左脚/右脚/躯干头部触地标志。"""
        contract = self._model_contract
        left = right = fall = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {contact.geom1, contact.geom2}
            if contract.floor_geom not in pair:
                continue  # 自碰撞已通过 contype/conaffinity 关闭
            other = next(g for g in pair if g != contract.floor_geom)
            if other == contract.left_foot_geom:
                left = True
            elif other == contract.right_foot_geom:
                right = True
            elif other in (contract.torso_geom, contract.head_geom):
                fall = True
        return ContactFlags(left, right, fall)

    def _build_info(
        self,
        state: DuckState,
        contacts: ContactFlags,
        reward_terms: RewardTerms | None = None,
        termination_reason: str | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "forward_speed": float(state.lin_vel[0]),
            "lateral_speed": float(state.lin_vel[1]),
            "body_height": state.body_height,
            "tilt_angle": state.tilt_angle,
            "command_speed": self._command,
            "left_foot_contact": contacts.left,
            "right_foot_contact": contacts.right,
            "termination_reason": termination_reason,
        }
        if reward_terms is not None:
            info["reward_terms"] = {key: float(reward_terms[key]) for key in REWARD_TERM_KEYS}
        return info
