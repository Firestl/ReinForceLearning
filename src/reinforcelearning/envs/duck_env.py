"""鸭形双足机器人 Gymnasium 环境规范实现。

遵循 docs/duck/02-physics-environment.md 接口定义。
"""

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

# 数值发散时 MuJoCo 会自动将 qpos 归零重置；单靠检查有限性会漏报崩溃，需监控 warning 计数
_PHYSICS_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADCTRL,
)


@dataclass(frozen=True, slots=True)
class DuckState:
    """单步物理快照；观测、奖励与终止准则消费同一份不可变数据。"""

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
    """关键接触标志位。"""

    left: bool
    right: bool
    fall: bool


class DuckEnv(gymnasium.Env[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    """平地直线行走的鸭形双足环境（位置执行器驱动，本体感知观测）。"""

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}  # noqa: RUF012

    def __init__(
        self,
        config: DuckEnvConfig | None = None,
        *,
        render_mode: str | None = None,
    ) -> None:
        self.config = config if config is not None else DuckEnvConfig()
        cfg = self.config

        # 帧率与物理周期绑定：render_fps = 1 / control_dt
        self.metadata = {**type(self).metadata, "render_fps": round(1.0 / cfg.control_dt)}
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"不支持的 render_mode={render_mode!r}，可选 {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(cfg.model_path))
        self.model.opt.timestep = cfg.sim_dt  # sim_dt 覆盖 XML timestep，保证步长唯一来源
        self.data = mujoco.MjData(self.model)

        self._model_contract = DuckModelContract.from_model(self.model, cfg)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(cfg.action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(cfg.obs_dim,), dtype=np.float32
        )

        self._ctrl_low = np.asarray(cfg.action_target_low)
        self._ctrl_high = np.asarray(cfg.action_target_high)

        self._step_count = 0
        self._command = 0.0
        self._last_action = np.zeros(cfg.action_dim, dtype=np.float64)
        self._warning_count = self._physics_warning_count()
        self._viewer: Any = None
        self._renderer: Any = None

    # ------------------------------------------------------------------ 核心生命周期

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        cfg = self.config
        _ = options

        mujoco.mj_resetData(self.model, self.data)

        # 随机偏航角初始位姿
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[0:3] = (0.0, 0.0, cfg.spawn_height)
        self.data.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))

        # 关节注入初始化噪声
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
        return self._get_obs(state), self._build_info(state, contacts)

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        cfg = self.config
        action_vec = self._sanitize_action(action)
        self._apply_action(action_vec)

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

    # ------------------------------------------------------------------ 动作与状态变换

    def _sanitize_action(self, action: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """校验并裁剪归一化动作输入向量。"""
        vec = np.asarray(action, dtype=np.float64).reshape(-1)
        if vec.shape != (self.config.action_dim,):
            raise ValueError(f"动作维度应为 {self.config.action_dim}，实际 shape={vec.shape}")
        if not np.isfinite(vec).all():
            raise ValueError("动作含非有限值 (NaN/Inf)，拒绝执行")
        return np.clip(vec, -1.0, 1.0)

    def _apply_action(self, action_vec: npt.NDArray[np.float64]) -> None:
        """映射归一化动作至位置执行器控制信号并施加安全截断。"""
        cfg = self.config
        q_target = cfg.default_joint_pos + cfg.action_scale * action_vec
        self.data.ctrl[:] = np.clip(q_target, self._ctrl_low, self._ctrl_high)

    def _current_state(self) -> DuckState:
        """提取动力学快照；观测、奖励与终止判断共享此单步数据。"""
        rotation = self.data.xmat[self._model_contract.torso_body].reshape(3, 3)
        lin_vel = rotation.T @ self.data.qvel[0:3]  # 世界坐标系 -> 机体坐标系
        ang_vel = self.data.qvel[3:6].copy()       # freejoint 角速度原生即为机体坐标系
        projected_gravity = -rotation[2, :]        # R^T @ (0, 0, -1)
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

    # ------------------------------------------------------------------ 奖励与终止判定

    def _compute_reward(
        self, action: npt.NDArray[np.float64], state: DuckState
    ) -> tuple[float, RewardTerms]:
        rc = self.config.reward
        speed_error = float(state.lin_vel[0]) - self._command
        tracking = float(np.exp(-((speed_error / rc.speed_sigma) ** 2)))
        upright = float(np.clip(state.body_up_z, 0.0, 1.0))
        height_term = float(
            np.exp(-(((state.body_height - rc.height_target) / rc.height_sigma) ** 2))
        )
        joint_tracking_error = state.joint_pos - state.ctrl

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

    # ------------------------------------------------------------------ 接触与诊断信息

    def _physics_warning_count(self) -> int:
        return int(sum(self.data.warning[w].number for w in _PHYSICS_WARNINGS))

    def _contact_flags(self) -> ContactFlags:
        """判定左脚、右脚及躯干/头部的触地接触状态。"""
        contract = self._model_contract
        left = right = fall = False
        for i in range(self.data.ncon):
            pair = {self.data.contact[i].geom1, self.data.contact[i].geom2}
            if contract.floor_geom not in pair:
                continue  # 鸭体自碰撞已由 contype/conaffinity 掩码隔离
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
