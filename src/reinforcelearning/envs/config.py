"""鸭形机器人环境配置与动力学契约唯一来源。

本模块统领所有物理参数、接口维度与奖励权重。
文档（docs/duck/）提供物理意图，此处提供不可变参数事实源。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal, get_args

# 物理模型资产路径（基于 __file__ 绝对解析）
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_MODEL_PATH = ASSETS_DIR / "duck.xml"

# 全局固定关节拓扑契约：左腿 4 自由度 + 右腿 4 自由度
JOINT_ORDER: tuple[str, ...] = (
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_pitch",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_pitch",
)

JOINT_LOWER: tuple[float, ...] = (-0.35, -1.30, 0.00, -0.90, -0.35, -1.30, 0.00, -0.90)
JOINT_UPPER: tuple[float, ...] = (0.35, 0.80, 2.20, 0.60, 0.35, 0.80, 2.20, 0.60)

# 关节旋转轴定义：hip_roll 绕 x 轴，其余均绕 y 轴
JOINT_AXES: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
) * 2

# 奖励分项键集合：排序决定 info["reward_terms"] 的输出顺序
RewardTerm = Literal[
    "tracking",
    "upright",
    "height_reward",
    "alive",
    "lateral",
    "yaw",
    "energy",
    "action_rate",
    "joint_tracking",
]
REWARD_TERM_KEYS: tuple[RewardTerm, ...] = get_args(RewardTerm)
RewardTerms = dict[RewardTerm, float]


@dataclass(frozen=True)
class RewardConfig:
    """奖励权重与形状参数配置；集中管理，禁止在仿真循环中硬编码。"""

    tracking_weight: float = 1.0
    upright_weight: float = 0.5
    height_reward_weight: float = 0.2
    alive_weight: float = 0.1
    lateral_weight: float = 0.05
    yaw_weight: float = 0.05
    energy_weight: float = 0.005
    action_rate_weight: float = 0.01
    joint_tracking_weight: float = 0.01

    speed_sigma: float = 0.5    # 速度跟踪高斯宽度 (m/s)
    height_sigma: float = 0.05  # 站立高度高斯宽度 (m)
    height_target: float = 0.233  # 默认站姿 root 高度 (m)

    def __post_init__(self) -> None:
        # 强制字段与分项严格同构，避免静默缺项
        weight_names = {f.name for f in fields(self) if f.name.endswith("_weight")}
        expected_names = {f"{term}_weight" for term in REWARD_TERM_KEYS}
        if weight_names != expected_names:
            raise ValueError(
                f"权重字段不匹配: 多余={sorted(weight_names - expected_names)}, "
                f"缺失={sorted(expected_names - weight_names)}"
            )

        for term in REWARD_TERM_KEYS:
            val = getattr(self, f"{term}_weight")
            if not math.isfinite(val):
                raise ValueError(f"{term}_weight 必须为有限数，实际 {val}")

        if not math.isfinite(self.height_target):
            raise ValueError(f"height_target 必须为有限数，实际 {self.height_target}")
        for name in ("speed_sigma", "height_sigma"):
            val = getattr(self, name)
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f"{name} 必须为正有限数，实际 {val}")


@dataclass(frozen=True)
class DuckEnvConfig:
    """DuckEnv 动力学与控制参数配置。"""

    model_path: Path = DEFAULT_MODEL_PATH
    sim_dt: float = 0.002
    frame_skip: int = 10
    max_episode_steps: int = 1000

    default_joint_pos: tuple[float, ...] = (0.0, -0.30, 0.60, -0.30) * 2
    action_scale: tuple[float, ...] = (0.20, 0.35, 0.50, 0.25) * 2
    joint_lower: tuple[float, ...] = JOINT_LOWER
    joint_upper: tuple[float, ...] = JOINT_UPPER
    joint_limits_margin: float = 0.02

    command_range: tuple[float, float] = (0.0, 1.0)
    spawn_height: float = 0.233  # 重置 root 高度；与奖励 target 独立解耦
    fall_height: float = 0.15
    max_tilt: float = math.radians(60.0)

    reset_joint_noise: float = 0.01
    reset_velocity_noise: float = 0.01

    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        n = len(JOINT_ORDER)

        # 1. 数组维度与正定性校验
        for name, arr in (
            ("default_joint_pos", self.default_joint_pos),
            ("action_scale", self.action_scale),
            ("joint_lower", self.joint_lower),
            ("joint_upper", self.joint_upper),
        ):
            if len(arr) != n:
                raise ValueError(f"{name} 长度应为 {n}，实际 {len(arr)}")

        if not all(math.isfinite(v) and v > 0 for v in self.action_scale):
            raise ValueError(f"action_scale 元素必须为正有限数，实际 {self.action_scale}")

        # 2. 关节限位区间与初始姿态合法性
        for i, name in enumerate(JOINT_ORDER):
            lo, hi = self.joint_lower[i], self.joint_upper[i]
            q0 = self.default_joint_pos[i]
            if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
                raise ValueError(f"{name} 限位区间无效: ({lo}, {hi})")
            if not (lo <= q0 <= hi):
                raise ValueError(f"{name} 默认姿态 {q0} 超出限位 [{lo}, {hi}]")

        # 3. 时间与控制参数
        if not math.isfinite(self.sim_dt) or self.sim_dt <= 0:
            raise ValueError(f"sim_dt 必须为正有限数，实际 {self.sim_dt}")
        for name, val in (
            ("frame_skip", self.frame_skip),
            ("max_episode_steps", self.max_episode_steps),
        ):
            if not isinstance(val, int) or val < 1:
                raise ValueError(f"{name} 必须为正整数，实际 {val!r}")

        for name, val in (
            ("joint_limits_margin", self.joint_limits_margin),
            ("reset_joint_noise", self.reset_joint_noise),
            ("reset_velocity_noise", self.reset_velocity_noise),
        ):
            if not math.isfinite(val) or val < 0:
                raise ValueError(f"{name} 必须为非负有限数，实际 {val}")

        # 4. 动作有效控制区间非空校验
        for i, name in enumerate(JOINT_ORDER):
            clip_lo = self.joint_lower[i] + self.joint_limits_margin
            clip_hi = self.joint_upper[i] - self.joint_limits_margin
            if clip_lo > clip_hi:
                raise ValueError(f"joint_limits_margin 导致 {name} 裁剪区间为空: [{clip_lo}, {clip_hi}]")

        for i, name in enumerate(JOINT_ORDER):
            if self.action_target_low[i] > self.action_target_high[i]:
                raise ValueError(f"{name} 动作目标区间为空: [{self.action_target_low[i]}, {self.action_target_high[i]}]")

        # 5. 几何高度与姿态阈值
        if not (math.isfinite(self.spawn_height) and math.isfinite(self.fall_height)):
            raise ValueError("spawn_height 与 fall_height 必须为有限数")
        if self.spawn_height <= self.fall_height:
            raise ValueError(f"spawn_height({self.spawn_height}) 必须大于 fall_height({self.fall_height})")

        if not math.isfinite(self.max_tilt) or self.max_tilt <= 0:
            raise ValueError(f"max_tilt 必须为正有限数，实际 {self.max_tilt}")

        if len(self.command_range) != 2 or not all(math.isfinite(v) for v in self.command_range):
            raise ValueError(f"command_range 必须为包含两个有限数的元组，实际 {self.command_range}")
        if self.command_range[0] > self.command_range[1]:
            raise ValueError(f"command_range 必须满足 min <= max，实际 {self.command_range}")

    @property
    def action_dim(self) -> int:
        return len(self.default_joint_pos)

    @property
    def action_target_low(self) -> tuple[float, ...]:
        """名义目标角下界 (default - scale) 与限位内收缩边界的交集。"""
        return tuple(
            max(q0 - s, lo + self.joint_limits_margin)
            for q0, s, lo in zip(self.default_joint_pos, self.action_scale, self.joint_lower, strict=True)
        )

    @property
    def action_target_high(self) -> tuple[float, ...]:
        """名义目标角上界 (default + scale) 与限位内收缩边界的交集。"""
        return tuple(
            min(q0 + s, hi - self.joint_limits_margin)
            for q0, s, hi in zip(self.default_joint_pos, self.action_scale, self.joint_upper, strict=True)
        )

    @property
    def obs_dim(self) -> int:
        # 3(线速度) + 3(角速度) + 3(重力投影) + 3*action_dim(位置/速度/上步动作) + 1(目标速度指令)
        return 10 + 3 * self.action_dim

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.frame_skip
