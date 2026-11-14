"""鸭子环境与奖励参数的集中配置。

本模块是 docs/02-physics-environment.md §11 所述配置项的唯一来源；
环境运行时会校验这里的契约值与 MJCF 模型一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal, get_args

# 模型资产路径：基于 __file__ 解析，不依赖当前工作目录
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_MODEL_PATH = ASSETS_DIR / "duck.xml"

# 固定关节顺序：观测、动作、q_default/action_scale 与 XML 执行器顺序共用此契约
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

# 关节轴契约：roll 绕 x 轴，pitch 绕 y 轴（顺序与 JOINT_ORDER 一致）
JOINT_AXES: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),  # hip roll
    (0.0, 1.0, 0.0),  # hip pitch
    (0.0, 1.0, 0.0),  # knee pitch
    (0.0, 1.0, 0.0),  # ankle pitch
) * 2


# 奖励分项键的唯一来源：顺序即 info["reward_terms"] 的顺序，权重字段由 "<term>_weight" 约定派生
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
    """奖励权重与形状参数；权重集中于此，不散落在 ``step()`` 中。"""

    tracking_weight: float = 1.0
    upright_weight: float = 0.5
    height_reward_weight: float = 0.2
    alive_weight: float = 0.1
    lateral_weight: float = 0.05
    yaw_weight: float = 0.05
    energy_weight: float = 0.005
    action_rate_weight: float = 0.01
    joint_tracking_weight: float = 0.01

    speed_sigma: float = 0.5  # 速度跟踪项的高斯宽度 (m/s)
    height_sigma: float = 0.05  # 高度项的高斯宽度 (m)
    height_target: float = 0.233  # 默认站姿 root 高度 (m)

    def __post_init__(self) -> None:
        # 权重字段必须与奖励分项键一一对应，避免 info["reward_terms"] 静默缺项
        weight_names = {f.name for f in fields(self) if f.name.endswith("_weight")}
        expected_names = {f"{term}_weight" for term in REWARD_TERM_KEYS}
        if weight_names != expected_names:
            raise ValueError(
                "RewardConfig 权重字段与奖励分项不一致: "
                f"多余={sorted(weight_names - expected_names)}，"
                f"缺失={sorted(expected_names - weight_names)}"
            )

        # 权重为奖励分项的系数；NaN/Inf 会静默污染奖励与训练 buffer
        for term in REWARD_TERM_KEYS:
            name = f"{term}_weight"
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} 必须为有限数，实际 {value}")

        if not math.isfinite(self.height_target):
            raise ValueError(f"height_target 必须为有限数，实际 {self.height_target}")
        for name in ("speed_sigma", "height_sigma"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                # sigma <= 0 会在奖励中产生除零/NaN
                raise ValueError(f"{name} 必须为正有限数，实际 {value}")


@dataclass(frozen=True)
class DuckEnvConfig:
    """DuckEnv 的运行参数。字段含义见 docs/02-physics-environment.md。"""

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

    spawn_height: float = 0.233  # 重置时的初始 root 高度；与奖励的 height_target 解耦
    fall_height: float = 0.15
    max_tilt: float = math.radians(60.0)

    reset_joint_noise: float = 0.01
    reset_velocity_noise: float = 0.01

    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        n = len(JOINT_ORDER)
        sizes = {
            "default_joint_pos": self.default_joint_pos,
            "action_scale": self.action_scale,
            "joint_lower": self.joint_lower,
            "joint_upper": self.joint_upper,
        }
        for name, value in sizes.items():
            if len(value) != n:
                raise ValueError(f"{name} 应有 {n} 项，实际 {len(value)} 项")
        if not all(math.isfinite(v) and v > 0 for v in self.action_scale):
            raise ValueError(f"action_scale 各元素必须为正有限数，实际 {self.action_scale}")

        # 关节数组契约：限位区间非退化，默认姿态落在限位内
        if not all(math.isfinite(q) for q in self.default_joint_pos):
            raise ValueError(f"default_joint_pos 各元素必须为有限数，实际 {self.default_joint_pos}")
        for i, name in enumerate(JOINT_ORDER):
            lower, upper = self.joint_lower[i], self.joint_upper[i]
            if not (math.isfinite(lower) and math.isfinite(upper)) or lower >= upper:
                raise ValueError(
                    f"joint_lower/joint_upper 第 {i} 项（{name}）应为有限数且 lower < upper，"
                    f"实际 ({lower}, {upper})"
                )
            q = self.default_joint_pos[i]
            if not lower <= q <= upper:
                raise ValueError(
                    f"default_joint_pos 第 {i} 项（{name}）应在关节限位 "
                    f"[{lower}, {upper}] 内，实际 {q}"
                )

        # 时间参数：sim_dt 是物理步长唯一来源，非法值会导致除零或负步长
        if not math.isfinite(self.sim_dt) or self.sim_dt <= 0:
            raise ValueError(f"sim_dt 必须为正有限数，实际 {self.sim_dt}")
        if not isinstance(self.frame_skip, int) or self.frame_skip < 1:
            raise ValueError(f"frame_skip 必须为 >= 1 的整数，实际 {self.frame_skip!r}")
        if not isinstance(self.max_episode_steps, int) or self.max_episode_steps < 1:
            raise ValueError(
                f"max_episode_steps 必须为 >= 1 的整数，实际 {self.max_episode_steps!r}"
            )

        for name, value in (
            ("joint_limits_margin", self.joint_limits_margin),
            ("reset_joint_noise", self.reset_joint_noise),
            ("reset_velocity_noise", self.reset_velocity_noise),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须为 >= 0 的有限数，实际 {value}")

        # 动作裁剪区间为 [lower + margin, upper - margin]，必须非空
        for i, name in enumerate(JOINT_ORDER):
            low = self.joint_lower[i] + self.joint_limits_margin
            high = self.joint_upper[i] - self.joint_limits_margin
            if low > high:
                raise ValueError(
                    f"joint_limits_margin({self.joint_limits_margin}) 使第 {i} 项（{name}）"
                    f"裁剪区间为空: [{low}, {high}]"
                )

        # 动作映射后的可达目标区间 = [default ± action_scale] ∩ 上述裁剪区间
        target_low, target_high = self.action_target_low, self.action_target_high
        for i, name in enumerate(JOINT_ORDER):
            if target_low[i] > target_high[i]:
                raise ValueError(
                    f"default_joint_pos/action_scale 使第 {i} 项（{name}）动作目标区间为空: "
                    f"[{target_low[i]}, {target_high[i]}]"
                )

        for name, value in (("spawn_height", self.spawn_height), ("fall_height", self.fall_height)):
            if not math.isfinite(value):
                raise ValueError(f"{name} 必须为有限数，实际 {value}")
        if self.spawn_height <= self.fall_height:
            raise ValueError(
                f"spawn_height({self.spawn_height}) 必须大于 fall_height({self.fall_height})，"
                "否则新回合会立即触发 too_low 终止"
            )
        # tilt_angle 在 [0, pi]；max_tilt <= 0 会让回合在重置后立即触发 excessive_tilt
        if not math.isfinite(self.max_tilt) or self.max_tilt <= 0:
            raise ValueError(f"max_tilt 必须为正有限数，实际 {self.max_tilt}")

        if len(self.command_range) != 2 or not all(math.isfinite(v) for v in self.command_range):
            raise ValueError(f"command_range 应为两个有限数 (low, high)，实际 {self.command_range}")
        if self.command_range[0] > self.command_range[1]:
            raise ValueError(f"command_range 应满足 low <= high，实际 {self.command_range}")

    @property
    def action_dim(self) -> int:
        return len(self.default_joint_pos)

    @property
    def action_target_low(self) -> tuple[float, ...]:
        """动作映射后可写入 ctrl 的目标角下限（default - scale 与限位内收缩的交集）。"""
        return tuple(
            max(default - scale, lower + self.joint_limits_margin)
            for default, scale, lower in zip(
                self.default_joint_pos, self.action_scale, self.joint_lower, strict=True
            )
        )

    @property
    def action_target_high(self) -> tuple[float, ...]:
        """动作映射后可写入 ctrl 的目标角上限（default + scale 与限位内收缩的交集）。"""
        return tuple(
            min(default + scale, upper - self.joint_limits_margin)
            for default, scale, upper in zip(
                self.default_joint_pos, self.action_scale, self.joint_upper, strict=True
            )
        )

    @property
    def obs_dim(self) -> int:
        # 3 基座线速度 + 3 基座角速度 + 3 重力投影 + 3*action_dim（关节角/角速度/上步动作）+ 1 指令
        return 10 + 3 * self.action_dim

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.frame_skip
