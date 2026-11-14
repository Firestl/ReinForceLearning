"""鸭子环境与奖励参数的集中配置。

本模块是 docs/02-physics-environment.md §11 所述配置项的唯一来源；
环境运行时会校验这里的契约值与 MJCF 模型一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass(frozen=True)
class RewardConfig:
    """奖励权重与形状参数；权重集中于此，不散落在 ``step()`` 中。"""

    tracking_weight: float = 1.0
    upright_weight: float = 0.5
    height_weight: float = 0.2
    alive_weight: float = 0.1
    lateral_weight: float = 0.05
    yaw_weight: float = 0.05
    energy_weight: float = 0.005
    action_rate_weight: float = 0.01
    joint_tracking_weight: float = 0.01

    speed_sigma: float = 0.5  # 速度跟踪项的高斯宽度 (m/s)
    height_sigma: float = 0.05  # 高度项的高斯宽度 (m)
    height_target: float = 0.233  # 默认站姿 root 高度 (m)


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
        if len(self.command_range) != 2 or self.command_range[0] > self.command_range[1]:
            raise ValueError(f"command_range 应为 (low, high) 且 low <= high，实际 {self.command_range}")

    @property
    def action_dim(self) -> int:
        return len(self.default_joint_pos)

    @property
    def obs_dim(self) -> int:
        # 3 基座线速度 + 3 基座角速度 + 3 重力投影 + 3*action_dim（关节角/角速度/上步动作）+ 1 指令
        return 10 + 3 * self.action_dim

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.frame_skip
