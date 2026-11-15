"""PPO 训练与评估参数配置。

与物理环境契约 (DuckEnvConfig) 解耦：环境定义物理规则，本配置定义算法实验环境。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_PATH = Path("models/duck_ppo")
DEFAULT_LOG_DIR = Path("runs")


def vecnormalize_file_for(model_path: Path) -> Path:
    """推导与模型同名配对的 VecNormalize 统计量文件路径 (*_vecnormalize.pkl)。"""
    model_file = model_path if model_path.suffix == ".zip" else model_path.with_suffix(".zip")
    return model_file.with_name(f"{model_file.stem}_vecnormalize.pkl")


def _require_int(name: str, value: object, *, minimum: int) -> None:
    # 严格类型防护：bool 是 int 的子类，必须显式拦截避免漏判
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须为 int，实际 {value!r}")
    if value < minimum:
        raise ValueError(f"{name} 必须为 >= {minimum} 的整数，实际 {value!r}")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """PPO 算法实验超参数与输出路径配置。"""

    total_timesteps: int = 100_000
    n_envs: int = 4
    seed: int = 0
    model_path: Path = DEFAULT_MODEL_PATH
    log_dir: Path | None = DEFAULT_LOG_DIR
    device: str = "cpu"
    gamma: float = 0.99
    normalize: bool = True
    norm_reward: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        if self.log_dir is not None:
            object.__setattr__(self, "log_dir", Path(self.log_dir))

        _require_int("total_timesteps", self.total_timesteps, minimum=1)
        _require_int("n_envs", self.n_envs, minimum=1)
        _require_int("seed", self.seed, minimum=0)

        if not isinstance(self.device, str):
            raise TypeError(f"device 必须为 str，实际 {self.device!r}")
        if not self.device:
            raise ValueError(f"device 必须为非空字符串，实际 {self.device!r}")

        if isinstance(self.gamma, bool) or not isinstance(self.gamma, (int, float)):
            raise TypeError(f"gamma 必须为数值，实际 {self.gamma!r}")
        if not math.isfinite(self.gamma) or not 0.0 < self.gamma < 1.0:
            raise ValueError(f"gamma 必须位于 (0, 1)，实际 {self.gamma!r}")

        if not isinstance(self.normalize, bool):
            raise TypeError(f"normalize 必须为 bool，实际 {self.normalize!r}")
        if not isinstance(self.norm_reward, bool):
            raise TypeError(f"norm_reward 必须为 bool，实际 {self.norm_reward!r}")

    @property
    def model_file(self) -> Path:
        """模型最终持久化路径（确保具备 .zip 后缀）。"""
        return self.model_path if self.model_path.suffix == ".zip" else self.model_path.with_suffix(".zip")

    @property
    def vecnormalize_file(self) -> Path:
        """与模型同名配对的 VecNormalize 统计量文件路径。"""
        return vecnormalize_file_for(self.model_file)
