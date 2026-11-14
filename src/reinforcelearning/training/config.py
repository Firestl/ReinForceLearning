"""PPO 训练与评估的运行参数。

与 ``DuckEnvConfig`` 分离：环境契约描述物理世界，训练配置描述一次实验。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_PATH = Path("models/duck_ppo")
DEFAULT_LOG_DIR = Path("runs")


def vecnormalize_file_for(model_path: Path) -> Path:
    """由模型路径推导 ``VecNormalize`` 统计量文件路径。"""
    model_file = model_path if model_path.suffix == ".zip" else model_path.with_suffix(".zip")
    return model_file.with_name(f"{model_file.stem}_vecnormalize.pkl")


def _require_int(name: str, value: object, *, minimum: int) -> None:
    """校验整数字段；``bool`` 是 ``int`` 的子类，必须显式拒绝。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须为 int，实际 {value!r}")
    if value < minimum:
        raise ValueError(f"{name} 必须为 >= {minimum} 的整数，实际 {value!r}")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """一次 PPO 训练实验的规模与输出位置。"""

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
        """模型文件的最终路径：统一补上 SB3 使用的 ``.zip`` 后缀。"""
        if self.model_path.suffix == ".zip":
            return self.model_path
        return self.model_path.with_suffix(".zip")

    @property
    def vecnormalize_file(self) -> Path:
        """``VecNormalize`` 统计量文件：与模型同目录、同 stem 的 ``.pkl``。"""
        return vecnormalize_file_for(self.model_file)
