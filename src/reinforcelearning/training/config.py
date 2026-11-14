"""PPO 训练与评估的运行参数。

与 ``DuckEnvConfig`` 分离：环境契约描述物理世界，训练配置描述一次实验。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_PATH = Path("models/duck_ppo")
DEFAULT_LOG_DIR = Path("runs")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """一次 PPO 训练实验的规模与输出位置。"""

    total_timesteps: int = 100_000
    n_envs: int = 4
    seed: int = 0
    model_path: Path = DEFAULT_MODEL_PATH
    log_dir: Path | None = DEFAULT_LOG_DIR
    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        if self.log_dir is not None:
            object.__setattr__(self, "log_dir", Path(self.log_dir))

        if not isinstance(self.total_timesteps, int) or self.total_timesteps < 1:
            raise ValueError(f"total_timesteps 必须为 >= 1 的整数，实际 {self.total_timesteps!r}")
        if not isinstance(self.n_envs, int) or self.n_envs < 1:
            raise ValueError(f"n_envs 必须为 >= 1 的整数，实际 {self.n_envs!r}")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed 必须为 >= 0 的整数，实际 {self.seed!r}")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError(f"device 必须为非空字符串，实际 {self.device!r}")

    @property
    def model_file(self) -> Path:
        """模型文件的最终路径：统一补上 SB3 使用的 ``.zip`` 后缀。"""
        if self.model_path.suffix == ".zip":
            return self.model_path
        return self.model_path.with_suffix(".zip")
