"""环境子包：导入即注册 ``Duck-v0``。"""

from __future__ import annotations

import gymnasium

from reinforcelearning.envs.config import DuckEnvConfig, RewardConfig
from reinforcelearning.envs.duck_env import DuckEnv

__all__ = ["DuckEnv", "DuckEnvConfig", "RewardConfig"]

_ENV_ID = "Duck-v0"
if _ENV_ID not in gymnasium.registry:
    gymnasium.register(id=_ENV_ID, entry_point="reinforcelearning.envs:DuckEnv")
