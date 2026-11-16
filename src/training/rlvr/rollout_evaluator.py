"""离线 rollout reward 查询器。

设计目标是把 golden generation 生成好的 all_rollout_rewards 字段，统一封装成一个
lookup 接口，供 verifier 在训练时直接查询，而不是在线再跑 SUMO。
"""

from __future__ import annotations

from typing import Any, Dict


class RolloutLookupError(KeyError):
    """候选动作 reward 查找失败。"""


class RolloutEvaluator:
    """封装单个样本的 rollout reward 查找逻辑。"""

    def __init__(self, sample: Dict[str, Any]):
        self.sample = sample or {}
        self.all_rollout_rewards = self.sample.get("all_rollout_rewards", {}) or {}

    def lookup(self, phase_id: int, duration: int) -> Dict[str, Any]:
        """返回指定动作的 rollout reward 结果。

        期望 all_rollout_rewards 的键形如 "{phase}_{duration}"。若找不到候选，直接抛
        RolloutLookupError，由 verifier 统一兜底为 -3.0。
        """
        key = f"{int(phase_id)}_{int(duration)}"
        payload = self.all_rollout_rewards.get(key)
        if payload is None:
            raise RolloutLookupError(key)

        if isinstance(payload, dict):
            return payload
        return {"r_traffic": float(payload)}
