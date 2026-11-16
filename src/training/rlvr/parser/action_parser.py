"""动作解析器。

这里的目标不是“尽量猜一个动作”，而是把 prompt / CoT 里真正写出来的 action
严格抽出来。只要模型没有给出可解析动作，就直接抛异常，由 verifier 走
R_GRPO = -3.0 的工程兜底。
"""

from __future__ import annotations

import json
import re
from typing import Tuple

# 直接定义候选集，与 tsc_wrapper.GREEN_DURATION_CANDIDATES 保持一致，
# 避免 import tsc_wrapper 时触发 gymnasium 等重型依赖，使 reward 模块可脱机测试。
GREEN_DURATION_CANDIDATES = [15, 20, 25, 30, 35, 40]


class ActionParseError(ValueError):
    """动作解析失败时抛出的异常。"""


def _nearest_duration(duration: int) -> int:
    """把任意 duration 吸附到最近的合法候选值。"""
    return min(GREEN_DURATION_CANDIDATES, key=lambda candidate: abs(candidate - duration))


def parse_action(response_text: str, num_phases: int | None = None) -> Tuple[int, int]:
    """解析 VLM 输出中的动作。

    支持三类写法：
    1. Action: {"phase": X, "duration": Y}
    2. Action: phase=X, duration=Y / phase: X, duration: Y
    3. 兼容旧式 Action: X

    只要 phase 无法解析、phase 越界，或者 response_text 根本不包含有效动作，
    就抛 ActionParseError。
    """
    if not isinstance(response_text, str) or not response_text.strip():
        raise ActionParseError("empty_response")

    text = response_text.strip()
    default_duration = GREEN_DURATION_CANDIDATES[len(GREEN_DURATION_CANDIDATES) // 2]

    json_match = re.search(r"Action\s*:?[\s\n]*(\{.*?\})", text, re.IGNORECASE | re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(1))
            phase_id = int(payload.get("phase", payload.get("phase_id")))
            duration = _nearest_duration(int(payload.get("duration", default_duration)))
            if phase_id < 0 or (num_phases is not None and phase_id >= num_phases):
                raise ActionParseError(f"phase_out_of_range:{phase_id}")
            return phase_id, duration
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ActionParseError(f"invalid_json_action:{exc}") from exc

    kv_match = re.search(
        r"Action:.*?[\"']?phase(?:_id)?[\"']?\s*[:=]\s*(\d+).*?[\"']?duration[\"']?\s*[:=]\s*(\d+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if kv_match:
        phase_id = int(kv_match.group(1))
        duration = _nearest_duration(int(kv_match.group(2)))
        if phase_id < 0 or (num_phases is not None and phase_id >= num_phases):
            raise ActionParseError(f"phase_out_of_range:{phase_id}")
        return phase_id, duration

    old_match = re.search(r"Action:?\s*\[?(\d+)\]?", text, re.IGNORECASE)
    if old_match:
        phase_id = int(old_match.group(1))
        if phase_id < 0 or (num_phases is not None and phase_id >= num_phases):
            raise ActionParseError(f"phase_out_of_range:{phase_id}")
        return phase_id, default_duration

    fallback_match = re.search(
        r"[\"']?phase(?:_id)?[\"']?\s*[:=]\s*(\d+).*?[\"']?duration[\"']?\s*[:=]\s*(\d+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if fallback_match:
        phase_id = int(fallback_match.group(1))
        duration = _nearest_duration(int(fallback_match.group(2)))
        if phase_id < 0 or (num_phases is not None and phase_id >= num_phases):
            raise ActionParseError(f"phase_out_of_range:{phase_id}")
        return phase_id, duration

    raise ActionParseError("action_not_found")
