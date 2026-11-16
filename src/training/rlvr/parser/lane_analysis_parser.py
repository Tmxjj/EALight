"""Lane Analysis 解析器。

目标是把模型输出中每条受控车道的密度词提取为统一字典：
    {"N_L1": "Short", "S_L2": "Medium", ...}

整段 Lane Analysis 缺失时返回 None；部分 lane 缺失则返回部分字典，
由 compute_r_lane 的“漏报”规则继续处理。
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from src.training.rlvr.reward.constants import LEVEL_TO_ID, LEVEL_THRESHOLDS


_APPROACH_ALIASES = {
    "n": "N",
    "north": "N",
    "s": "S",
    "south": "S",
    "e": "E",
    "east": "E",
    "w": "W",
    "west": "W",
}


def _normalize_level(value: str) -> Optional[str]:
    """把数字或文本统一映射成 Empty/Short/Medium/Long/Critical。"""
    text = str(value).strip().strip(",.;")
    if not text:
        return None

    if text in LEVEL_TO_ID:
        return text

    try:
        count = float(text)
    except ValueError:
        return None

    # LEVEL_THRESHOLDS 格式为 (lo, hi, name)，区间为闭区间 [lo, hi]
    n = int(count)
    if n < 0:
        n = 0
    for lo, hi, name in LEVEL_THRESHOLDS:
        if lo <= n <= hi:
            return name
    return None


def parse_lane_analysis(response_text: str) -> Optional[Dict[str, str]]:
    """解析 Lane Analysis 段。

    支持两类常见写法：
    1. `N: L1(L): Short, L2(S): Medium`
    2. `North Approach: Lane 1(Left-Turn): 3, Lane 2(Straight): 1`

    返回的 key 统一为 `N_L1` / `S_L2` 形式。
    """
    if not isinstance(response_text, str) or not response_text.strip():
        return None

    match = re.search(
        r"Lane Analysis\s*:\s*(.+?)(?:\n\s*(?:Phase Mapping|Scene Analysis|Event Recognition|Selection Logic|Action)\b|\Z)",
        response_text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None

    section = match.group(1)
    result: Dict[str, str] = {}

    for raw_line in re.split(r"\n+|;", section):
        line = raw_line.strip()
        if not line:
            continue

        approach_match = re.match(
            r"(?P<approach>[NESW]|North|South|East|West)\s*(?:\([^)]*\))?\s*(?:Approach)?\s*:\s*(?P<body>.+)",
            line,
            re.IGNORECASE,
        )
        if not approach_match:
            continue

        approach = _APPROACH_ALIASES.get(approach_match.group("approach").lower())
        if not approach:
            continue

        body = approach_match.group("body")
        for lane_match in re.finditer(
            r"(?:Lane\s*)?(?P<lane_idx>\d+)\s*(?:\([^)]*\))?\s*:\s*(?P<level>Empty|Short|Medium|Long|Critical|\d+(?:\.\d+)?)",
            body,
            re.IGNORECASE,
        ):
            lane_idx = int(lane_match.group("lane_idx"))
            level = _normalize_level(lane_match.group("level"))
            if level is None:
                continue
            result[f"{approach}_L{lane_idx}"] = level

    return result or None
