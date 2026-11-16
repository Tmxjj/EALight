"""场景 lane → prompt key 的静态映射层。

当前仓库中已冻结 `lane_mapping_generated.py` 的完整映射结果，
这里直接作为最终真值加载，供 gt_collector / verifier / 数据转换流程使用。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple


from .lane_mapping_generated import LANE_LAYOUT_GENERATED  # type: ignore


# 冻结后的场景映射真值，后续若有新版本只需重新生成 lane_mapping_generated.py。
# tuple 格式：(direction, lane_idx, movement_type, affected_phase)
LANE_LAYOUT: Dict[str, Dict[str, Tuple[str, Optional[int], str, Optional[int]]]] = dict(LANE_LAYOUT_GENERATED or {})


def _is_unrestricted(function: str) -> bool:
    return str(function).endswith("_unrestricted")


def lane_to_prompt_key(scenario: str, lane_id: str) -> Optional[str]:
    """返回标准化的 prompt key。

    受控 lane 返回 'N_L1' 形式；unrestricted lane 返回 'N_R_Unrestricted' / 'N_L_Unrestricted'。
    未注册 lane 返回 None。
    """
    scenario_layout = LANE_LAYOUT.get(str(scenario), {})
    entry = scenario_layout.get(str(lane_id))
    if not entry:
        return None

    approach, lane_idx, function = entry[0], entry[1], entry[2]
    if _is_unrestricted(function):
        suffix = str(function).split("_", 1)[0].upper()
        return f"{approach}_{suffix}_Unrestricted"
    if lane_idx is None:
        return None
    return f"{approach}_L{int(lane_idx)}"


def lane_prompt_key_from_layout(approach: str, lane_idx: Optional[int], function: str) -> Optional[str]:
    """仅基于 layout tuple 生成 prompt key。

    这是 lane_to_prompt_key 的无查询版本，供 collector 在 layout 已知但映射函数
    不可用时兜底使用。
    """
    if not approach:
        return None
    if _is_unrestricted(function):
        suffix = str(function).split("_", 1)[0].upper()
        return f"{approach}_{suffix}_Unrestricted"
    if lane_idx is None:
        return None
    return f"{approach}_L{int(lane_idx)}"


def get_controlled_lane_keys(scenario: str) -> set[str]:
    """返回该场景所有受控 lane 的 prompt key 集合。"""
    scenario_layout = LANE_LAYOUT.get(str(scenario), {})
    controlled_keys: set[str] = set()
    for lane_id, entry in scenario_layout.items():
        approach, lane_idx, function = entry[0], entry[1], entry[2]
        if lane_idx is None or _is_unrestricted(function):
            continue
        controlled_keys.add(f"{approach}_L{int(lane_idx)}")
    return controlled_keys


def get_all_lane_prompt_keys(scenario: str) -> set[str]:
    """返回该场景内所有已注册 lane 的 prompt key 集合。

    包括受控 lane 和 unrestricted lane，用于 GT 初始化时保留空车道条目。
    """
    scenario_layout = LANE_LAYOUT.get(str(scenario), {})
    all_keys: set[str] = set()
    for lane_id in scenario_layout:
        prompt_key = lane_to_prompt_key(scenario, lane_id)
        if prompt_key:
            all_keys.add(prompt_key)
    return all_keys


def get_layout(scenario: str) -> Dict[str, Tuple[str, Optional[int], str, Optional[int]]]:
    """返回指定场景的完整 lane layout，供 gt_collector 等模块读取。"""
    return dict(LANE_LAYOUT.get(str(scenario), {}))


def lane_to_affected_phase(scenario: str, lane_id: str) -> Optional[int]:
    """返回该车道对应的受控信号相位编号，不受信控（R_unrestricted 等）返回 None。"""
    entry = LANE_LAYOUT.get(str(scenario), {}).get(str(lane_id))
    if entry is None or len(entry) < 4:
        return None
    return entry[3]


__all__ = [
    "LANE_LAYOUT",
    "lane_to_prompt_key",
    "lane_prompt_key_from_layout",
    "get_controlled_lane_keys",
    "get_all_lane_prompt_keys",
    "get_layout",
    "lane_to_affected_phase",
]
