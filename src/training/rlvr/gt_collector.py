"""基于 e2 detector 的 RLVR 真值采集器。

这个模块的目标很明确：尽量只依赖 detector 和 SUMO/TraCI 的运行时状态，
把 lane-level GT 和 event GT 从仿真环境里提取出来，供 golden generation、
RLVR verifier 以及后续数据增强流程复用。

设计上这里做了两层兼容：
1. 如果 lane_mapping.py 已经补齐，就优先使用它把 lane_id 转成 prompt key；
2. 如果 lane_mapping 还没准备好，就退化为使用 lane_id 本身，保证当前开发不被
   场景映射阻塞。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .reward.constants import (
    BARRIER_VTYPE_PREFIXES,
    DEBRIS_VTYPE_PREFIXES,
    EMERGENCY_VTYPES,
    EVENT_TYPES,
    LEVEL_THRESHOLDS,
    TRANSIT_VTYPES,
    VTYPE_TO_EVENT_TYPE,
)

try:  # pragma: no cover - lane_mapping 可能尚未补齐，collector 需要优雅降级
    from .lane_mapping import get_layout as _get_layout_impl
    from .lane_mapping import get_all_lane_prompt_keys as _get_all_lane_prompt_keys_impl
    from .lane_mapping import get_controlled_lane_keys as _get_controlled_lane_keys_impl
    from .lane_mapping import lane_prompt_key_from_layout as _lane_prompt_key_from_layout_impl
    from .lane_mapping import lane_to_prompt_key as _lane_to_prompt_key_impl
    from .lane_mapping import lane_to_affected_phase as _lane_to_affected_phase_impl
except Exception:  # pragma: no cover - 保持运行兼容
    _get_layout_impl = None
    _get_all_lane_prompt_keys_impl = None
    _get_controlled_lane_keys_impl = None
    _lane_prompt_key_from_layout_impl = None
    _lane_to_prompt_key_impl = None
    _lane_to_affected_phase_impl = None


_DETECTOR_ID_RE = re.compile(
    r"^e2det--(?P<junction>.+?)--(?P<edge>.+?)--(?P<lane>.+?)--(?P<kind>[A-Za-z0-9_]+)$"
)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """安全取属性，避免 detector / trafficlight 接口在不同环境中略有差异。"""
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _get_detector_ids(sumo) -> List[str]:
    """收集当前 SUMO 实例里可见的 detector id 列表。"""
    for namespace_name in ("lanearea", "e2", "inductionloop"):
        namespace = _safe_getattr(sumo, namespace_name)
        if namespace is None:
            continue
        get_id_list = _safe_getattr(namespace, "getIDList")
        if callable(get_id_list):
            try:
                return list(get_id_list())
            except Exception:
                continue
    return []


def _parse_detector_id(detector_id: str) -> Optional[Tuple[str, str, str, str]]:
    """解析 e2 detector id。

    期望格式为：e2det--{junction_id}--{edge_id}--{lane_id}--{function}
    如果不符合这个格式，就返回 None，让上层走兜底路径。
    """
    match = _DETECTOR_ID_RE.match(detector_id or "")
    if not match:
        return None
    return (
        match.group("junction"),
        match.group("edge"),
        match.group("lane"),
        match.group("kind"),
    )


def _get_lane_id(detector_id: str) -> str:
    """从 detector id 中尽量恢复 lane_id。"""
    parsed = _parse_detector_id(detector_id)
    if parsed is None:
        return detector_id
    return parsed[2]


def _count_to_level(count: int) -> str:
    """按 LEVEL_THRESHOLDS 把车辆数映射成文本等级。

    LEVEL_THRESHOLDS 格式为 (lo, hi, name)，区间为闭区间 [lo, hi]。
    """
    n = int(count or 0)
    if n < 0:
        n = 0
    for lo, hi, name in LEVEL_THRESHOLDS:
        if lo <= n <= hi:
            return name
    return "Critical"


def _level_rank(level: str) -> int:
    """把文本等级转换成可比较的整数，方便做 max 聚合。"""
    for idx, (_lo, _hi, candidate_level) in enumerate(LEVEL_THRESHOLDS):
        if candidate_level == level:
            return idx
    return 0


def _classify_vtype(vtype: str) -> Optional[str]:
    """把 SUMO vType 统一映射成设计文档中的事件类型。"""
    if not vtype:
        return None

    raw = str(vtype).strip()
    if raw in VTYPE_TO_EVENT_TYPE:
        return VTYPE_TO_EVENT_TYPE[raw]

    lowered = raw.lower()
    if lowered in EMERGENCY_VTYPES:
        if lowered == "police":
            return "Police"
        if lowered == "fire_engine":
            return "FireTruck"
        return "Ambulance"
    if lowered in TRANSIT_VTYPES:
        return "SchoolBus" if lowered == "school_bus" else "PublicBus"

    if any(lowered.startswith(prefix) for prefix in BARRIER_VTYPE_PREFIXES):
        return "ConstructionBarrier"
    if any(lowered.startswith(prefix) for prefix in DEBRIS_VTYPE_PREFIXES):
        return "RoadDebris"
    if lowered.startswith("crash_vehicle"):
        return "Accident"

    # 允许未来数据集中直接出现规范化事件类型字符串。
    if raw in EVENT_TYPES:
        return raw
    return None


def _lane_to_prompt_key(scenario_key: str, lane_id: str, detector_id: str) -> str:
    """把 lane_id 转成 prompt key。

    优先使用 lane_mapping.py；如果该模块尚未准备好，就退化为 lane_id 本身。
    """
    if callable(_lane_to_prompt_key_impl):
        try:
            mapped = _lane_to_prompt_key_impl(scenario_key, lane_id)
            if mapped:
                return str(mapped)
        except Exception:
            pass
    return lane_id or detector_id


def _lane_key_from_layout(scenario_key: str, lane_id: str) -> str:
    """从 layout tuple 兜底生成 prompt key。"""
    if callable(_get_layout_impl) and callable(_lane_prompt_key_from_layout_impl):
        try:
            layout = _get_layout_impl(scenario_key) or {}
            entry = layout.get(lane_id)
            if entry:
                approach, lane_idx, function = entry
                mapped = _lane_prompt_key_from_layout_impl(approach, lane_idx, function)
                if mapped:
                    return str(mapped)
        except Exception:
            pass
    return lane_id


def _get_known_lane_prompt_keys(scenario_key: str) -> set[str]:
    """返回该场景能稳定识别的所有 prompt key。

    优先使用显式导出的 all keys；如果不可用，则从 layout tuple 现算。
    """
    keys: set[str] = set()
    if callable(_get_all_lane_prompt_keys_impl):
        try:
            keys.update(_get_all_lane_prompt_keys_impl(scenario_key) or set())
        except Exception:
            keys.clear()

    if keys:
        return keys

    if callable(_get_layout_impl) and callable(_lane_prompt_key_from_layout_impl):
        try:
            layout = _get_layout_impl(scenario_key) or {}
            for lane_id, (approach, lane_idx, function) in layout.items():
                prompt_key = _lane_prompt_key_from_layout_impl(approach, lane_idx, function)
                if prompt_key:
                    keys.add(prompt_key)
        except Exception:
            return set()
    return keys


def _get_vehicle_ids_from_detector(sumo, detector_id: str) -> List[str]:
    """读取单个 detector 当前时刻覆盖到的车辆 id 列表。"""
    for namespace_name in ("lanearea", "e2", "inductionloop"):
        namespace = _safe_getattr(sumo, namespace_name)
        if namespace is None:
            continue

        getter = _safe_getattr(namespace, "getLastStepVehicleIDs")
        if callable(getter):
            try:
                return list(getter(detector_id) or [])
            except Exception:
                continue

        getter = _safe_getattr(namespace, "getLastStepVehicleIDList")
        if callable(getter):
            try:
                return list(getter(detector_id) or [])
            except Exception:
                continue
    return []


def _get_lane_to_phase_fn(sumo) -> Callable[[str, str, str], Optional[int]]:
    """构造 lane → phase 的反查函数。

    这里不强依赖完整 lane_mapping，而是直接利用 trafficlight 的控制关系做一个
    轻量反查。若接口不可用，则返回一个始终产出 None 的函数。
    """

    def _fallback(*_args: str) -> Optional[int]:
        return None

    trafficlight = _safe_getattr(sumo, "trafficlight")
    if trafficlight is None:
        return _fallback

    get_links = _safe_getattr(trafficlight, "getControlledLinks")
    get_programs = _safe_getattr(trafficlight, "getAllProgramLogics")
    if not callable(get_links) or not callable(get_programs):
        return _fallback

    # 预构造一个缓存，避免对同一路口重复扫描。
    cache: Dict[str, Dict[str, Optional[int]]] = {}

    def _resolve(_scenario_key: str, junction_id: str, lane_id: str) -> Optional[int]:
        junction_cache = cache.get(junction_id)
        if junction_cache is None:
            junction_cache = {}
            try:
                links = list(get_links(junction_id) or [])
                programs = list(get_programs(junction_id) or [])
            except Exception:
                cache[junction_id] = junction_cache
                return None

            phases = []
            if programs:
                phases = list(getattr(programs[0], "phases", []) or [])

            for phase_idx, phase in enumerate(phases):
                state = getattr(phase, "state", None)
                if state is None:
                    continue
                for link_idx, link_group in enumerate(links):
                    if link_idx >= len(state):
                        continue
                    if str(state[link_idx]).upper() not in {"G", "g"}:
                        continue
                    for link in link_group or []:
                        in_lane = link[0] if isinstance(link, (list, tuple)) and link else None
                        if in_lane == lane_id:
                            junction_cache[lane_id] = phase_idx
                            break
                    if lane_id in junction_cache:
                        break
                if lane_id in junction_cache:
                    break

            cache[junction_id] = junction_cache

        return junction_cache.get(lane_id)

    return _resolve


def collect_ground_truth(sumo, scenario_key: str, junction_id: str) -> Dict[str, Any]:
    """采集单个路口的 lane-level GT 和 event GT。

    返回值尽量保持简洁，只暴露 RLVR / golden generation 真正需要的两部分：
    - lane_level_gt: {prompt_key: level_str}
    - event_gt:      [{"event_type": str, "located_lane": str, "affected_phase": int | None}, ...]
    """
    controlled_lane_keys = sorted(get_controlled_lane_keys(scenario_key))
    all_lane_keys = sorted(_get_known_lane_prompt_keys(scenario_key) or controlled_lane_keys)
    # 先把所有已知 lane 初始化为 Empty，避免“无车车道”被直接丢弃。
    lane_level_candidates: Dict[str, Tuple[int, str]] = {
        key: (_level_rank("Empty"), "Empty") for key in all_lane_keys
    }
    event_gt: List[Dict[str, Any]] = []
    seen_vehicle_ids: set[str] = set()
    allow_unmapped_lane_keys = not all_lane_keys

    for detector_id in _get_detector_ids(sumo):
        try:
            parsed = _parse_detector_id(detector_id)
            if parsed is not None and parsed[0] != junction_id:
                continue

            vehicle_ids = _get_vehicle_ids_from_detector(sumo, detector_id)
            if not vehicle_ids:
                continue

            lane_id = _get_lane_id(detector_id)
            prompt_key = _lane_to_prompt_key(scenario_key, lane_id, detector_id)
            if prompt_key not in lane_level_candidates:
                prompt_key = _lane_key_from_layout(scenario_key, lane_id)
            if not allow_unmapped_lane_keys and prompt_key not in lane_level_candidates:
                continue
            level = _count_to_level(len(vehicle_ids))

            # 同一个 prompt_key 可能对应多个 detector 的碎片化观测，这里按更“拥挤”的
            # 等级保留，避免真值被低估。
            current = lane_level_candidates.get(prompt_key)
            candidate_rank = _level_rank(level)
            if current is None or candidate_rank > current[0]:
                lane_level_candidates[prompt_key] = (candidate_rank, level)

            for vehicle_id in vehicle_ids:
                if vehicle_id in seen_vehicle_ids:
                    continue
                seen_vehicle_ids.add(vehicle_id)

                try:
                    vtype = sumo.vehicle.getTypeID(vehicle_id)
                except Exception:
                    continue

                event_type = _classify_vtype(vtype)
                if event_type is None:
                    continue

                affected_phase = (
                    _lane_to_affected_phase_impl(scenario_key, lane_id)
                    if _lane_to_affected_phase_impl is not None
                    else None
                )

                event_gt.append(
                    {
                        "event_type": event_type,
                        "located_lane": prompt_key,
                        "affected_phase": affected_phase,
                    }
                )
        except Exception:
            # 单个 detector 异常不应让整条样本退化成空 GT。
            continue

    lane_level_gt = {
        prompt_key: level
        for prompt_key, (_rank, level) in lane_level_candidates.items()
    }

    return {
        "lane_level_gt": lane_level_gt,
        "event_gt": event_gt,
    }


def get_controlled_lane_keys(scenario_key: str) -> set[str]:
    """尽量返回该场景的受控 lane prompt key 集合。

    如果 lane_mapping.py 还没准备好，就返回空集合，调用方可以继续用 lane_level_gt
    的 key 集合作为保底。
    """
    if callable(_get_controlled_lane_keys_impl):
        try:
            return set(_get_controlled_lane_keys_impl(scenario_key) or set())
        except Exception:
            return set()
    return set()


__all__ = [
    "collect_ground_truth",
    "get_controlled_lane_keys",
]
