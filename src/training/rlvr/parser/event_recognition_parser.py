"""Event Recognition 解析器。"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.training.rlvr.reward.constants import EVENT_TYPES


# VLM response 中常见的 event type 写法 → 设计文档标准名称
# prompt_builder.py 的 EVENT_DESCRIPTIONS 用 "Police Car" / "Fire Truck" /
# "Traffic Accident" 等带空格的写法，而 gt_collector 存储 "Police" / "FireTruck" /
# "Accident"，此处统一做规范化映射。
_EVENT_TYPE_ALIASES = {
    # ── Ambulance ──────────────────────────────────────────────────────────
    "ambulance": "Ambulance",
    # ── Police ─────────────────────────────────────────────────────────────
    "police": "Police",
    "police car": "Police",
    "police vehicle": "Police",
    "police officer": "Police",
    # ── FireTruck ───────────────────────────────────────────────────────────
    "firetruck": "FireTruck",
    "fire truck": "FireTruck",
    "fire engine": "FireTruck",
    "fire-truck": "FireTruck",
    "fire-engine": "FireTruck",
    # ── PublicBus ───────────────────────────────────────────────────────────
    "publicbus": "PublicBus",
    "public bus": "PublicBus",
    "bus": "PublicBus",
    "transit bus": "PublicBus",
    # ── SchoolBus ───────────────────────────────────────────────────────────
    "schoolbus": "SchoolBus",
    "school bus": "SchoolBus",
    "school-bus": "SchoolBus",
    # ── Accident ────────────────────────────────────────────────────────────
    "accident": "Accident",
    "traffic accident": "Accident",
    "vehicle collision": "Accident",   # golden 生成脚本输出
    "collision": "Accident",
    "crash": "Accident",
    "vehicle crash": "Accident",
    # ── RoadDebris ──────────────────────────────────────────────────────────
    "roaddebris": "RoadDebris",
    "road debris": "RoadDebris",
    "debris": "RoadDebris",
    "road obstruction": "RoadDebris",
    "fallen debris": "RoadDebris",
    # ── ConstructionBarrier ─────────────────────────────────────────────────
    "constructionbarrier": "ConstructionBarrier",
    "construction barrier": "ConstructionBarrier",
    "barrier": "ConstructionBarrier",
    "construction zone": "ConstructionBarrier",
    "road barrier": "ConstructionBarrier",
    "roadblock": "ConstructionBarrier",
}

# 方向全称/缩写 → 单字母代码（与 gt_collector 的 prompt_key 格式一致）
_DIRECTION_ALIASES = {
    "n": "N",
    "north": "N",
    "s": "S",
    "south": "S",
    "e": "E",
    "east": "E",
    "w": "W",
    "west": "W",
}

# ── 位置描述规范化的各类正则，按优先级顺序尝试 ──────────────────────────────
#
# 数据集中观察到的真实写法（来自 04_sft_train_dataset.jsonl 抽样）：
#   常规：  "North L1"、"South Approach L2"、"East Approach L2(S)"
#           "North Approach Lane 1"、"N L2"、"West L3"
#   带括号功能标注：  "East L1(L)"、"North Approach L1(L)"
#   逗号分隔：  "West Approach, L1"、"West Approach, L3"
#   顺序颠倒：  "Approach East L1"（方向在中间）
#   "&" 连接：  "West & L2(S)"
#   多车道（取首个）：  "North Approach L1 and L2"、
#                       "North Approach L2 and South Approach L2"

# 模式0（优先级最高）：不受限转向描述
#   覆盖：  "North Right-Turn"、"West Unrestricted"、
#           "East Free Right"、"South Left-Turn (Unrestricted)"
#   affected_phase=None 时优先尝试，生成 "{Dir}_R_Unrestricted" / "{Dir}_L_Unrestricted"
_LOC_RE_UNRESTRICTED = re.compile(
    r"(?P<dir>[NESW]|North|South|East|West)"
    r"(?:\s+Approach(?:es)?)?"
    r"(?:\s*,)?"
    r"\s+(?:(?P<move>Right|Left)[-\s]?Turn|Free\s+(?P<free>Right|Left)|(?P<unr>Unrestricted))",
    re.IGNORECASE,
)

# 模式1：标准顺序 "[Dir] [Approach[,]] [Lane] N"
#   覆盖：North L1 / South Approach L2 / West Approach, L3 /
#         East Approach Lane 2 / N_L1（已规范）/ N L2
_LOC_RE_STANDARD = re.compile(
    r"(?P<dir>[NESW]|North|South|East|West)"
    r"(?:\s+Approach(?:es)?)?"           # 可选 "Approach"
    r"(?:\s*,)?"                          # 可选逗号（"West Approach, L1"）
    r"(?:\s+Lane)?"                       # 可选 "Lane"
    r"\s+L?(?P<lane>\d+)",               # L1 或 1
    re.IGNORECASE,
)

# 模式2：顺序颠倒 "Approach [Dir] LN"（"Approach East L1"）
_LOC_RE_REVERSED = re.compile(
    r"Approach\s+(?P<dir>[NESW]|North|South|East|West)"
    r"(?:\s+Lane)?"
    r"\s+L?(?P<lane>\d+)",
    re.IGNORECASE,
)

# 模式3："[Dir] & LN(func)" ("West & L2(S)")
_LOC_RE_AMPERSAND = re.compile(
    r"(?P<dir>[NESW]|North|South|East|West)"
    r"\s*&\s*"
    r"L?(?P<lane>\d+)",
    re.IGNORECASE,
)

# 模式4：仅包含方向（供 affected_phase=None 时提取方向后缀兜底）
_LOC_RE_DIR_ONLY = re.compile(
    r"(?P<dir>[NESW]|North|South|East|West)",
    re.IGNORECASE,
)

_LOC_PATTERNS = [_LOC_RE_STANDARD, _LOC_RE_REVERSED, _LOC_RE_AMPERSAND]

# ── Event Recognition 段终止符 ────────────────────────────────────────────
# 覆盖各版本 CoT 模板中紧跟 Event Recognition 之后可能出现的行首关键词
_SECTION_END_RE = re.compile(
    r"\n\s*-?\s*(?:"
    r"Neighbor(?:ing)?\s*Msgs?"      # "- Neighbor Msgs:" 和 "Neighboring Messages:"
    r"|Condition"
    r"|Selection\s*Logic"
    r"|Action"
    r"|Broadcast\s*Notice"
    r"|Impact\s*Analysis"
    r"|Phase\s*Reasoning"
    r")\b",
    re.IGNORECASE,
)

# ── 主解析正则（两个变体）────────────────────────────────────────────────
# 变体1：标准 "affects/affected Phase N"
_EVENT_PATTERN = re.compile(
    r"(?P<type>[A-Za-z][A-Za-z\s\-]+?)\s*"    # 事件类型（允许连字符和空格）
    r"(?:\([^)]+\))?\s*"                        # 可选括号类别
    r"detected\s+at\s+"                         # "detected at"
    r"(?P<location>.+?)\s*,\s*"                # 位置（懒惰）
    r"(?:directly\s+)?affect(?:s|ed)?\s+[Pp]hase\s*[:\-]?\s*"
    r"(?P<phase>None|\d+)",
    re.IGNORECASE,
)

# 变体2："impacts Phase N"
_EVENT_PATTERN_IMPACTS = re.compile(
    r"(?P<type>[A-Za-z][A-Za-z\s\-]+?)\s*"
    r"(?:\([^)]+\))?\s*"
    r"detected\s+at\s+"
    r"(?P<location>.+?)\s*,\s*"
    r"impacts?\s+[Pp]hase\s+"
    r"(?P<phase>None|\d+)",
    re.IGNORECASE,
)

_EVENT_PATTERNS = [_EVENT_PATTERN, _EVENT_PATTERN_IMPACTS]


def _normalize_event_type(raw_text: str) -> Optional[str]:
    """把多种自然语言写法统一成设计文档中的事件类型。"""
    # 去除非字母/空格字符（连字符转空格），压缩多余空白，转小写
    key = re.sub(r"[^a-zA-Z ]+", " ", raw_text).strip().lower()
    key = re.sub(r"\s+", " ", key)
    if key in _EVENT_TYPE_ALIASES:
        return _EVENT_TYPE_ALIASES[key]
    # 直接匹配标准名称集合（如 "Ambulance"）
    if raw_text.strip() in EVENT_TYPES:
        return raw_text.strip()
    return None


def _normalize_location_lane(raw_location: str, affected_phase: Optional[int] = None) -> str:
    """把 VLM 输出的位置描述统一成 gt_collector 的 prompt_key 格式（如 "N_L1"）。

    已覆盖的真实写法（来自训练数据抽样）：
      - 标准：      "North L1" / "South Approach L2" / "East Approach Lane 3"
      - 带功能括号："East L1(L)" / "North Approach L2(S)"
      - 逗号分隔：  "West Approach, L1"
      - 顺序颠倒：  "Approach East L1"
      - & 连接：    "West & L2(S)"
      - 多车道：    "North Approach L1 and L2"（取第一个车道）
      - 已规范：    "N_L1" / "N_R_Unrestricted"（直接返回）

    当 affected_phase=None 时（不受信控的自由转向车道），优先尝试提取
    转向类型生成 "{Dir}_R_Unrestricted" / "{Dir}_L_Unrestricted"，
    避免将不受限车道误识别为受控车道（如 "North L2" → "N_L2"）。

    无法解析时原样返回，让下游 reward 产生 lane miss 而非崩溃。
    """
    text = raw_location.strip()

    # 已经是标准格式，直接返回
    if re.match(r"^[NESW]_L\d+$", text):
        return text
    # 已经是 unrestricted 格式，直接返回
    if re.match(r"^[NESW]_[RL]_Unrestricted$", text, re.IGNORECASE):
        return text

    # affected_phase=None 表明该车道不受信控（自由转向），优先匹配不受限描述
    if affected_phase is None:
        # 先尝试含方向关键词的不受限描述
        m = _LOC_RE_UNRESTRICTED.search(text)
        if m:
            direction = _DIRECTION_ALIASES.get(m.group("dir").lower())
            if direction:
                move_raw = m.group("move") or m.group("free") or ""
                move = move_raw.strip().capitalize()  # "Right" / "Left" / ""
                if move in ("Right", "Left"):
                    return f"{direction}_{move[0]}_Unrestricted"
                # "Unrestricted" 关键词但无 R/L 信息，默认右转（RHT 规则下最常见）
                return f"{direction}_R_Unrestricted"

        # 无明确转向关键词：尝试从括号功能标注推断 "(R)" / "(L)"
        bracket_m = re.search(
            r"(?P<dir>[NESW]|North|South|East|West)"
            r"(?:[^()]*)"
            r"\((?P<func>[RL])\)",
            text,
            re.IGNORECASE,
        )
        if bracket_m:
            direction = _DIRECTION_ALIASES.get(bracket_m.group("dir").lower())
            func = bracket_m.group("func").upper()
            if direction and func in ("R", "L"):
                return f"{direction}_{func}_Unrestricted"

        # 无明确转向信息：提取方向后默认右转（RHT 场景下不受限通常为右转；
        # Hongkong LHT 为左转，但此处仅能给出方向级的兜底 key）
        dir_m = _LOC_RE_DIR_ONLY.search(text)
        if dir_m:
            direction = _DIRECTION_ALIASES.get(dir_m.group("dir").lower())
            if direction:
                return f"{direction}_R_Unrestricted"

    for pattern in _LOC_PATTERNS:
        m = pattern.search(text)
        if m:
            direction = _DIRECTION_ALIASES.get(m.group("dir").lower())
            if direction:
                return f"{direction}_L{m.group('lane')}"

    return text


def _extract_section(response_text: str) -> Optional[str]:
    """从完整 response 中提取 Event Recognition 段内容。

    先用精确的段终止符截断，找不到时退化到字符串末尾。
    """
    start_m = re.search(r"Event Recognition\s*:\s*", response_text, re.IGNORECASE)
    if not start_m:
        return None

    tail = response_text[start_m.end():]
    end_m = _SECTION_END_RE.search(tail)
    if end_m:
        return tail[: end_m.start()].strip()
    return tail.strip()


def parse_event_recognition(response_text: str) -> Optional[List[Dict[str, object]]]:
    """解析 Event Recognition 段。

    返回值说明：
    - None：整段解析失败；
    - []：明确无事件（模型输出 None）；
    - list[dict]：事件列表，每项包含 event_type / located_lane / affected_phase。
    """
    if not isinstance(response_text, str) or not response_text.strip():
        return None

    section = _extract_section(response_text)
    if section is None:
        return None

    # 明确无事件
    if re.fullmatch(r"`?none`?\.?", section, re.IGNORECASE):
        return []

    # 切分 segment：支持分号、换行、管道符
    segments = [s.strip() for s in re.split(r"[;\n|]+", section) if s.strip()]
    result: List[Dict[str, object]] = []

    for segment in segments:
        seen_spans = set()
        for pat in _EVENT_PATTERNS:
            for event_match in pat.finditer(segment):
                # 只去掉同一段文本被多个 pattern 重复命中的情况；不要按 entry 内容
                # 去重，否则同车道上的两辆同类型事件会被错误折叠成一个事件。
                if event_match.span() in seen_spans:
                    continue
                seen_spans.add(event_match.span())

                event_type = _normalize_event_type(event_match.group("type"))
                if event_type is None:
                    continue

                phase_text = event_match.group("phase")
                affected_phase = None if phase_text.lower() == "none" else int(phase_text)
                entry = {
                    "event_type": event_type,
                    "located_lane": _normalize_location_lane(
                        event_match.group("location").strip(),
                        affected_phase=affected_phase,
                    ),
                    "affected_phase": affected_phase,
                }
                result.append(entry)

    return result or None

# if main := __name__:
#     # 简单测试
#     test_response = """
# Thought: [
# A. Scene Understanding:
# - Lane Analysis:
# N: L1(L):Medium, L2(S):Short
# S: L1(L):Medium, L2(S):Short
# E: L1(L):Empty, L2(S):Short
# W: L1(L):Empty, L2(S):Empty
# - Phase Mapping:
# Phase 0 (ETWT): Pressure: Low | MaxQueue: Short
# Phase 1 (NTST): Pressure: Low | MaxQueue: Short
# Phase 2 (ELWL): Pressure: Empty | MaxQueue: Empty
# Phase 3 (NLSL): Pressure: Medium | MaxQueue: Medium
# Tie-Breaker: None

# B. Scene Analysis:
# - Event Recognition: Public Bus (Transit) detected at West L2, affects Phase 0 | Public Bus (Transit) detected at West L2, affects Phase 0
# - Neighbor Msgs: Inactive
# - Condition: Special

# C. Adaptive Reasoning:
# - Impact Analysis: A fire truck is detected in the North left-turn lane, requiring immediate emergency preemption to ensure safe and rapid passage.
# - Phase Reasoning: Phase 3 must be selected immediately to clear the path for the emergency vehicle in the North left-turn lane.
# - Dur Reasoning: A duration of 35s is selected to ensure the fire truck and the vehicles queued ahead of it can fully clear the intersection.
# - Broadcast Notice: Fire Truck - High priority passage required for emergency vehicle.
# ]
# Action: {"phase": 3, "duration": 35}
#     """
#     print(parse_event_recognition(test_response))