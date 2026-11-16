"""从 data/raw/{scenario}/add/e2.add.xml 和 net.xml 生成 lane_mapping 骨架。

这个脚本负责两件事：
1. 从 e2.add.xml 解析 lane_id / function；
2. 从 net.xml 解析 edge 的 from/to 节点坐标，推导 approach 方向（N/E/S/W）。

如果 net.xml 不可用，approach 仍写为 '?'。
SouthKorea_Songdo（多车道，车辆视角左→右须反向）、Hongkong_YMT（左行特例）
保留定制排序规则，以确保生成结果与 prompt 模板一致。
生成结果会直接写入 lane_mapping_generated.py，作为冻结后的最终映射。
对于映射结果人工已经revier，后讯禁止运行此脚本覆盖 lane_mapping_generated.py，以免误改已 review 的映射结果。
"""

from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


SCENARIOS = [
    "JiNan",
    "Hangzhou",
    "NewYork",
    "SouthKorea_Songdo",
    "Hongkong_YMT",
    "France_Massy",
    "Syn-Train",
]

# 每个场景对应的 net.xml 文件名（相对于 data/raw/{scenario}/env/）
_NET_XML_NAMES = {
    "JiNan": "jinan.net.xml",
    "Hangzhou": "Hangzhou.net.xml",
    "NewYork": "NewYork.net.xml",
    "SouthKorea_Songdo": "songdo.net.xml",
    "Hongkong_YMT": "YMT.net.xml",
    "France_Massy": "massy.net.xml",
    "Syn-Train": "roadnet_4_4.net.xml",
}


def _parse_e2_id(e2_id: str):
    """解析 e2 detector id：e2det--{junction}--{edge}--{lane}--{func}"""
    parts = str(e2_id).split("--")
    if len(parts) < 5:
        return None
    return {
        "junction_id": parts[1],
        "edge_id": parts[2],
        "lane_id": parts[3],
        "func": parts[4].lower(),
    }


def _lane_index(lane_id: str):
    """从 lane_id 末尾提取 SUMO lane index。"""
    match = re.search(r"_(\d+)$", str(lane_id))
    if not match:
        return None
    return int(match.group(1))


def _build_edge_approach_map(net_xml_path: str) -> dict[str, str]:
    """解析 net.xml，返回 edge_id → approach_direction 的字典。

    approach_direction 是车辆进入路口的来向（N/E/S/W），
    等于车辆行驶方向的反方向：向东行驶 → 来自西方 → approach = W。
    """
    if not os.path.exists(net_xml_path):
        return {}

    tree = ET.parse(net_xml_path)
    root = tree.getroot()

    # 采集所有路口节点的坐标
    junctions: dict[str, tuple[float, float]] = {}
    for j in root.findall(".//junction"):
        jid = j.attrib.get("id", "")
        try:
            x, y = float(j.attrib.get("x", 0)), float(j.attrib.get("y", 0))
        except ValueError:
            continue
        junctions[jid] = (x, y)

    edge_to_approach: dict[str, str] = {}
    for edge in root.findall(".//edge"):
        eid = edge.attrib.get("id", "")
        if eid.startswith(":"):  # 内部连接边，跳过
            continue
        from_node = edge.attrib.get("from", "")
        to_node = edge.attrib.get("to", "")
        fp = junctions.get(from_node)
        tp = junctions.get(to_node)
        if fp is None or tp is None:
            continue

        dx, dy = tp[0] - fp[0], tp[1] - fp[1]
        # SUMO 坐标系：x 向右为正（东），y 向上为正（北）
        if abs(dx) >= abs(dy):
            approach = "W" if dx > 0 else "E"
        else:
            approach = "S" if dy > 0 else "N"
        edge_to_approach[eid] = approach

    return edge_to_approach


def _build_one_scenario(scenario: str) -> dict:
    """基于 e2.add.xml 和 net.xml 生成一个场景的骨架 layout。"""
    add_path = os.path.join(_PROJECT_ROOT, "data", "raw", scenario, "add", "e2.add.xml")
    if not os.path.exists(add_path):
        return {}

    net_name = _NET_XML_NAMES.get(scenario)
    net_path = os.path.join(_PROJECT_ROOT, "data", "raw", scenario, "env", net_name) if net_name else ""
    edge_approach_map = _build_edge_approach_map(net_path)

    tree = ET.parse(add_path)
    layout: dict[str, tuple] = {}
    by_junction_edge: dict[tuple[str, str], list] = defaultdict(list)

    for elem in tree.getroot().iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag != "laneAreaDetector":
            continue
        info = _parse_e2_id(elem.attrib.get("id", ""))
        if info is None:
            continue
        info["lane_index"] = _lane_index(info["lane_id"])
        by_junction_edge[(info["junction_id"], info["edge_id"])].append(info)

    for (_junction_id, edge_id), entries in by_junction_edge.items():
        # 按 lane index 升序排列，与 SUMO 从右到左的顺序一致
        entries.sort(key=lambda item: item["lane_index"] if item["lane_index"] is not None else -1)

        # 从 net.xml 反查 approach 方向；缺失时保留 '?' 等待人工 review
        approach = edge_approach_map.get(edge_id, "?")

        normalized_entries = []
        for info in entries:
            func = info["func"]
            if scenario == "Hongkong_YMT" and func == "l":
                function = "L_unrestricted"
            elif scenario == "Hongkong_YMT" and func == "r":
                function = "R"
            elif func == "r":
                function = "R_unrestricted"
            elif func in {"s", "l"}:
                function = func.upper()
            else:
                function = func.upper()

            normalized_entries.append(
                {
                    "lane_id": info["lane_id"],
                    "lane_index": info["lane_index"] if info["lane_index"] is not None else -1,
                    "function": function,
                }
            )

        # lane_idx 从 1 开始，只对受控车道计数，并按 prompt 语义排序：
        # - 非香港：L 优先于 S（与 prompt 模板一致）
        # - 香港：R 优先于 S（左行特例）
        if scenario == "Hongkong_YMT":
            function_priority = {"R": 0, "S": 1, "L": 2}
            lane_index_sign = 1
        else:
            function_priority = {"L": 0, "S": 1, "R": 2}
            lane_index_sign = -1

        controlled = [
            item
            for item in normalized_entries
            if not str(item["function"]).endswith("_unrestricted")
        ]
        controlled.sort(
            key=lambda item: (
                function_priority.get(str(item["function"]), 9),
                lane_index_sign * int(item["lane_index"]),
                str(item["lane_id"]),
            )
        )

        lane_to_prompt_idx = {
            str(item["lane_id"]): idx for idx, item in enumerate(controlled, start=1)
        }

        for item in normalized_entries:
            lane_id = str(item["lane_id"])
            lane_idx = lane_to_prompt_idx.get(lane_id)
            function = str(item["function"])
            layout[lane_id] = (approach, lane_idx, function)

    return layout


def main() -> None:
    """生成 lane_mapping_generated.py。"""
    out_path = os.path.join(_PROJECT_ROOT, "src", "training", "rlvr", "lane_mapping_generated.py")
    chunks = [
        '"""自动生成的 LANE_LAYOUT 骨架。请人工 review 后并入 lane_mapping.py。"""',
        "",
        "LANE_LAYOUT_GENERATED = {",
    ]

    for scenario in SCENARIOS:
        layout = _build_one_scenario(scenario)
        chunks.append(f'    "{scenario}": {{')
        for lane_id, layout_tuple in sorted(layout.items()):
            chunks.append(f"        {lane_id!r}: {layout_tuple!r},")
        chunks.append("    },")

    chunks.append("}")

    with open(out_path, "w", encoding="utf-8") as file:
        file.write("\n".join(chunks) + "\n")

    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
