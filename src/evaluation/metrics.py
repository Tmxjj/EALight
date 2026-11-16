import numpy as np
import xml.etree.ElementTree as ET
from loguru import logger
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.scenairo_config import SCENARIO_CONFIGS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 紧急车辆 vType 集合（与 scripts/event_scene_generation/add_emergency_vehicles.py 中的 vType id 严格对齐）
_DEFAULT_EMERGENCY_VTYPES = {"emergency", "ambulance", "police", "police_car", "fire_engine", "fire_truck"}
_DEFAULT_BUS_VTYPES = {"bus", "public_bus", "school_bus"}
_DEFAULT_SPECIAL_VTYPES = _DEFAULT_EMERGENCY_VTYPES | _DEFAULT_BUS_VTYPES

# 默认事件车辆 ID 前缀（不同场景可能只传其中的子集）
_ALL_EVENT_PREFIXES = ("accident_", "pedestrian_", "debris_", "ped_")


def _mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def _is_event_vehicle_id(veh_id: str, event_id_prefixes) -> bool:
    return bool(event_id_prefixes) and any(veh_id.startswith(p) for p in event_id_prefixes)


def _iter_existing_route_candidates(route_file: str = None,
                                    queue_file: str = None,
                                    tripinfo_file: str = None,
                                    scenario_name: str = None):
    """按优先级生成可能的 .rou.xml 路径。"""
    candidates = []

    def add(path):
        if not path:
            return
        if os.path.isabs(path):
            candidates.append(path)
        else:
            candidates.append(os.path.join(PROJECT_ROOT, path))

    add(route_file)

    # SUMO 输出 XML 的开头注释中通常包含 <route-files value="..."/>。
    for output_file in (tripinfo_file, queue_file):
        if not output_file or not os.path.exists(output_file):
            continue
        try:
            with open(output_file, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(12000)
            for route_value in re.findall(r'<route-files\s+value="([^"]+)"', head):
                for item in route_value.split(","):
                    item = item.strip()
                    if not item:
                        continue
                    if item.startswith("./") and scenario_name:
                        add(os.path.join("data", "raw", scenario_name, item[2:]))
                    else:
                        add(item)
        except Exception as e:
            logger.debug(f"[EVAL] 读取输出文件路由元信息失败: {e} - {output_file}")

    # data/eval/{scenario}/{route_stem}/{method}/queue_output.xml
    if queue_file and scenario_name:
        parts = os.path.normpath(queue_file).split(os.sep)
        try:
            eval_idx = parts.index("eval")
            if len(parts) > eval_idx + 2 and parts[eval_idx + 1] == scenario_name:
                route_stem = parts[eval_idx + 2]
                if route_stem.endswith(".rou"):
                    add(os.path.join("data", "raw", scenario_name, "env", f"{route_stem}.xml"))
                else:
                    add(os.path.join("data", "raw", scenario_name, "env", f"{route_stem}.rou.xml"))
                    add(os.path.join("data", "raw", scenario_name, "env", f"{route_stem}.xml"))
        except ValueError:
            pass

    seen = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            yield candidate


def _load_route_vehicle_types(route_file: str = None,
                              queue_file: str = None,
                              tripinfo_file: str = None,
                              scenario_name: str = None) -> dict:
    """
    解析 raw .rou.xml，返回 {vehicle_id: vType}。

    混合事件路由中普通车和特殊车混排，tripinfo 里的 vType 可能被 SUMO
    默认类型覆盖或缺失；专项指标以路由文件中的 id -> type 为准。
    """
    for candidate in _iter_existing_route_candidates(route_file, queue_file, tripinfo_file, scenario_name):
        vehicle_types = {}
        try:
            for _event, elem in ET.iterparse(candidate, events=("start",)):
                if elem.tag not in ("vehicle", "trip"):
                    continue
                veh_id = elem.get("id")
                veh_type = elem.get("type")
                if veh_id and veh_type:
                    vehicle_types[veh_id] = veh_type
            logger.debug(f"[EVAL] 已从路由文件读取车辆类型: {candidate}, count={len(vehicle_types)}")
            return vehicle_types
        except Exception as e:
            logger.warning(f"[EVAL] 解析路由文件车辆类型失败: {e} - {candidate}")
    return {}


def _count_regular_route_trips(event_id_prefixes,
                               route_file: str = None,
                               queue_file: str = None,
                               tripinfo_file: str = None,
                               scenario_name: str = None):
    """统计 raw .rou.xml 中排除事件前缀后的普通车辆总行程数。"""
    for candidate in _iter_existing_route_candidates(route_file, queue_file, tripinfo_file, scenario_name):
        vehicle_ids = set()
        try:
            for _event, elem in ET.iterparse(candidate, events=("start",)):
                if elem.tag not in ("vehicle", "trip"):
                    continue
                veh_id = elem.get("id")
                if veh_id and not _is_event_vehicle_id(veh_id, event_id_prefixes):
                    vehicle_ids.add(veh_id)
            logger.debug(f"[EVAL] 已从路由文件统计普通车辆总行程数: {candidate}, count={len(vehicle_ids)}")
            return len(vehicle_ids)
        except Exception as e:
            logger.warning(f"[EVAL] 统计路由文件普通车辆总行程数失败: {e} - {candidate}")
    return None


def _resolve_raw_route_file(scenario_name: str, route_stem: str) -> str:
    """根据 data/eval 中的路由 stem 解析 data/raw/{scenario}/env 下的 .rou.xml。"""
    env_dir = os.path.join(PROJECT_ROOT, "data", "raw", scenario_name, "env")
    candidates = []
    if route_stem.endswith(".rou.xml"):
        candidates.append(os.path.join(env_dir, route_stem))
    elif route_stem.endswith(".rou"):
        candidates.append(os.path.join(env_dir, f"{route_stem}.xml"))
        candidates.append(os.path.join(env_dir, route_stem))
    elif route_stem.endswith(".xml"):
        candidates.append(os.path.join(env_dir, route_stem))
    else:
        candidates.append(os.path.join(env_dir, f"{route_stem}.rou.xml"))
        candidates.append(os.path.join(env_dir, f"{route_stem}.xml"))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return ""


def _event_id_prefixes_for_route(route_stem: str):
    if "accident_debris" in route_stem:
        return ["accident_", "debris_"]
    if "accident" in route_stem:
        return ["accident_"]
    if "debris" in route_stem:
        return ["debris_"]
    if "pedestrian" in route_stem or "ped_" in route_stem:
        return ["pedestrian_", "ped_"]
    return None


class MetricsCalculator:
    """交通指标计算器，支持在线（仿真步）和离线（SUMO 文件）两种模式。"""

    def __init__(self):
        self.reset()

    def reset(self):
        # {veh_id: {start_time, end_time, trip_time, waiting_time, type}}
        self.vehicle_data = {}
        self.queue_lengths = []

    def update(self, step_data):
        """在线模式：逐步更新仿真数据。"""
        if 'current_queue_lengths' in step_data:
            self.queue_lengths.append(np.mean(step_data['current_queue_lengths']))

        if 'arrived_vehicles' in step_data:
            for veh in step_data['arrived_vehicles']:
                self.vehicle_data[veh['id']] = veh

    def compute_metrics(self, scenario_name=None):
        """在线模式：汇总计算最终指标。"""
        num_junctions = 1
        if scenario_name and scenario_name in SCENARIO_CONFIGS:
            junction_name = SCENARIO_CONFIGS[scenario_name].get("JUNCTION_NAME", [])
            num_junctions = len(junction_name) if isinstance(junction_name, list) else 1

        all_travel_times  = [v['travel_time'] for v in self.vehicle_data.values()]
        all_waiting_times = [v['waiting_time'] for v in self.vehicle_data.values()]
        queue_mean = np.mean(self.queue_lengths) if self.queue_lengths else 0.0

        metrics = {
            "ATT":   np.mean(all_travel_times)  if all_travel_times  else 0.0,
            "AWT":   np.mean(all_waiting_times) if all_waiting_times else 0.0,
            "AQL":   queue_mean / num_junctions,
            "MaxQL": max(self.queue_lengths) if self.queue_lengths else 0.0,
            "TCR":   0.0,
            "EATT":  0.0,
            "EAWT":  0.0,
            "BATT":  0.0,
            "BAWT":  0.0,
            "Special_ATT": 0.0,
            "Special_AWT": 0.0,
        }

        emergency_travel_times = [
            v['travel_time'] for v in self.vehicle_data.values()
            if v.get('type') in _DEFAULT_EMERGENCY_VTYPES
        ]
        emergency_waiting_times = [
            v['waiting_time'] for v in self.vehicle_data.values()
            if v.get('type') in _DEFAULT_EMERGENCY_VTYPES
        ]
        bus_travel_times = [
            v['travel_time'] for v in self.vehicle_data.values()
            if v.get('type') in _DEFAULT_BUS_VTYPES
        ]
        bus_waiting_times = [
            v['waiting_time'] for v in self.vehicle_data.values()
            if v.get('type') in _DEFAULT_BUS_VTYPES
        ]
        metrics["EATT"] = _mean_or_zero(emergency_travel_times)
        metrics["EAWT"] = _mean_or_zero(emergency_waiting_times)
        metrics["BATT"] = _mean_or_zero(bus_travel_times)
        metrics["BAWT"] = _mean_or_zero(bus_waiting_times)
        metrics["Special_ATT"] = _mean_or_zero(emergency_travel_times + bus_travel_times)
        metrics["Special_AWT"] = _mean_or_zero(emergency_waiting_times + bus_waiting_times)

        return metrics

    # ------------------------------------------------------------------
    def calculate_from_files(
        self,
        statistic_file: str,
        queue_file: str,
        scenario_name: str,
        tripinfo_file: str = None,
        event_id_prefixes: list = None,
        special_vtypes: set = None,
        route_file: str = None,
        emergency_vtypes: set = None,
        bus_vtypes: set = None,
    ) -> dict:
        """
        离线模式：从 SUMO 输出文件计算全套指标。

        Args:
            statistic_file    : statistic_output.xml 路径
            queue_file        : queue_output.xml 路径
            scenario_name     : 场景名称（用于获取路口数量做 AQL 归一化）
            tripinfo_file     : tripinfo.out.xml 路径（可选）
            event_id_prefixes : 事件车辆 ID 前缀列表，如 ["accident_", "pedestrian_"]
                                提供后：
                                  - ATT/AWT 从 tripinfo 重算（排除事件车辆）
                                  - TCR = 完成普通车辆行程数 / 路由普通车辆总行程数
                                不提供时 ATT/AWT 来自 statistic_output.xml（向后兼容）
            special_vtypes    : 合并专项车辆 vType 集合
                                默认统计紧急车辆 + 公交/校车
            route_file        : 原始 .rou.xml 路由文件路径；提供后会按 vehicle id
                                将 tripinfo 与路由文件中的 type 匹配，以计算特殊车辆指标
            emergency_vtypes  : 紧急车辆 vType 集合（默认 emergency/police/fire_engine 等）
            bus_vtypes        : 公交/校车 vType 集合（默认 bus/public_bus/school_bus）

        Returns:
            dict: {ATT, AWT, AQL, MaxQL, TCR, EATT, EAWT, BATT, BAWT,
                   Special_ATT, Special_AWT}
        """
        if special_vtypes is None:
            special_vtypes = _DEFAULT_SPECIAL_VTYPES
        if emergency_vtypes is None:
            emergency_vtypes = _DEFAULT_EMERGENCY_VTYPES
        if bus_vtypes is None:
            bus_vtypes = _DEFAULT_BUS_VTYPES

        metrics = {
            "ATT": 0.0, "AWT": 0.0, "AQL": 0.0,
            "MaxQL": 0.0, "TCR": 0.0,
            "EATT": 0.0, "EAWT": 0.0,
            "BATT": 0.0, "BAWT": 0.0,
            "Special_ATT": 0.0, "Special_AWT": 0.0,
        }

        num_junctions = 1
        if scenario_name and scenario_name in SCENARIO_CONFIGS:
            junction_name = SCENARIO_CONFIGS[scenario_name].get("JUNCTION_NAME", [])
            num_junctions = len(junction_name) if isinstance(junction_name, list) else 1

        # ── 1. ATT / AWT ──────────────────────────────────────────────
        if event_id_prefixes is None:
            # 向后兼容：从 statistic_output.xml 读取聚合均值
            try:
                tree = ET.parse(statistic_file)
                root = tree.getroot()
                veh_stats = root.find('vehicleTripStatistics')
                if veh_stats is not None:
                    metrics["ATT"] = float(veh_stats.get('duration', 0.0))
                    metrics["AWT"] = float(veh_stats.get('waitingTime', 0.0))
            except Exception as e:
                logger.error(f"[EVAL] 解析 statistic 文件失败: {e} - {statistic_file}")
        else:
            # 事件场景：从 tripinfo 过滤事件车辆后重算 ATT/AWT
            if tripinfo_file and os.path.exists(tripinfo_file):
                try:
                    tree = ET.parse(tripinfo_file)
                    root = tree.getroot()
                    reg_durations, reg_waits = [], []
                    arrived_regular_count = 0
                    for ti in root.findall('tripinfo'):
                        veh_id = ti.get('id', '')
                        # 排除所有前缀匹配的事件车辆
                        if _is_event_vehicle_id(veh_id, event_id_prefixes):
                            continue
                        # arrival="-1.00" 表示仿真结束时车辆仍在路网中，未完成行程，不计入 TCR 分子
                        actually_arrived = float(ti.get('arrival', -1.0)) >= 0.0
                        if actually_arrived:
                            arrived_regular_count += 1
                        dur = ti.get('duration')
                        wt  = ti.get('waitingTime')
                        if dur is not None:
                            reg_durations.append(float(dur))
                        if wt is not None:
                            reg_waits.append(float(wt))
                    metrics["ATT"] = float(np.mean(reg_durations)) if reg_durations else 0.0
                    metrics["AWT"] = float(np.mean(reg_waits))     if reg_waits     else 0.0
                    regular_total = _count_regular_route_trips(
                        event_id_prefixes,
                        route_file=route_file,
                        queue_file=queue_file,
                        tripinfo_file=tripinfo_file,
                        scenario_name=scenario_name,
                    )
                    if regular_total:
                        # TCR = 普通车辆行程完成率，消除不同路网总车辆数差异。
                        metrics["TCR"] = arrived_regular_count / regular_total
                    else:
                        logger.warning(
                            "[EVAL] 无法计算 TCR：未找到普通车辆总行程数或总数为 0，"
                            f"arrived_regular={arrived_regular_count}, route_file={route_file}"
                        )
                except Exception as e:
                    logger.error(f"[EVAL] 解析 tripinfo（事件过滤）失败: {e} - {tripinfo_file}")
            else:
                # tripinfo 不存在时降级到 statistic_output.xml
                logger.warning(f"[EVAL] event_id_prefixes 已指定但 tripinfo 不存在，降级读取 statistic_output。")
                try:
                    tree = ET.parse(statistic_file)
                    root = tree.getroot()
                    veh_stats = root.find('vehicleTripStatistics')
                    if veh_stats is not None:
                        metrics["ATT"] = float(veh_stats.get('duration', 0.0))
                        metrics["AWT"] = float(veh_stats.get('waitingTime', 0.0))
                except Exception as e:
                    logger.error(f"[EVAL] 解析 statistic 文件失败: {e} - {statistic_file}")

        # ── 2. AQL / MaxQL ─────────────────────────────────────────────
        try:
            tree = ET.parse(queue_file)
            root = tree.getroot()
            step_totals = []
            for data in root.findall('data'):
                lanes = data.find('lanes')
                step_queue = 0.0
                if lanes is not None:
                    for lane in lanes.findall('lane'):
                        step_queue += float(lane.get('queueing_length', 0.0))
                step_totals.append(step_queue)

            if step_totals:
                metrics["AQL"]   = (float(np.mean(step_totals))) / num_junctions
                metrics["MaxQL"] = float(np.max(step_totals))
        except Exception as e:
            logger.error(f"[EVAL] 解析 queue 文件失败: {e} - {queue_file}")

        # ── 3. EATT/EAWT/BATT/BAWT ────────────────────────────────────
        if tripinfo_file and os.path.exists(tripinfo_file):
            try:
                route_vehicle_types = _load_route_vehicle_types(
                    route_file=route_file,
                    queue_file=queue_file,
                    tripinfo_file=tripinfo_file,
                    scenario_name=scenario_name,
                )
                tree = ET.parse(tripinfo_file)
                root = tree.getroot()
                emergency_durations, emergency_waits = [], []
                bus_durations, bus_waits = [], []
                sp_durations, sp_waits = [], []
                for ti in root.findall('tripinfo'):
                    veh_id = ti.get('id', '')
                    veh_type = route_vehicle_types.get(veh_id, ti.get('vType', ''))
                    dur = ti.get('duration')
                    wt = ti.get('waitingTime')
                    if dur is None and wt is None:
                        continue

                    dur_val = float(dur) if dur is not None else None
                    wt_val = float(wt) if wt is not None else None

                    if veh_type in emergency_vtypes:
                        if dur_val is not None:
                            emergency_durations.append(dur_val)
                        if wt_val is not None:
                            emergency_waits.append(wt_val)
                    if veh_type in bus_vtypes:
                        if dur_val is not None:
                            bus_durations.append(dur_val)
                        if wt_val is not None:
                            bus_waits.append(wt_val)
                    if veh_type in special_vtypes:
                        if dur_val is not None:
                            sp_durations.append(dur_val)
                        if wt_val is not None:
                            sp_waits.append(wt_val)

                metrics["EATT"] = _mean_or_zero(emergency_durations)
                metrics["EAWT"] = _mean_or_zero(emergency_waits)
                metrics["BATT"] = _mean_or_zero(bus_durations)
                metrics["BAWT"] = _mean_or_zero(bus_waits)
                metrics["Special_ATT"] = _mean_or_zero(sp_durations)
                metrics["Special_AWT"] = _mean_or_zero(sp_waits)
            except Exception as e:
                logger.error(f"[EVAL] 解析 tripinfo（专项车辆）失败: {e} - {tripinfo_file}")

        return metrics


if __name__ == "__main__":
    secnario_dict = {
        "JiNan": [
        #     # "anon_3_4_jinan_real",
        #     # "anon_3_4_jinan_real_2000",
        #     # "anon_3_4_jinan_real_2500",
                # "anon_3_4_jinan_real_accident",
                # "anon_3_4_jinan_real_bus",
                # "anon_3_4_jinan_real_bus",
                # "anon_3_4_jinan_real_debris",
                # "anon_3_4_jinan_real_debris_专为产数据",
                # "anon_3_4_jinan_real_emergy"
            "anon_3_4_jinan_real_accident_debris",
        #     "anon_3_4_jinan_real_emergy_bus",
        #     # "anon_3_4_jinan_real_2500",
        #     # "anon_3_4_jinan_synthetic_24000_60min.rou",
        ],
        "Hangzhou": [
            "anon_4_4_hangzhou_real_accident_debris",
        #     "anon_4_4_hangzhou_real_emergy_bus",
        #     "anon_4_4_hangzhou_real.rou",
        #     "anon_4_4_hangzhou_real_5816.rou",
        #     "anon_4_4_hangzhou_synthetic_24000_60min.rou",
        ],
        "France_Massy":
        [
            "massy_accident_debris",
        #     "massy_emergy_bus"
        ],
        "Hongkong_YMT":
        [
            "YMT_accident_debris",
        #     "YMT_emergy_bus"
        #     "YMT_accident",
        #     "YMT_bus",
        #         "YMT_debris",
        #         "YMT_debris_1",
        #         "YMT_emergy"

        ],
        "SouthKorea_Songdo":
        [
            "songdo_accident_debris",
        #     "songdo_emergy_bus"
        ],
        "NewYork":
        [
            "anon_28_7_newyork_real_double_accident_debris",
            # "anon_28_7_newyork_real_double_emergy_bus"
        ]
    }
    # methods = ['fixed_time', 'max_pressure', 'qwen3-vl-8b_temp_0','qwen3-vl-8b_temp_0.7','random','gemini3.1_pro','qwen3-vl-8b_wo_Cyclic']  # 'intellilight', 'frap'
    methods = ['qwen3-vl-8b']  # 'intellilight', 'fr
    for secnario_name, net_files in secnario_dict.items():
        for net_file in net_files:
            for method in methods:
                base_path  = f"data/eval/{secnario_name}/{net_file}/{method}"
                stat_path  = f"{base_path}/statistic_output.xml"
                queue_path = f"{base_path}/queue_output.xml"
                tripinfo_path = f"{base_path}/tripinfo.out.xml"
                route_path = _resolve_raw_route_file(secnario_name, net_file)
                event_id_prefixes = _event_id_prefixes_for_route(net_file)
                if not os.path.exists(stat_path) or not os.path.exists(queue_path):
                    print(f"⚠️  [Skip] 文件缺失，跳过目录: {base_path}")
                    continue

                try:
                    calculator = MetricsCalculator()
                    metrics = calculator.calculate_from_files(
                        stat_path,
                        queue_path,
                        secnario_name,
                        tripinfo_file=tripinfo_path,
                        route_file=route_path,
                        event_id_prefixes=event_id_prefixes,
                    )
                    print(f"✅ {secnario_name}/{net_file}/{method}", metrics)
                except Exception as e:
                    print(f"❌ {secnario_name}/{net_file}/{method} 处理出错: {e}")
                    continue
