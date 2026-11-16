"""交通效率 reward 的单一来源实现（R_traffic，设计文档 §2.1）。

这个文件是 RLVR / golden generation / 环境 wrapper 三处共用的核心。
设计原则：
  1. 不依赖 SUMO 的在线执行逻辑，只消费已经采集好的 trace；
  2. 不把公式拆散到多个文件里，避免同一 reward 在训练和数据生成阶段不一致；
  3. 对缺字段和空 trace 做保守降级，保证整体不会因为单个样本的局部异常崩溃；
  4. vehicle_wait_times 同时支持 tuple (vid, vtype, wt) 和 dict 两种上游格式。

trace 数据结构（与 tsc_wrapper._copy_tls_rollout_snapshot 完全一致）：
{
    "movement_ids":             list[str],
    "phase2movements":          dict[int|str, list[str]],
    "jam_length_vehicle":       list[float],          # 与 movement_ids 顺序对齐
    "last_step_occupancy":      list[float],          # 同上，单位百分比 [0, 100]
    "last_step_vehicle_id_list": list[list[str]],     # 同上，per detector
    "vehicle_wait_times":       list[tuple|dict],     # (vid, vtype, wt) 或 dict
}
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .constants import (
    EMERGENCY_VTYPES,
    EMERGENCY_WAIT_NORM,
    R_TRAFFIC_DIVISOR,
    THROUGHPUT_VEHICLES_PER_SEC_PER_LANE,
    TRANSIT_VTYPES,
    TRANSIT_WAIT_NORM,
    W_DELTA_Q,
    W_EMERGENCY,
    W_GREEN_WASTE,
    W_OCC_ALL,
    W_QUEUE_UNSERVED,
    W_THROUGHPUT,
    W_TRANSIT,
)


def _clip(value: float, lower: float, upper: float) -> float:
    """统一裁剪函数，保持数值边界行为一致。"""
    return max(lower, min(upper, value))


def _safe_mean(values: Sequence[float]) -> float:
    """空序列返回 0，避免 reward 公式里到处写 if 分支。"""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _norm_occ(value: Any) -> float:
    """把 occupancy 从百分比归一化并裁剪到 [0, 1]。"""
    return _clip(float(value or 0.0) / 100.0, 0.0, 1.0)


def _normalize_vtype(vtype: Any) -> str:
    """统一 vType 字符串格式：去除空格、转小写、空格替换为下划线。

    兼容仿真中出现的 "Ambulance "/"Police Car"/"fire_engine" 等各种写法。
    """
    return str(vtype or "").strip().lower().replace(" ", "_")


def _get_common_movements(phase2movements: Dict[Any, Sequence[str]]) -> set:
    """返回所有相位共有的 movement 集合（即不受信号管控的自由通行车道）。

    出现在每个相位中的 movement 意味着无论哪个相位放行它都始终通行，
    不属于任何相位的专属管控范围，不应计入 selected 或 unserved 的统计。
    只对非空相位列表求交集；少于 2 个非空相位时无法判定公共性，返回空集。
    """
    non_empty_sets = [set(mvs) for mvs in phase2movements.values() if mvs]
    if len(non_empty_sets) < 2:
        return set()
    return set.intersection(*non_empty_sets)


def _get_selected_indices(
    movement_ids: Sequence[str],
    phase2movements: Dict[Any, Sequence[str]],
    selected_phase: int,
) -> List[int]:
    """把 selected_phase 映射成 detector 下标集合。

    不同场景里 phase2movements 的 key 可能是 int 或 str，这里同时兼容。
    出现在所有相位中的公共 movement（如自由右转）不受信号管控，
    从 selected 集合中剔除，避免将其纳入排队消解或通行量统计。
    """
    if not phase2movements:
        return []

    common_movements = _get_common_movements(phase2movements)

    selected_movements = (
        phase2movements.get(selected_phase)
        or phase2movements.get(str(selected_phase))
        or []
    )
    selected_set = set(selected_movements) - common_movements
    return [idx for idx, movement_id in enumerate(movement_ids) if movement_id in selected_set]


def _sum_selected_queue(jam_length_vehicle: Sequence[float], selected_indices: Sequence[int]) -> float:
    """把 selected 相位对应 detector 的排队车辆数加总成 Q^sel。"""
    total = 0.0
    for idx in selected_indices:
        if idx < len(jam_length_vehicle):
            total += float(jam_length_vehicle[idx])
    return total


def _mean_all_occupancy(last_step_occupancy: Sequence[float]) -> float:
    """把 detector 占有率统一归一化到 [0, 1]，clip 防止超界。"""
    if not last_step_occupancy:
        return 0.0
    return float(sum(_norm_occ(v) for v in last_step_occupancy) / len(last_step_occupancy))


def _mean_other_occupancy(last_step_occupancy: Sequence[float], selected_indices: Sequence[int]) -> float:
    """计算非 selected detector 的平均占有率 O^other，归一化并 clip。"""
    selected_set = set(selected_indices)
    other_values = [_norm_occ(occ) for idx, occ in enumerate(last_step_occupancy) if idx not in selected_set]
    return _safe_mean(other_values)


def _count_passed_selected(
    first_round_trace: Sequence[dict],
    selected_indices: Sequence[int],
) -> float:
    """统计第一轮 rollout 区间内 selected detector 的通过车辆数。

    N_passed^sel = ever_seen_ids - final_seen_ids：
    - ever_seen_ids：整段 first_round_trace 里曾出现在 selected detector 上的所有车辆 ID；
    - final_seen_ids：最后一个 step 仍在 selected detector 上的车辆 ID；
    - 差值即"进入后已离开"的通过车辆。

    修复旧实现只做 t0-t1 差集的缺陷：中途进入又离开的车辆会被正确计入。
    """
    if not first_round_trace:
        return 0.0

    ever_seen_ids: set[str] = set()
    final_seen_ids: set[str] = set()

    for step_snapshot in first_round_trace:
        id_list = step_snapshot.get("last_step_vehicle_id_list", []) or []
        for idx in selected_indices:
            if idx < len(id_list):
                ever_seen_ids.update(id_list[idx])

    last_id_list = first_round_trace[-1].get("last_step_vehicle_id_list", []) or []
    for idx in selected_indices:
        if idx < len(last_id_list):
            final_seen_ids.update(last_id_list[idx])

    return float(len(ever_seen_ids - final_seen_ids))


def _extract_vehicle_wait_records(step_snapshot: dict) -> List[dict]:
    """从 trace 中抽取等待时间记录，统一归一化为 dict 格式。

    支持两种上游格式：
    - dict:  {"vid": ..., "vtype": ..., "waiting_time": ...}
    - tuple: (vid, vtype, waiting_time)
    """
    raw_records = step_snapshot.get("vehicle_wait_times", []) or []
    normalized_records: List[dict] = []
    for record in raw_records:
        if isinstance(record, dict):
            normalized_records.append(
                {
                    "vid": record.get("vid") or record.get("vehicle_id"),
                    "vtype": record.get("vtype") or record.get("vehicle_type"),
                    "waiting_time": float(record.get("waiting_time", 0.0) or 0.0),
                }
            )
        elif isinstance(record, (list, tuple)) and len(record) >= 3:
            # 兼容 (vid, vtype, wait_time) 形式
            normalized_records.append(
                {
                    "vid": record[0],
                    "vtype": record[1],
                    "waiting_time": float(record[2] or 0.0),
                }
            )
    return normalized_records


def _aggregate_wait_norm(
    trace: Sequence[dict],
    vtypes: Iterable[str],
    norm_base: float,
) -> Tuple[float, int]:
    """计算一段 rollout 内的等待时间归一化值与 alpha 指示。

    返回 (wait_norm ∈ [0, 1], alpha ∈ {0, 1})。
    对每个 step：sum(WT_v) / (max(1, |X_t|) * norm_base)，再对所有 step 取平均。

    只统计出现在 e2 detector 范围内（last_step_vehicle_id_list）的车辆等待时间；
    不在检测器覆盖范围内的车辆即使 vType 匹配也不计入，避免将远端进场车辆纳入统计。
    vType 匹配在归一化后进行，兼容大小写、空格等变体写法。
    """
    wanted_types = {_normalize_vtype(v) for v in vtypes}
    per_step_scores: List[float] = []
    alpha = 0

    for step_snapshot in trace:
        # 构建本步骤 e2 检测器覆盖到的所有车辆 ID 集合（跨所有 detector 取并集）
        id_list = step_snapshot.get("last_step_vehicle_id_list", []) or []
        e2_vehicle_ids: set = {
            vid for det_ids in id_list for vid in (det_ids or [])
        }

        step_records = _extract_vehicle_wait_records(step_snapshot)
        # 同时满足：vType 命中 + 车辆在 e2 检测器范围内
        matched = [
            r for r in step_records
            if _normalize_vtype(r.get("vtype")) in wanted_types
            and r.get("vid") in e2_vehicle_ids
        ]
        if matched:
            alpha = 1
            denom = max(1, len(matched)) * norm_base
            per_step_scores.append(_clip(sum(r["waiting_time"] for r in matched) / denom, 0.0, 1.0))
        else:
            per_step_scores.append(0.0)

    wait_norm = _clip(_safe_mean(per_step_scores), 0.0, 1.0)
    return wait_norm, alpha


def compute_r_base(
    *,
    initial_snapshot: dict,
    first_round_trace: List[dict],
    second_round_trace: Optional[List[dict]] = None,
    selected_phase: int,
    candidate_duration: int,
) -> Tuple[float, dict]:
    """计算交通效率 reward 的基础项 R_base。

    兼容旧接口的 second_round_trace 参数，但当前 rollout 已收敛为单轮，
    因此仅使用 first_round_trace 作为真实评估窗口。

    5 个基础信号：排队消解、通行效率、未服务方向压力、全局占用率、绿灯浪费。
    不看 emergency / transit 等事件等待项。

    返回：(R_base 原始值, components_dict)
    """
    base_snapshot = initial_snapshot or {}
    rollout_trace = list(first_round_trace or [])

    if not base_snapshot and rollout_trace:
        base_snapshot = rollout_trace[0]

    movement_ids = list(base_snapshot.get("movement_ids", []) or [])
    phase2movements = dict(base_snapshot.get("phase2movements", {}) or {})

    # 公共 movement（出现在所有相位中）对应自由通行车道，不纳入任何相位的统计范围
    common_movements = _get_common_movements(phase2movements)
    common_index_set = {
        idx for idx, mid in enumerate(movement_ids) if mid in common_movements
    }

    selected_indices = _get_selected_indices(movement_ids, phase2movements, selected_phase)
    selected_detector_count = max(1, len(selected_indices))
    selected_index_set = set(selected_indices)

    # 诊断字段：尽早暴露 trace 数据质量问题
    valid_snapshot = bool(movement_ids and phase2movements)
    selected_indices_empty = len(selected_indices) == 0

    # t0 时刻 selected 方向的真实排队量
    q_t0_sel = _sum_selected_queue(
        base_snapshot.get("jam_length_vehicle", []) or [], selected_indices
    )

    # t1 时刻 selected 方向的排队量（rollout 末尾快照）
    if rollout_trace:
        q_t1_sel = _sum_selected_queue(
            rollout_trace[-1].get("jam_length_vehicle", []) or [], selected_indices
        )
    else:
        q_t1_sel = q_t0_sel

    # 1) ΔQ^sel_norm = 1[Q_t0>0] · clip((Q_t0 − Q_t1) / Q_t0, 0, 1)
    if q_t0_sel > 0:
        delta_q_sel_norm = _clip((q_t0_sel - q_t1_sel) / q_t0_sel, 0.0, 1.0)
    else:
        delta_q_sel_norm = 0.0

    # 2) Throughput^sel = clip(N_passed^sel / (0.5·duration·|D_sel|), 0, 1)
    #    N_passed^sel 统计整段 rollout_trace 里曾到达又已离开 selected detector 的车辆，
    #    修复旧实现只差 t0/t1 集合而遗漏中途通过车辆的缺陷。
    passed_count = _count_passed_selected(rollout_trace, selected_indices)
    throughput_denom = (
        THROUGHPUT_VEHICLES_PER_SEC_PER_LANE
        * max(1, candidate_duration)
        * selected_detector_count
    )
    throughput_sel = _clip(passed_count / throughput_denom, 0.0, 1.0)

    # 3) Q̄^unserved = (1/N) Σ_t (1/|D\D_sel|) Σ_{i∈D\D_sel} clip(Q_t^i / 10, 0, 1)
    #    公共 movement（自由通行）同时排除在 unserved 之外，不作为"未服务"方向惩罚。
    excluded_index_set = selected_index_set | common_index_set
    unserved_indices = [i for i in range(len(movement_ids)) if i not in excluded_index_set]
    unserved_values: List[float] = []
    if unserved_indices:
        for step_snapshot in rollout_trace:
            jam_values = step_snapshot.get("jam_length_vehicle", []) or []
            step_vals = [
                _clip(float(jam_values[i]) / 10.0, 0.0, 1.0)
                for i in unserved_indices
                if i < len(jam_values)
            ]
            unserved_values.append(_safe_mean(step_vals))
    q_unserved_bar = _safe_mean(unserved_values)

    # 4) Ō^all = (1/N) Σ_t (1/|D|) Σ_i norm_occ(occ_t^i)
    occ_values = [
        _mean_all_occupancy(step_snapshot.get("last_step_occupancy", []) or [])
        for step_snapshot in rollout_trace
    ]
    o_all_bar = _safe_mean(occ_values)

    # 5) W̄^green：第一轮区间，按"1[Q^sel=0] · clip(remaining_ratio,0,1) · O^other"累计
    green_values: List[float] = []
    if rollout_trace and candidate_duration > 0:
        n1 = len(rollout_trace)
        for step_idx, step_snapshot in enumerate(rollout_trace, start=1):
            q_t_sel = _sum_selected_queue(
                step_snapshot.get("jam_length_vehicle", []) or [], selected_indices
            )
            if q_t_sel > 0:
                green_values.append(0.0)
                continue
            remaining_ratio = _clip(
                (n1 - step_idx) / float(max(candidate_duration, 1)),
                0.0,
                1.0,
            )
            other_occ = _mean_other_occupancy(
                step_snapshot.get("last_step_occupancy", []) or [], selected_indices
            )
            green_values.append(_clip(remaining_ratio * other_occ, 0.0, 1.0))
    w_green_bar = _safe_mean(green_values)

    r_base = (
        W_DELTA_Q * delta_q_sel_norm
        + W_THROUGHPUT * throughput_sel
        - W_QUEUE_UNSERVED * q_unserved_bar
        - W_OCC_ALL * o_all_bar
        - W_GREEN_WASTE * w_green_bar
    )

    components = {
        "selected_phase": int(selected_phase),
        "candidate_duration": int(candidate_duration),
        "num_all_detectors": len(movement_ids),
        "num_selected_detectors": len(selected_indices),
        "q_t0_sel": float(q_t0_sel),
        "q_t1_sel": float(q_t1_sel),
        "delta_q_sel_norm": float(delta_q_sel_norm),
        "throughput_sel": float(throughput_sel),
        "q_unserved_bar": float(q_unserved_bar),
        "o_all_bar": float(o_all_bar),
        "w_green_bar": float(w_green_bar),
        "n_rollout_steps": len(rollout_trace),
        "n_second_round_steps": 0,
        # 诊断字段
        "valid_snapshot": bool(valid_snapshot),
        "selected_indices_empty": bool(selected_indices_empty),
        "selected_indices": list(selected_indices),
    }
    return float(r_base), components


def compute_r_overlay(
    *,
    first_round_trace: List[dict],
    second_round_trace: Optional[List[dict]] = None,
) -> Tuple[float, dict]:
    """计算事件覆盖项 R_overlay 与诊断字段。

    兼容旧接口的 second_round_trace 参数，但当前 rollout 已收敛为单轮，
    因此只使用 first_round_trace 作为事件等待时间统计窗口。

    这一项不看动作本身，只看 rollout 中是否确实存在 emergency / transit 车辆，
    以及这些车辆在路口范围内的等待时间是否被压低。
    """
    rollout_trace = list(first_round_trace or [])

    emergency_wait_norm, alpha_emerg = _aggregate_wait_norm(
        rollout_trace, EMERGENCY_VTYPES, EMERGENCY_WAIT_NORM
    )
    transit_wait_norm, alpha_transit = _aggregate_wait_norm(
        rollout_trace, TRANSIT_VTYPES, TRANSIT_WAIT_NORM
    )

    p_emerg = W_EMERGENCY * alpha_emerg * emergency_wait_norm
    p_transit = W_TRANSIT * alpha_transit * transit_wait_norm
    r_overlay = -(
        1.0
        - (1.0 - p_emerg) * (1.0 - p_transit)
    )
    r_overlay = _clip(r_overlay, -1.0, 0.0)
    components = {
        "emergency_wait_norm": float(emergency_wait_norm),
        "transit_wait_norm": float(transit_wait_norm),
        "alpha_emerg": int(alpha_emerg),
        "alpha_transit": int(alpha_transit),
        "p_emerg": float(p_emerg),
        "p_transit": float(p_transit),
    }
    return float(r_overlay), components


def compute_r_traffic(
    *,
    initial_snapshot: dict,
    first_round_trace: List[dict],
    second_round_trace: Optional[List[dict]] = None,
    selected_phase: int,
    candidate_duration: int,
) -> Dict[str, Any]:
    """合成最终 traffic reward，并输出完整诊断字段。

    R_traffic = clip((R_base + R_overlay) / R_TRAFFIC_DIVISOR, -1, 1)

    这是单一来源输出，wrapper、golden generation、verifier 三处均调用本函数。
    """
    r_base, base_components = compute_r_base(
        initial_snapshot=initial_snapshot,
        first_round_trace=first_round_trace,
        second_round_trace=None,
        selected_phase=selected_phase,
        candidate_duration=candidate_duration,
    )
    r_overlay, overlay_components = compute_r_overlay(
        first_round_trace=first_round_trace,
        second_round_trace=None,
    )
    r_raw = r_base + r_overlay
    r_traffic = _clip(r_raw / R_TRAFFIC_DIVISOR, -1.0, 1.0)

    return {
        "r_traffic": float(r_traffic),
        "r_base": float(r_base),
        "r_overlay": float(r_overlay),
        "r_raw": float(r_raw),
        "base_components": base_components,
        "overlay_components": overlay_components,
        **base_components,
        **overlay_components,
    }


# ──────────────────────────────────────────────────────────────────────
# 自测样例（python -m pytest traffic_reward.py 或直接 python traffic_reward.py）
# ──────────────────────────────────────────────────────────────────────

def _run_self_tests() -> None:
    """覆盖 5 类修复点的最小单元测试，直接断言，不依赖 pytest。"""

    # ── Case 1: Throughput 中间进入又离开 ──────────────────────────────
    # step1: selected detector 空；step2: veh_1 在；step3: 已离开
    trace_c1 = [
        {"last_step_vehicle_id_list": [[], []]},
        {"last_step_vehicle_id_list": [["veh_1"], []]},
        {"last_step_vehicle_id_list": [[], []]},
    ]
    passed = _count_passed_selected(trace_c1, [0])
    assert passed == 1.0, f"Case1 failed: expected 1.0, got {passed}"

    # 旧实现（t0 - t1）会得到 0（t0 为空 → first_ids[0] 为空）
    # 新实现通过 ever_seen - final_seen 正确得到 1

    # ── Case 2: Q_unserved 超过 10 时被 clip ──────────────────────────
    # jam_length_vehicle[0] = 25，unserved_indices = [0]
    snapshot_c2 = {
        "movement_ids": ["m0"],
        "phase2movements": {0: []},   # selected_indices = []
        "jam_length_vehicle": [25.0],
        "last_step_occupancy": [0.0],
        "last_step_vehicle_id_list": [[]],
        "vehicle_wait_times": [],
    }
    # unserved_indices 包含 index 0（因为 selected 为空）
    clipped = _clip(float(25.0) / 10.0, 0.0, 1.0)
    assert clipped == 1.0, f"Case2 failed: expected 1.0, got {clipped}"

    # ── Case 3: occupancy 被 clip 到 [0, 1]
    assert _norm_occ(150) == 1.0, "Case3 failed: _norm_occ(150) should be 1.0 (clipped)"
    assert _norm_occ(50) == 0.5, "Case3 failed: _norm_occ(50) should be 0.5"
    assert _norm_occ(0) == 0.0, "Case3 failed: _norm_occ(0) should be 0.0"

    # ── Case 4: remaining_ratio 被 clip 到 [0, 1] ─────────────────────
    # n1 = 5, step_idx = 1, candidate_duration = 3 → (5-1)/3 = 1.333 → clip 为 1.0
    remaining = _clip((5 - 1) / float(max(3, 1)), 0.0, 1.0)
    assert remaining == 1.0, f"Case4 failed: expected 1.0, got {remaining}"

    # ── Case 5: vType alias 匹配 ─────────────────────────────────────
    emerg_aliases = [
        "Ambulance ", "ambulance", "Police Car", "police_car",
        "Fire Truck", "fire_engine",
    ]
    transit_aliases = ["Public Bus", "bus", "School Bus", "school_bus"]

    emerg_wanted = {_normalize_vtype(v) for v in EMERGENCY_VTYPES}
    transit_wanted = {_normalize_vtype(v) for v in TRANSIT_VTYPES}

    for alias in emerg_aliases:
        norm = _normalize_vtype(alias)
        assert norm in emerg_wanted, (
            f"Case5 failed: '{alias}' → '{norm}' not in EMERGENCY_VTYPES={emerg_wanted}"
        )
    for alias in transit_aliases:
        norm = _normalize_vtype(alias)
        assert norm in transit_wanted, (
            f"Case5 failed: '{alias}' → '{norm}' not in TRANSIT_VTYPES={transit_wanted}"
        )

    print("All self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
