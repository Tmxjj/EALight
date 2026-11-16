"""事件识别 reward（R_event_recog，设计文档 §2.2.2）。

该模块评价 CoT 中 Event Recognition 段的识别质量：
  - 类型是否正确（S_type，权重 0.3）
  - 所在车道是否正确（S_lane，权重 0.3）
  - 受影响相位是否正确（S_phase，权重 0.4，三者中最高）
  - 是否有幻觉（多余 pred 计为幻觉，每个扣 0.2，封顶 3 次）

匹配规则：
  1. 已被某 gt event 匹配的 pred event 不可再次被其他 gt event 匹配；
  2. 在所有可能配对中选择全局最高分，避免多事件样本受 pred_events /
     gt_events 顺序影响；
  3. 没有匹配到任何有效 pred 的 gt 记 S_type=S_lane=S_phase=0（得 -1.0）；
  4. 多余的 pred（未匹配任何 gt）视为幻觉，每个扣 0.2，封顶 3 次。
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from .constants import (
    HALLUCINATION_CAP,
    HALLUCINATION_PENALTY,
    NO_EVENT_HALLUCINATION_PENALTY,
    S_LANE_WEIGHT,
    S_PHASE_WEIGHT,
    S_TYPE_WEIGHT,
)


def _single_event_score(s_type: int, s_lane: int, s_phase: int) -> float:
    """2(0.3·S_type + 0.3·S_lane + 0.4·S_phase) − 1，结果 ∈ [-1, 1]。"""
    return 2.0 * (S_TYPE_WEIGHT * s_type + S_LANE_WEIGHT * s_lane + S_PHASE_WEIGHT * s_phase) - 1.0


def _event_has_key(event: Any, field: str) -> bool:
    return isinstance(event, dict) and field in event


def _event_value(event: Any, field: str) -> Any:
    if not isinstance(event, dict):
        return None
    value = event.get(field)
    if isinstance(value, str):
        value = value.strip()
        if field == "affected_phase":
            if value.lower() == "none":
                return None
            if value.isdigit():
                return int(value)
        return value
    return value


def _field_matches(gt: Dict, pred: Dict, field: str) -> bool:
    """比较字段，缺字段不算命中；显式 None 只在双方都有该字段时算命中。"""
    if not _event_has_key(gt, field) or not _event_has_key(pred, field):
        return False
    return _event_value(gt, field) == _event_value(pred, field)


def _has_non_empty_lane(event: Dict) -> bool:
    lane = _event_value(event, "located_lane")
    return lane is not None and lane != ""


def _score_pred_for_gt(gt: Dict, pred: Dict) -> float:
    """计算一个 pred/gt 配对的单事件分数。"""
    s_type = 1 if _field_matches(gt, pred, "event_type") else 0
    s_lane = 1 if _field_matches(gt, pred, "located_lane") else 0
    s_phase = 1 if _field_matches(gt, pred, "affected_phase") else 0
    return _single_event_score(s_type, s_lane, s_phase)


def _can_match_pred_to_gt(gt: Dict, pred: Dict) -> bool:
    """判断 pred 是否是 gt 的有效候选。

    同类型事件始终可作为候选，以保留 type-only 部分分；跨类型事件只有在车道相同
    时才允许配对，避免仅 phase 碰巧相同的无关 pred 被吞掉。
    """
    if _field_matches(gt, pred, "event_type"):
        return True
    return _has_non_empty_lane(gt) and _has_non_empty_lane(pred) and _field_matches(
        gt, pred, "located_lane"
    )


def _optimal_event_match_scores(
    gt_events: List[Dict],
    pred_events: List[Dict],
) -> Tuple[List[float], Set[int]]:
    """返回顺序无关的最优 gt 分数列表和已使用 pred 下标集合。

    多事件样本中，同类型事件可能重复出现。如果按 gt 顺序贪心匹配，较早的 gt 可能
    抢走另一个 gt 的完美匹配 pred。这里用小规模动态搜索枚举“当前 gt 漏报”或“当前
    gt 匹配任一剩余有效 pred”，以最大化最终 reward 的原始值：
    sum(per_gt_scores) / n_gt - hallucination_penalty * n_unmatched_pred。
    """
    miss_score = _single_event_score(0, 0, 0)
    n_gt = len(gt_events)
    n_pred = len(pred_events)
    match_options: List[List[Tuple[int, float]]] = []
    for gt in gt_events:
        options: List[Tuple[int, float]] = []
        for pred_idx, pred in enumerate(pred_events):
            if _can_match_pred_to_gt(gt, pred):
                options.append((pred_idx, _score_pred_for_gt(gt, pred)))
        match_options.append(options)

    @lru_cache(maxsize=None)
    def _search(gt_idx: int, used_mask: int) -> Tuple[Tuple[float, int, int], Tuple[float, ...], int]:
        if gt_idx >= n_gt:
            n_halluc = n_pred - bin(used_mask).count("1")
            n_capped = min(n_halluc, HALLUCINATION_CAP)
            raw = -HALLUCINATION_PENALTY * n_capped
            return (raw, 0, -n_halluc), (), used_mask

        best_key, best_scores, best_mask = _search(gt_idx + 1, used_mask)
        best_key = (best_key[0] + miss_score / max(1, n_gt), best_key[1], best_key[2])
        best_scores = (miss_score,) + best_scores

        for pred_idx, score in match_options[gt_idx]:
            bit = 1 << pred_idx
            if used_mask & bit:
                continue
            key, scores, mask = _search(gt_idx + 1, used_mask | bit)
            candidate_key = (
                key[0] + score / max(1, n_gt),
                key[1] + 1,
                key[2],
            )
            candidate_scores = (score,) + scores
            if candidate_key > best_key:
                best_key = candidate_key
                best_scores = candidate_scores
                best_mask = mask

        return best_key, best_scores, best_mask

    _, scores, used_mask = _search(0, 0)
    used_pred_indices: Set[int] = {idx for idx in range(n_pred) if used_mask & (1 << idx)}
    return list(scores), used_pred_indices


def compute_r_event_recog(
    *,
    pred_events: Optional[List[Dict]],
    gt_events: List[Dict],
) -> Tuple[float, dict]:
    """计算事件识别 reward 并返回完整诊断字段。

    参数：
        pred_events  - 解析器输出的事件列表；None 表示整段解析失败，返回中性分
        gt_events    - ground truth 事件列表

    返回：(R_event_recog ∈ [-1, 1], components_dict)
    """
    # 整段解析失败 → 中性分
    if pred_events is None:
        return 0.0, {
            "case": "parse_failed",
            "fallback_neutral": True,
            "n_halluc_capped": 0,
        }

    gt_events = list(gt_events or [])
    pred_events = list(pred_events or [])

    # ── 无事件样本 ──
    if not gt_events:
        if not pred_events:
            return 1.0, {
                "case": "no_event_correct",
                "fallback_neutral": False,
                "n_halluc_capped": 0,
            }
        return float(-NO_EVENT_HALLUCINATION_PENALTY), {
            "case": "no_event_hallucination",
            "fallback_neutral": False,
            "n_halluc_capped": 0,
        }

    # ── 有事件样本：全局最优匹配，避免多事件样本受列表顺序影响 ──
    per_gt_scores, used_pred_indices = _optimal_event_match_scores(gt_events, pred_events)

    # 幻觉惩罚：未被匹配的 pred 数量，封顶 HALLUCINATION_CAP
    n_halluc = len(pred_events) - len(used_pred_indices)
    n_capped = min(n_halluc, HALLUCINATION_CAP)
    base = sum(per_gt_scores) / max(1, len(per_gt_scores))
    raw = base - HALLUCINATION_PENALTY * n_capped
    r = float(max(-1.0, min(1.0, raw)))

    miss_score = _single_event_score(0, 0, 0)
    return r, {
        "case": "event_match",
        "fallback_neutral": False,
        "n_halluc_capped": int(n_capped),
        "n_matched": sum(1 for score in per_gt_scores if score != miss_score),
        "n_gt": len(gt_events),
        "base_score": float(base),
    }
