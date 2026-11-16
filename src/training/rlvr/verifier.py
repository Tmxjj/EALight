"""RLVR reward 计算主入口。"""

from __future__ import annotations

from typing import Any, Dict

from .parser.action_parser import ActionParseError, parse_action
from .parser.event_recognition_parser import parse_event_recognition
from .parser.lane_analysis_parser import parse_lane_analysis
from .reward.composer import compose_r_grpo, compose_r_vision
from .reward.constants import R_GRPO_FALLBACK
from .reward.event_reward import compute_r_event_recog
from .reward.lane_reward import compute_r_lane


def compute_rlvr_reward(
    response_text: str,
    sample: Dict[str, Any],
    rollout_evaluator,
) -> Dict[str, Any]:
    """把 response_text、样本 GT 和离线 rollout reward 组合成最终 RLVR reward。

    异常优先级：
    1. Action 解析失败 / phase 越界 / rollout reward 缺失 → 直接返回 R_GRPO = -3.0；
    2. Lane Analysis / Event Recognition 解析失败 → 子项置 0，不让格式问题破坏整个样本；
    3. 其余字段尽量完整返回，方便训练日志和后续排查。
    """
    num_phases = int(sample.get("num_phases", 0) or 0) or None

    try:
        phase_id, duration = parse_action(response_text, num_phases=num_phases)
        rollout_result = rollout_evaluator.lookup(phase_id, duration)
    except (ActionParseError, Exception) as exc:
        return {
            "reward": float(R_GRPO_FALLBACK),
            "r_traffic": None,
            "r_vision": None,
            "r_mix": None,
            "r_base": None,
            "r_overlay": None,
            "r_lane": 0.0,
            "r_event_recog": 0.0,
            "emergency_wait_norm": None,
            "transit_wait_norm": None,
            "is_event_sample": bool(sample.get("event_gt")),
            "parsed_action": None,
            "debug": {"error": str(exc)},
        }

    lane_pred = parse_lane_analysis(response_text)
    event_pred = parse_event_recognition(response_text)

    gt_lane = sample.get("lane_level_gt") or {}
    controlled_lane_keys = set(sample.get("controlled_lane_keys") or gt_lane.keys())
    r_lane, lane_components = compute_r_lane(
        pred_lane_levels=lane_pred or {},
        gt_lane_levels=gt_lane,
        controlled_lane_keys=controlled_lane_keys,
    )

    gt_events = sample.get("event_gt") or []
    r_event_recog, event_components = compute_r_event_recog(
        pred_events=event_pred or [],
        gt_events=gt_events,
    )

    is_event_sample = bool(gt_events)
    r_vision, vision_components = compose_r_vision(r_lane, r_event_recog, is_event_sample)

    r_traffic = float(rollout_result.get("r_traffic", 0.0))
    r_grpo, mix_components = compose_r_grpo(r_traffic, r_vision)

    return {
        "reward": float(r_grpo),
        "r_traffic": r_traffic,
        "r_vision": float(r_vision),
        "r_mix": float(mix_components["r_mix"]),
        "r_base": rollout_result.get("r_base"),
        "r_overlay": rollout_result.get("r_overlay"),
        "r_lane": float(r_lane),
        "r_event_recog": float(r_event_recog),
        "emergency_wait_norm": rollout_result.get("emergency_wait_norm"),
        "transit_wait_norm": rollout_result.get("transit_wait_norm"),
        "is_event_sample": is_event_sample,
        "parsed_action": {"phase_id": phase_id, "duration": duration},
        "lane_components": lane_components,
        "event_components": event_components,
        "vision_components": vision_components,
        "mix_components": mix_components,
        "debug": {
            "rollout_result": rollout_result,
        },
    }
