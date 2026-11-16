"""SFT JSONL / RLVR 原始 JSONL → RLVR JSONL 的轻量增强器。

这个模块不负责重新跑 SUMO，也不负责在线采集 GT；它只做两件事：
1. 把 golden generation / SFT / 01_dataset_raw 产物里已有的字段整理成 verifier
   需要的最小集合；
2. 把旧版 all_rollout_rewards 的"纯 float"格式兼容升级为统一的 dict schema。

字段来源说明：
- 1_golden_generation.py 产出字段：image_paths（非 images）、neighbor_messages、
  scenario、junction_id、sumo_step、current_phase_id、num_phases、prompt、
  vlm_response、vlm_action、vlm_thought、best_action、best_reward、
  lane_level_gt、event_gt、controlled_lane_keys、all_rollout_rewards、label、
  rollout_follow_steps。
- SFT 产物字段：images（非 image_paths）等，与上述有差异，均做兼容处理。

这样后续不管是训练前的数据落盘，还是 RLVR 训练时的离线读取，都会使用同一
套字段规范。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_images(sample: Dict[str, Any]) -> List[str]:
    """兼容 image_paths / images 两种写法，统一输出到 images 字段。"""
    images = sample.get("images")
    if not images:
        images = sample.get("image_paths")
    if not images:
        return []
    return [str(path) for path in images]


def _infer_route_name(sample: Dict[str, Any]) -> str:
    """从 image_paths 推断 route 目录名。

    image_paths 典型格式：
      data/rlvr_dataset/{scenario}/{control}/{route_name}/{jid}/{step}/xxx.png
    取倒数第 4 层目录作为 route_name；若无法推断则返回空字符串。
    """
    image_paths: List[str] = sample.get("image_paths") or sample.get("images") or []
    for p in image_paths:
        if not p:
            continue
        parts = Path(p).parts
        # 期望层级：…/route_name/jid/step/img.png -> 倒数第 4 层
        if len(parts) >= 4:
            return parts[-4]
    return ""


def build_sample_id(sample: Dict[str, Any]) -> str:
    """构造 RLVR 样本的唯一 ID。

    格式：{scenario}-{route_name}-step_{sumo_step}-{junction_id}

    route_name 优先从 image_paths 路径中推断，找不到时留空。
    """
    scenario = sample.get("scenario") or "unknown"
    junction_id = sample.get("junction_id") or "unknown"
    sumo_step = sample.get("sumo_step")
    sumo_step_str = str(int(sumo_step)) if sumo_step is not None else "unknown"
    route_name = _infer_route_name(sample)
    return f"{scenario}-{route_name}-step_{sumo_step_str}-{junction_id}"


def _make_unique_sample_id(sample_id: str, seen_sample_ids: set[str]) -> str:
    """为重复 sample_id 追加稳定后缀。

    约定与过滤脚本保持一致：首个重复项使用 ``__dup1``，后续依次递增。
    """
    if sample_id not in seen_sample_ids:
        seen_sample_ids.add(sample_id)
        return sample_id

    duplicate_index = 1
    candidate = f"{sample_id}__dup{duplicate_index}"
    while candidate in seen_sample_ids:
        duplicate_index += 1
        candidate = f"{sample_id}__dup{duplicate_index}"

    seen_sample_ids.add(candidate)
    return candidate


def _normalize_rollout_entry(value: Any) -> Dict[str, Any]:
    """把单个候选动作的 reward 统一成 dict schema，同时保留细分组件。"""
    if isinstance(value, dict):
        components = value.get("components")
        if not isinstance(components, dict):
            components = {}
        base_reward = components.get("r_traffic", value.get("r_traffic", value.get("reward", 0.0)))
        base_components = value.get("base_components", components.get("base_components", {}))
        overlay_components = value.get("overlay_components", components.get("overlay_components", {}))
        if not isinstance(overlay_components, dict):
            overlay_components = {}
        return {
            "r_traffic": float(base_reward or 0.0),
            "r_base": float(components.get("r_base", value.get("r_base", base_reward)) or 0.0),
            "r_overlay": float(components.get("r_overlay", value.get("r_overlay", 0.0)) or 0.0),
            "emergency_wait_norm": float(
                components.get(
                    "emergency_wait_norm",
                    overlay_components.get("emergency_wait_norm", value.get("emergency_wait_norm", 0.0)),
                )
                or 0.0
            ),
            "transit_wait_norm": float(
                components.get(
                    "transit_wait_norm",
                    overlay_components.get("transit_wait_norm", value.get("transit_wait_norm", 0.0)),
                )
                or 0.0
            ),
            "base_components": base_components if isinstance(base_components, dict) else {},
            "overlay_components": overlay_components,
        }

    try:
        reward = float(value)
    except Exception:
        reward = 0.0

    # 旧版 float schema 只记录单一 reward。为了兼容 verifier，其他字段给保底值。
    return {
        "r_traffic": reward,
        "r_base": reward,
        "r_overlay": 0.0,
        "emergency_wait_norm": 0.0,
        "transit_wait_norm": 0.0,
        "base_components": {},
        "overlay_components": {},
    }


def _normalize_all_rollout_rewards(payload: Any) -> Dict[str, Dict[str, Any]]:
    """把 all_rollout_rewards 统一成 {"phase_duration": {components...}} 的字典。"""
    if not isinstance(payload, dict):
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        normalized[str(key)] = _normalize_rollout_entry(value)
    return normalized


def _pick_lane_gt(sample: Dict[str, Any]) -> Dict[str, Any]:
    """尽量从已有样本中保留 lane GT。"""
    lane_gt = sample.get("lane_level_gt")
    if isinstance(lane_gt, dict):
        return lane_gt
    return {}


def _pick_event_gt(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """尽量从已有样本中保留 event GT。"""
    event_gt = sample.get("event_gt")
    if isinstance(event_gt, list):
        return event_gt
    return []


def _build_rlvr_sample(
    sample: Dict[str, Any],
    seen_sample_ids: Optional[set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """把单条原始样本整理成 RLVR 训练样本。

    兼容 1_golden_generation.py（image_paths、neighbor_messages 等字段）与
    SFT 产物（images 字段）两种来源格式。

    如果连 prompt / image 都没有，说明这条样本不是可训练样本，直接返回 None。
    """
    prompt = sample.get("prompt")
    images = _normalize_images(sample)
    if not prompt or not images:
        return None

    lane_level_gt = _pick_lane_gt(sample)
    event_gt = _pick_event_gt(sample)
    all_rollout_rewards = _normalize_all_rollout_rewards(sample.get("all_rollout_rewards"))

    controlled_lane_keys = sample.get("controlled_lane_keys")
    if not controlled_lane_keys:
        controlled_lane_keys = sorted(lane_level_gt.keys())
    elif isinstance(controlled_lane_keys, set):
        controlled_lane_keys = sorted(controlled_lane_keys)
    elif not isinstance(controlled_lane_keys, list):
        controlled_lane_keys = list(controlled_lane_keys)

    return {
        "sample_id": _make_unique_sample_id(build_sample_id(sample), seen_sample_ids)
        if seen_sample_ids is not None
        else build_sample_id(sample),
        "prompt": [{"role": "user", "content": str(prompt)}],
        "images": images,
        "lane_level_gt": lane_level_gt,
        "event_gt": event_gt,
        "all_rollout_rewards": all_rollout_rewards,
        "scenario": sample.get("scenario"),
        "junction_id": sample.get("junction_id"),
        "sumo_step": sample.get("sumo_step"),
        "current_phase_id": sample.get("current_phase_id"),
        "num_phases": int(sample.get("num_phases", 0) or 0),
        "sampling_meta": sample.get("sampling_meta") if isinstance(sample.get("sampling_meta"), dict) else {},
        # verifier 优先读取此字段；若上游未提供，退化为 lane_level_gt 的 key 集合。
        "controlled_lane_keys": controlled_lane_keys,
    }


def enhance_sft_jsonl(input_path: str | Path, output_path: str | Path) -> Dict[str, int]:
    """将输入 JSONL 转成 RLVR 训练可直接消费的 JSONL。

    返回值是简单统计，便于批处理脚本打印日志。
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats = {"read": 0, "written": 0, "skipped": 0}
    seen_sample_ids: set[str] = set()

    with input_file.open("r", encoding="utf-8") as reader, output_file.open("w", encoding="utf-8") as writer:
        for line in reader:
            text = line.strip()
            if not text:
                continue

            stats["read"] += 1
            try:
                sample = json.loads(text)
            except Exception:
                stats["skipped"] += 1
                continue

            rlvr_sample = _build_rlvr_sample(sample, seen_sample_ids)
            if rlvr_sample is None:
                stats["skipped"] += 1
                continue

            writer.write(json.dumps(rlvr_sample, ensure_ascii=False) + "\n")
            stats["written"] += 1

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    """命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="将 SFT/golden JSONL 规范化为 RLVR JSONL")
    parser.add_argument("input_path", type=str, help="输入 JSONL 文件")
    parser.add_argument("output_path", type=str, help="输出 JSONL 文件")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_arg_parser().parse_args()
    stats = enhance_sft_jsonl(args.input_path, args.output_path)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
