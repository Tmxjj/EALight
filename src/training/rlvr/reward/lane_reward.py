"""车道密度感知 reward（R_lane，设计文档 §2.2.1）。

该模块只负责一件事：把模型在 Lane Analysis 里输出的等级文本，和 GT 车道等级
做 soft 评分比较。不混入交通效率 reward，也不处理事件识别逻辑。

评分规则（soft scoring）：
  - err=0 → +1.0（完全正确）
  - err=1 → +0.7（相邻档位，视觉上本就模糊）
  - err=2 →  0.0
  - err=3 → -0.5
  - err=4 → -1.0（跨四档，不可能误判）
  - 漏报受控 lane → -1.0
  - 伪造（写入不受控 lane）→ -0.5
  - 整段解析失败（pred=None/空）→ R_lane = 0.0（中性，避免训练初期崩塌）
"""

from typing import Dict, Optional, Set, Tuple

from .constants import (
    LANE_FAKE_SCORE,
    LANE_MISSING_SCORE,
    LANE_SCORE_BY_ERR,
    LEVEL_TO_ID,
)


def _level_id(level: str) -> Optional[int]:
    """安全把 level 字符串转成 int id；非法返回 None。"""
    if level is None:
        return None
    return LEVEL_TO_ID.get(str(level))


def compute_r_lane(
    *,
    pred_lane_levels: Optional[Dict[str, str]],
    gt_lane_levels: Dict[str, str],
    controlled_lane_keys: Set[str],
) -> Tuple[float, dict]:
    """软评分聚合并返回完整诊断字段。

    参数：
        pred_lane_levels  - 解析器输出的 {prompt_key: level_str}，None 视为整段解析失败
        gt_lane_levels    - ground truth {prompt_key: level_str}
        controlled_lane_keys - 受信号控制的 lane 的 prompt_key 集合

    返回：(R_lane ∈ [-1, 1], components_dict)
    """
    # 整段解析失败 → 中性分（避免崩塌）
    if pred_lane_levels is None or len(pred_lane_levels) == 0:
        return 0.0, {
            "fallback_neutral": True,
            "n_scored": 0,
            "n_missing": 0,
            "n_fake": 0,
        }

    scores = []
    n_missing = 0
    n_fake = 0

    # 1) 受控 lane：按 soft 评分或漏报打分
    for key in controlled_lane_keys:
        gt_id = _level_id(gt_lane_levels.get(key))
        pred_id = _level_id(pred_lane_levels.get(key))

        if pred_id is None:
            # 受控 lane 没报出来，或者 level 字符串非法，均按漏报处理
            scores.append(LANE_MISSING_SCORE)
            n_missing += 1
            continue

        if gt_id is None:
            # 该 lane 的 GT 缺失（理论上不应出现），跳过，不计入分母
            continue

        err = abs(pred_id - gt_id)
        scores.append(LANE_SCORE_BY_ERR.get(err, -1.0))

    # 2) 不受控但模型仍写出 → 伪造惩罚（计入分母）
    for key in pred_lane_levels:
        if key not in controlled_lane_keys:
            scores.append(LANE_FAKE_SCORE)
            n_fake += 1

    if not scores:
        # 无任何评分项（controlled_lane_keys 为空，且 pred 也为空）→ 中性
        return 0.0, {
            "fallback_neutral": True,
            "n_scored": 0,
            "n_missing": n_missing,
            "n_fake": n_fake,
        }

    r_lane = float(max(-1.0, min(1.0, sum(scores) / len(scores))))
    return r_lane, {
        "fallback_neutral": False,
        "n_scored": len(scores),
        "n_missing": n_missing,
        "n_fake": n_fake,
        "raw_score": r_lane,
    }
