"""视觉 reward 与最终 GRPO reward 的合成层。

这里不再做任何语义判断，只负责把 lane reward、event reward 和 traffic reward
按照设计文档中的固定权重合成，并在数值上做最后一层裁剪。
"""

from typing import Dict, Tuple

from .constants import (
    R_GRPO_SCALE,
    W_LANE_EVENT_SAMPLE,
    W_LANE_NORMAL_SAMPLE,
    W_TRAFFIC,
    W_VISION,
)


def _clip(value: float, lower: float, upper: float) -> float:
    """统一的裁剪函数，避免各文件各写一套边界判断。"""
    return max(lower, min(upper, value))

def compose_r_vision(r_lane: float, r_event_recog: float, is_event_sample: bool) -> Tuple[float, dict]:
    # 事件样本要更重视事件识别，非事件样本则更重视车道密度理解。
    w_lane, w_event = W_LANE_EVENT_SAMPLE if is_event_sample else W_LANE_NORMAL_SAMPLE
    r_vision_raw = w_lane * r_lane + w_event * r_event_recog
    
    # 理论上已经落在 [-1, 1]，这里再裁一次是为了防止浮点误差。
    r_vision = _clip(r_vision_raw, -1.0, 1.0)
    
    return r_vision, {"w_lane": w_lane, "w_event": w_event, "r_vision_raw": r_vision_raw, "r_vision": r_vision}

def compose_r_grpo(r_traffic: float, r_vision: float) -> Tuple[float, dict]:
    # 最终训练 reward 采用 traffic / vision 的固定融合比例。
    r_mix = W_TRAFFIC * r_traffic + W_VISION * r_vision
    r_grpo = R_GRPO_SCALE * r_mix
    
    # 训练时保留一个稳定的数值边界，防止极端样本把梯度推得过大。
    r_grpo = _clip(r_grpo, -R_GRPO_SCALE, R_GRPO_SCALE)
    
    return r_grpo, {"r_mix": r_mix, "r_grpo": r_grpo}