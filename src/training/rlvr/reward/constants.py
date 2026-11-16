"""RLVR reward 相关的统一常量区。

这一层只放"不会随着样本变化而变化"的固定量，避免 reward 公式分散到各个
文件里之后出现重复定义、阈值不一致或调参后忘记同步的问题。

所有常量均与 rlvr_reward_design.md 完全对齐，修改时须同步更新设计文档。
"""

# ──────────────────────────────────────────────────────────────────────
# 1. SUMO vType 集合（与项目实际 vtype id 一致）
# ──────────────────────────────────────────────────────────────────────
EMERGENCY_VTYPES = {
    "emergency",
    "ambulance",
    "police",
    "police_car",
    "fire_engine",
    "fire_truck",
}
TRANSIT_VTYPES = {
    "bus",
    "public_bus",
    "school_bus",
}

# ──────────────────────────────────────────────────────────────────────
# 2. 车道密度等级映射（与 prompt_builder.DENSITY_LEVELS 同源）
# ──────────────────────────────────────────────────────────────────────
LEVEL_TO_ID = {
    "Empty": 0,
    "Short": 1,
    "Medium": 2,
    "Long": 3,
    "Critical": 4,
}
ID_TO_LEVEL = {v: k for k, v in LEVEL_TO_ID.items()}

# 阈值表用于车辆数 → 等级映射；区间含义为 [lo, hi]（闭区间）
# 格式：(下界 lo, 上界 hi, 等级名称)
LEVEL_THRESHOLDS = (
    (0, 0, "Empty"),
    (1, 3, "Short"),
    (4, 6, "Medium"),
    (7, 9, "Long"),
    (10, float("inf"), "Critical"),
)


def level_from_count(count) -> str:
    """将整数车辆数映射为密度等级字符串。

    阈值与 LEVEL_THRESHOLDS 完全一致；浮点输入取下界整数；负数按 0 处理。
    """
    n = int(count)
    if n < 0:
        n = 0
    for lo, hi, name in LEVEL_THRESHOLDS:
        if lo <= n <= hi:
            return name
    return "Critical"


# ──────────────────────────────────────────────────────────────────────
# 3. 事件类型集合 / vtype 映射（设计文档 §2.2.2）
# ──────────────────────────────────────────────────────────────────────
EVENT_TYPES = {
    "None",
    "Ambulance",
    "Police",
    "FireTruck",
    "PublicBus",
    "SchoolBus",
    "Accident",
    "RoadDebris",
    "ConstructionBarrier",
}

# SUMO 车辆类型到设计文档中的事件类型映射。
# 像 barrier_*、road_debris_* 这类对象会在 gt_collector 中按前缀识别。
VTYPE_TO_EVENT_TYPE = {
    "emergency": "Ambulance",
    "police": "Police",
    "fire_engine": "FireTruck",
    "bus": "PublicBus",
    "school_bus": "SchoolBus",
    "crash_vehicle_a": "Accident",
    "crash_vehicle_b": "Accident",
}

# 前缀型 vtype（barrier 与 road_debris 含 length 后缀，见 generate_*.py）
BARRIER_VTYPE_PREFIXES = ("barrier_",)
DEBRIS_VTYPE_PREFIXES = ("road_debris_", "tree_branch_")

# ──────────────────────────────────────────────────────────────────────
# 4. R_traffic 系数（设计文档 §2.1.2 / §2.1.3 / §2.1.4）
# ──────────────────────────────────────────────────────────────────────
W_DELTA_Q = 1.5
W_THROUGHPUT = 0.5
W_QUEUE_UNSERVED = 1.5
W_OCC_ALL = 2.5
W_GREEN_WASTE = 2.0
W_EMERGENCY = 1.0
W_TRANSIT = 0.5
EMERGENCY_WAIT_NORM = 30.0
TRANSIT_WAIT_NORM = 60.0
R_TRAFFIC_DIVISOR = 2.0

# Throughput^sel 分母系数：0.5 vehicles/sec/lane（设计文档 §2.1.2） 1800 veh/h/lane 是常见的饱和流率，考虑到我们场景中车道数较多且通常不完全饱和，设置一个较保守的 0.5 veh/s/lane（1800 veh/h/lane 的一半）作为基准。实际计算时会乘以选定的检测器数量和持续时间，以得到合理的 throughput^sel 范围。
THROUGHPUT_VEHICLES_PER_SEC_PER_LANE =0.5

# ──────────────────────────────────────────────────────────────────────
# 5. R_vision 子项权重（设计文档 §2.2.3）
# ──────────────────────────────────────────────────────────────────────
W_LANE_EVENT_SAMPLE = (0.4, 0.6)    # (w_lane, w_event) for event sample
W_LANE_NORMAL_SAMPLE = (0.7, 0.3)   # (w_lane, w_event) for non-event sample

# ──────────────────────────────────────────────────────────────────────
# 6. R_lane soft 评分（设计文档 §2.2.1）
# ──────────────────────────────────────────────────────────────────────
# err → score 映射表；键为绝对误差整数
LANE_SCORE_BY_ERR = {0: 1.0, 1: 0.7, 2: 0.0, 3: -0.5, 4: -1.0}
LANE_MISSING_SCORE = -1.0       # 漏报受控车道
LANE_FAKE_SCORE = -0.5          # 伪造（写入不受控车道）

# ──────────────────────────────────────────────────────────────────────
# 7. R_event_recog（设计文档 §2.2.2）
# ──────────────────────────────────────────────────────────────────────
S_TYPE_WEIGHT = 0.3             # event_type 匹配得分权重
S_LANE_WEIGHT = 0.3             # located_lane 匹配得分权重（新增）
S_PHASE_WEIGHT = 0.4            # affected_phase 匹配得分权重（仍为三者中最高）
NO_EVENT_HALLUCINATION_PENALTY = 0.3 # 没有事件时的虚假事件惩罚（设计文档 §2.2.2 最后一段）
HALLUCINATION_PENALTY = 0.2 # 每个虚假事件的惩罚（设计文档 §2.2.2 最后一段）
HALLUCINATION_CAP = 3 # 虚假事件惩罚上限（设计文档 §2.2.2 最后一段）

# ──────────────────────────────────────────────────────────────────────
# 8. R_GRPO 合成（设计文档 §3）
# ──────────────────────────────────────────────────────────────────────
W_TRAFFIC = 0.65
W_VISION = 0.35
R_GRPO_SCALE = 3.0
R_GRPO_FALLBACK = -3.0
