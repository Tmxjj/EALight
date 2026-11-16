
# JiNan 路口 ID 列表（4×3=12 个路口）
_JINAN_JUNCTIONS  = [f"intersection_{i}_{j}" for i in range(1, 5) for j in range(1, 4)]
# Hangzhou 路口 ID 列表（4×4=16 个路口）
_HANGZHOU_JUNCTIONS = [f"intersection_{i}_{j}" for i in range(1, 5) for j in range(1, 5)]
# NewYork 路口 ID 列表（7×28=196 个路口）
_NEWYORK_JUNCTIONS = [f"intersection_{i}_{j}" for i in range(1, 8) for j in range(1, 29)]
# Syn-Train 路口 ID 列表（4×4=16 个路口）
_SYNTRAIN_JUNCTIONS = [f"intersection_{i}_{j}" for i in range(1, 5) for j in range(1, 5)]
# 通用进口道 / 上游摄像头默认参数，被 _tls_sensor_cfg() 批量复用于各路口。
# 各字段含义见文件头 Description。
_APPROACH_CAM_CFG = {
    # 传感器渲染模式：junction_front_all 包含车辆、地面、地图、天空全部图层
    "sensor_types": ["junction_front_all"],
    # 相机安装高度（米）；进口道停止线 / 上游端点均使用 15m
    "tls_camera_height": 15,
    # 俯仰角（°）：None → 原始模式（lookAt z = height//3）；
    # 指定值（建议 30°–60°）→ 以该角度向地面（z=0）看，水平投影距离 = height/tan(pitch_deg)
    "camera_pitch_deg": 46.5,  # 
    # 相机沿来车方向的后退距离（米）：0 = 位于 element_position 正上方，不后退；
    # 正值 = 向来车方向偏移，扩大近端视野（停止线摄像头建议 5–15m）
    "camera_setback": 10,
    # 沿 stop_line 平行方向的横向偏移（米）：
    # 正值 = 向中间分隔带（内侧）平移，负值 = 向路肩（外侧）平移，0 = 不偏移
    "camera_lateral": 0,
}

def _generate_grid_topology(m: int, n: int) -> dict:
    """
    针对 m*n 规则路网（横向东西方向 m 个，纵向南北方向 n 个），构建合法拓扑字典。
    只在这个内部相互传播，排除掉外围的虚拟交叉口。
    """
    topology = {}
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            jid = f"intersection_{i}_{j}"
            neighbors = []
            if i < m: neighbors.append(f"intersection_{i+1}_{j}")
            if i > 1: neighbors.append(f"intersection_{i-1}_{j}")
            if j < n: neighbors.append(f"intersection_{i}_{j+1}")
            if j > 1: neighbors.append(f"intersection_{i}_{j-1}")
            topology[jid] = neighbors
    return topology

def _tls_sensor_cfg(junction_ids):
    """为路口 ID 列表批量生成 tls/upstream sensor_cfg 字典。"""
    if isinstance(junction_ids, str):
        junction_ids = [junction_ids]
    return {jid: _APPROACH_CAM_CFG for jid in junction_ids}


SCENARIO_CONFIGS = {
    "Hongkong_YMT": {
        "SCENARIO_NAME": "Hongkong_YMT",
        "NETFILE": "ymt_eval",
        "JUNCTION_NAME": "J1",
        "PHASE_NUMBER": 3,
        "APPROACH_DIRS": ["N", "E", "S", "W"],
        "CENTER_COORDINATES": (172, 201, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:0, 3:1},
        "RENDERER_CFG": {
            "preset": "SQUARE_720",
            "resolution": 0.8,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False
        },
        "SENSOR_CFG": {
            "tls": _tls_sensor_cfg("J1"),         # 进口道摄像头（停止线处）
            # "upstream": _tls_sensor_cfg("J1"),     # 上游摄像头（lane 起点处）
            # "aircraft": {"sensor_types": ["aircraft_all"], "height": 65.0}
        },
        "TOPOLOGY": {}  # 单交叉口场景不考虑事件广播
    },

    "SouthKorea_Songdo": {
        "SCENARIO_NAME": "SouthKorea_Songdo",
        "NETFILE": "songdo",
        "JUNCTION_NAME": "J2",
        "PHASE_NUMBER": 4,
        "APPROACH_DIRS": ["N", "E", "S", "W"],
        "CENTER_COORDINATES": (900, 1641, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:1, 3:0},
        "RENDERER_CFG": {
            "preset": "SQUARE_720",
            "resolution": 0.85,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False
        },
        "SENSOR_CFG": {
            "tls": _tls_sensor_cfg("J2"),
            # "upstream": _tls_sensor_cfg("J2"),
            # "aircraft": {
            #     "sensor_types": ["aircraft_all"],
            #     "height": 85.0,
            #     # Songdo 路口道路走向为 NE/NW/SE/SW（约 45° 角），
            #     # [-1, 1, 0] → 相机顺时针旋转 45°，使 NE 进口道显示在图像顶部（北方）
            #     "camera_heading": [-1, 1, 0]
            # }
        },
        "TOPOLOGY": {}  # 单交叉口场景不考虑事件广播
    },

    "France_Massy": {
        "SCENARIO_NAME": "France_Massy",
        "NETFILE": "massy",
        "JUNCTION_NAME": "INT1",
        "PHASE_NUMBER": 2,
        # T字路口：仅 North / South / West 三个进口道，无 East 进口道
        "APPROACH_DIRS": ["N", "S", "W"],
        "CENTER_COORDINATES": (173, 244, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:1, 2:0},
        "RENDERER_CFG": {
            "preset": "SQUARE_720",
            "resolution": 0.75,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False,
        },
        "SENSOR_CFG": {
            "tls": _tls_sensor_cfg("INT1"),
            # "upstream": _tls_sensor_cfg("INT1"),
            # "aircraft": {"sensor_types": ["aircraft_all"], "height": 55.0}
        },
        "TOPOLOGY": {}  # 单交叉口场景不考虑事件广播
    },

    "JiNan": {
        "SCENARIO_NAME": "JiNan",
        "NETFILE": "jinan",
        "JUNCTION_NAME": _JINAN_JUNCTIONS,
        "PHASE_NUMBER": 4,
        "APPROACH_DIRS": ["N", "E", "S", "W"],
        "CENTER_COORDINATES": (173, 244, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:1, 3:0},
        "RENDERER_CFG": {
            "preset": "SQUARE_720",
            "resolution": 0.8,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False
        },
        "SENSOR_CFG": {
            "tls": _tls_sensor_cfg(_JINAN_JUNCTIONS),
            # "upstream": _tls_sensor_cfg(_JINAN_JUNCTIONS),
            # "aircraft": {"sensor_types": ["aircraft_all"], "height": 60.0}
        },
        "TOPOLOGY": _generate_grid_topology(m=4, n=3)  # JiNan 是 4x3 路网，排除虚拟路口
    },

    "Syn-Train": {
        "SCENARIO_NAME": "Syn-Train",
        "NETFILE": "roadnet_4_4",
        "JUNCTION_NAME": _SYNTRAIN_JUNCTIONS,
        "PHASE_NUMBER": 4,
        "APPROACH_DIRS": ["N", "E", "S", "W"],
        "CENTER_COORDINATES": (173, 244, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:1, 3:0},
        "RENDERER_CFG": {
            "preset": "SQUARE_720",
            "resolution": 0.8,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False
        },
        "SENSOR_CFG": {
            "tls": _tls_sensor_cfg(_HANGZHOU_JUNCTIONS),
            # "upstream": _tls_sensor_cfg(_HANGZHOU_JUNCTIONS),     
        },
        "TOPOLOGY": _generate_grid_topology(m=4, n=4)  # Syn-Train 是 4x4 路网网络
    },

    "NewYork": {
        "SCENARIO_NAME": "NewYork",
        "NETFILE": "NewYork",
        "JUNCTION_NAME": _NEWYORK_JUNCTIONS,
        "PHASE_NUMBER": 4,
        "APPROACH_DIRS": ["N", "E", "S", "W"],
        "CENTER_COORDINATES": (172, 201, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:0, 3:1},
        "RENDERER_CFG": {
            "preset": "SQUARE_720",
            "resolution": 0.8,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False
        },
        "SENSOR_CFG": {
            "tls": _tls_sensor_cfg(_NEWYORK_JUNCTIONS),
            # "upstream": _tls_sensor_cfg(_NEWYORK_JUNCTIONS),
            # "aircraft": {"sensor_types": ["aircraft_all"], "height": 60.0}
        },
        "TOPOLOGY": _generate_grid_topology(m=7, n=28)  # NewYork 是 7*28 路网网络
    },

    "Hangzhou": {
        "SCENARIO_NAME": "Hangzhou",
        "NETFILE": "Hangzhou",
        "JUNCTION_NAME": _HANGZHOU_JUNCTIONS,
        "PHASE_NUMBER": 4,
        "APPROACH_DIRS": ["N", "E", "S", "W"],
        "CENTER_COORDINATES": (172, 201, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:0, 3:1},
        "RENDERER_CFG": {
            "preset": "SQUARE_720",
            "resolution": 0.8,
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False
        },
        "SENSOR_CFG": {
            "tls": _tls_sensor_cfg(_HANGZHOU_JUNCTIONS),
            # "upstream": _tls_sensor_cfg(_HANGZHOU_JUNCTIONS),
            # "aircraft": {"sensor_types": ["aircraft_all"], "height": 60.0}
        },
        "TOPOLOGY": _generate_grid_topology(m=4, n=4)   # Hangzhou 是 4x4 路网网络
    },

    # ── 论文展示专用：单路口标准四叉路口，含四类事件同时展示 ───────────────────────────
    "paper_Display": {
        "SCENARIO_NAME": "paper_Display",
        "NETFILE": "hangzhou_display",
        "JUNCTION_NAME": "J_center",
        "PHASE_NUMBER": 4,
        "APPROACH_DIRS": ["N", "E", "S", "W"],
        # J_center 在 SUMO 坐标系中约为 (250, 250)（节点偏移后）
        "CENTER_COORDINATES": (250, 250, 100),
        "SENSOR_INDEX_2_PHASE_INDEX": {0:2, 1:3, 2:0, 3:1},
        "RENDERER_CFG": {
            "preset": "SQUARE_720",          # 高清 1920×1080，适合论文图表
            "resolution": 0.63,         # FOV 略微收窄以聚焦车道细节
            "vehicle_model": "high",
            "render_mode": "offscreen",
            "should_count_vehicles": True,
            "debuger_print_node": False,
            "debuger_spin_camera": False,
            "is_render": True,
            "is_every_frame": False
        },
        "SENSOR_CFG": {
            "tls": {
                "J_center": {
                    "sensor_types":    ["junction_front_all"],
                    "tls_camera_height": 15,
                    "camera_pitch_deg":  40,  # None → 使用原始 lookAt（height//3）
                    "camera_setback":    5,     # 位于停止线位置，不后退
                    "camera_lateral":    5.5,     # 向中间分隔带平移
                }
            }
        },
        "TOPOLOGY": {}  # 单交叉口，无路口间广播
    },
}
