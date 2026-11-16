import numpy as np
import gymnasium as gym
from gymnasium.core import Env
from collections import deque
import copy
from typing import Any, Optional, SupportsFloat, Tuple, Dict, List, Union

from src.training.rlvr.reward.constants import EMERGENCY_VTYPES, TRANSIT_VTYPES
from src.training.rlvr.reward.traffic_reward import compute_r_traffic

# 与 choose_next_phase_with_duration 一致的绿灯候选集（VLM 可选时长，单位：秒）
GREEN_DURATION_CANDIDATES = [15, 20, 25, 30, 35, 40]

# 固定配时/MaxPressure 基线的绿灯时长
FIXED_TIME_GREEN_DURATION = 25 # 调大fixed-tiem 降低效果

class OccupancyList:
    """缓存一个决策窗口内的 occupancy，并在窗口结束时求平均。

    这里的“决策窗口”不是单个 SUMO step，而是从一次控制动作下发后，
    SUMO 连续推进到下一个 can_perform_action=True 的整段时间。

    例如 choose_next_phase_with_duration 模式下，如果本次动作选择
    27s 绿灯 + 3s 黄灯，那么这 30s 内每个仿真步采集到的 occupancy
    会先暂存在 self.elements 中；到下一个可决策时刻，再对这 30s 的
    occupancy 求平均，形成历史状态窗口 self.states 中的一帧。

    数据形态：
      - 单路口：self.elements 是 List[List[float]]
        例如 [[occ_m0, occ_m1, ...], [occ_m0, occ_m1, ...], ...]
      - 多路口：self.elements 是 List[Dict[str, List[float]]]
        例如 [{'J1': [...], 'J2': [...]}, {'J1': [...], 'J2': [...]}, ...]
    """
    def __init__(self) -> None:
        # 临时缓存一个决策窗口内每个仿真 step 的 occupancy。
        # 每次 calculate_average() 后会被清空，避免跨决策窗口混合。
        self.elements = []

    def add_element(self, element) -> None:
        # 支持 dict (多路口) 或 list (单路口)
        if isinstance(element, list) or isinstance(element, dict):
            # 单路口 element: List[float]，长度为该路口真实 movement 数。
            # 多路口 element: Dict[tls_id, List[float]]，每个路口可有不同 movement 数。
            # 这里不逐项校验 float 类型，默认上游 state_wrapper 已保证结构正确。
            self.elements.append(element)
        else:
            raise TypeError("添加的元素必须是列表或字典类型")

    def clear_elements(self) -> None:
        # 清空当前决策窗口缓存；reset 时也会调用，避免上一个 episode 残留数据。
        self.elements = []

    def calculate_average(self) -> Any:
        """计算一段时间的平均 occupancy
        如果 elements 是 list of lists (单路口): 返回一个平均后的 list
        如果 elements 是 list of dicts (多路口): 返回一个 dict，key 为 tls_id, value 为该路口的平均 occupancy list
        """
        if not self.elements:
            return None
            
        first_elem = self.elements[0]
        
        # Case 1: Single Junction (List of Lists)
        if isinstance(first_elem, list):
            # arr shape: (窗口内仿真步数, movement 数)
            # axis=0 表示对时间维求平均，得到每个 movement 在本决策窗口内的平均占有率。
            arr = np.array(self.elements)
            # SUMO occupancy 原始单位是百分比 [0, 100]，这里除以 100 转成 [0, 1]。
            averages = np.mean(arr, axis=0, dtype=np.float32)/100
            self.clear_elements()
            return averages.tolist()
            
        # Case 2: Multi Junction (List of Dicts)
        elif isinstance(first_elem, dict):
            # self.elements = [ {'J1': [0,0,..], 'J2': [..]}, {'J1': [..], ...} ]
            # 需要转置为: {'J1': [[0,0,..], [..]], 'J2': ...}
            tls_ids = first_elem.keys()
            result = {}
            for tid in tls_ids:
                # 提取该路口的所有 step 数据
                # tls_data shape: (窗口内仿真步数, 当前路口 movement 数)
                tls_data = [step_data[tid] for step_data in self.elements]
                arr = np.array(tls_data)
                # 每个路口独立求平均，因为不同路口 movement 数可能不同。
                avg = np.mean(arr, axis=0, dtype=np.float32)/100
                result[tid] = avg.tolist()
            
            self.clear_elements()
            return result
        
        return None


class TSCEnvWrapper(gym.Wrapper):
    """TSC Env Wrapper for single junction or multiple junctions.

    这个 wrapper 把底层 TSHub/SUMO 返回的完整状态，压缩成交通信号控制
    常用的 observation：

        最近 max_states 个“决策窗口”的 movement occupancy 历史。

    注意：
      - max_states 不是 SUMO 仿真步数，而是历史决策窗口数量。
      - 一个决策窗口通常覆盖多个 SUMO step，例如 27s 绿灯 + 3s 黄灯。
      - 单路口 obs shape: (max_states, 当前路口 movement 数)。
      - 多路口 obs: List[Dict[tls_id, List[float]]]，保留原始结构。
    """
    def __init__(self, env: Env, tls_id: Union[str, List[str]], number_phases:int, max_states:int=5) -> None:
        super().__init__(env)
        # 单路口时 tls_id 是 str，例如 "J1"；多路口时是 List[str]。
        self.tls_id = tls_id
        self.is_multi_agent = isinstance(tls_id, list)

        # 当前场景 prompt/action space 中允许选择的绿灯相位数量。
        # 例如 Hongkong_YMT 是 3，France_Massy 是 2，Jinan/Hangzhou/NewYork 是 4。
        self.number_phases = number_phases

        # 历史状态窗口长度：保留最近 max_states 个决策窗口的平均 occupancy。
        # 默认 5 表示模型/控制器能看到最近 5 次决策周期的交通趋势。
        self.max_states = max_states

        # 单路口静态结构信息：
        # movement_ids 的顺序决定 obs 每一列对应哪个 movement。
        # phase2movements 用于把 movement occupancy 汇总成每个 phase 的 occupancy。
        self.movement_ids = None
        self.phase2movements = None

        # 多路口静态结构信息：
        # 每个路口独立保存 movement_ids/phase2movements，因为异构路口 movement 数不同。
        self.multi_movement_ids = {}
        self.multi_phase2movements = {}
        
        # self.states 是最终 observation 的历史缓存。
        # reset() 前还不知道真实 movement 数，因此这里先用 12 作为兼容 fallback；
        # reset() 拿到底层 state 后，会通过 _reset_state_history() 按真实 movement 数重建。
        #
        # 单路口：deque[List[float]]，最终 get_state() 转成 np.ndarray。
        # 多路口：deque[Dict[str, List[float]]]，get_state() 返回 list(self.states)。
        self.states = deque(
            [copy.deepcopy(self._get_initial_state()) for _ in range(max_states)],
            maxlen=max_states,
        )

        # self.occupancy 不是最终 obs，而是“当前决策窗口内”的临时缓存。
        # step() 内每推进一个 SUMO step，就把当步 occupancy 放进去；
        # 窗口结束时求平均，再 append 到 self.states。
        self.occupancy = OccupancyList()

        # 保存上一轮决策结束时的 TLS 快照，供下一次 step() 计算 reward。
        # 这样主环境 step 和 rollout 都可以共享同一套 detector-only reward 计算器。
        self._last_tls_snapshot: Union[dict, Dict[str, dict], None] = None

    @staticmethod
    def _copy_tls_rollout_snapshot(tls_info: dict, vehicle_wait_times: Optional[List[dict]] = None) -> dict:
        """复制 reward 计算所需的最小 TLS 截面信息。

        这里显式做深拷贝，避免后续仿真 step 继续推进时，
        render_json / states 内部引用被底层环境复用，导致 rollout trace 被污染。

        返回字段均与 reward 定义直接相关：
          - movement_ids / phase2movements：用于确定 D 与 D_sel
          - jam_length_vehicle：对应公式中的 jamVeh
          - last_step_occupancy：对应公式中的 occ
                    - last_step_vehicle_id_list：对应 Throughput 的集合差近似
                    - vehicle_wait_times：当前仿真步的事件车辆等待时间快照
          - can_perform_action / this_phase_index：仅用于调试与一致性核验
        """
        return {
            "movement_ids": list(tls_info.get("movement_ids", []) or []),
            "phase2movements": copy.deepcopy(tls_info.get("phase2movements", {}) or {}),
                        "last_step_vehicle_id_list": copy.deepcopy(tls_info.get("last_step_vehicle_id_list", []) or []),
            "jam_length_vehicle": list(tls_info.get("jam_length_vehicle", []) or []),
            "last_step_occupancy": list(tls_info.get("last_step_occupancy", []) or []),
                        "vehicle_wait_times": list(vehicle_wait_times or []),
            "can_perform_action": bool(tls_info.get("can_perform_action", False)),
            "this_phase_index": tls_info.get("this_phase_index"),
        }

    def _capture_current_tls_snapshot(self, state: dict) -> Union[dict, Dict[str, dict]]:
        """从底层完整状态中提取当前时刻的 TLS 快照。

        这里只保留 reward 计算需要的字段，避免把完整 states 长期缓存到 wrapper 中。
        """
        tls_states = state.get("tls", {}) if state else {}
        if self.is_multi_agent:
            return {
                tid: self._copy_tls_rollout_snapshot(tls_states.get(tid, {}))
                for tid in self.tls_id
            }
        return self._copy_tls_rollout_snapshot(tls_states.get(self.tls_id, {}))

    @staticmethod
    def _get_phase_detector_indices(
        movement_ids: List[str],
        phase2movements: dict,
        selected_phase: int,
    ) -> List[int]:
        """返回候选相位 selected_phase 对应的 detector 下标集合。

        约定：
          - D 使用当前路口全部受信号控制的 movement/detector，即 movement_ids；
          - D_sel 由 phase2movements[selected_phase] 给出的 movement 集合确定；
          - 若 phase2movements 的 key 在不同场景中混用 int / str，这里统一兼容。
        """
        selected_movements = (
            phase2movements.get(selected_phase)
            or phase2movements.get(str(selected_phase))
            or []
        )
        selected_movement_set = set(selected_movements)
        return [idx for idx, mid in enumerate(movement_ids) if mid in selected_movement_set]

    @staticmethod
    def _sum_selected_queue(jam_length_vehicle: List[float], selected_indices: List[int]) -> float:
        """汇总当前时刻被选相位 detector 的排队车辆数 Q_t^sel。"""
        total = 0.0
        for idx in selected_indices:
            if idx < len(jam_length_vehicle):
                total += float(jam_length_vehicle[idx])
        return total

    @staticmethod
    def _mean_all_occupancy(last_step_occupancy: List[float]) -> float:
        """计算当前路口全部受控 detector 的平均占有率，范围归一化到 [0, 1]。"""
        if not last_step_occupancy:
            return 0.0
        return float(sum(float(v) for v in last_step_occupancy) / len(last_step_occupancy) / 100.0)

    @staticmethod
    def _mean_other_occupancy(last_step_occupancy: List[float], selected_indices: List[int]) -> float:
        """计算其它 detector 的平均占有率 O_t^other，范围归一化到 [0, 1]。"""
        selected_index_set = set(selected_indices)
        other_values = [
            float(occ) / 100.0
            for idx, occ in enumerate(last_step_occupancy)
            if idx not in selected_index_set
        ]
        if not other_values:
            return 0.0
        return float(sum(other_values) / len(other_values))

    def _collect_vehicle_wait_times(self) -> List[dict]:
        """收集当前仿真步里与事件 reward 相关的车辆等待时间。

        这里先按“活跃车辆 + vType 过滤”采集，保持训练链路可运行。
        后续如果 lane_mapping 完整，可继续把范围收紧到当前路口受控 lane。
        """
        try:
            sumo_conn = self.env.unwrapped.tsc_env.tshub_env.sumo
            vehicle_ids = list(sumo_conn.vehicle.getIDList())
        except Exception:
            return []

        wait_times: List[dict] = []
        for vehicle_id in vehicle_ids:
            try:
                vtype = sumo_conn.vehicle.getTypeID(vehicle_id)
            except Exception:
                continue
            if vtype not in EMERGENCY_VTYPES and vtype not in TRANSIT_VTYPES:
                continue

            try:
                waiting_time = float(sumo_conn.vehicle.getWaitingTime(vehicle_id))
            except Exception:
                waiting_time = 0.0

            try:
                road_id = sumo_conn.vehicle.getRoadID(vehicle_id)
            except Exception:
                road_id = None

            wait_times.append(
                {
                    "vid": vehicle_id,
                    "vtype": vtype,
                    "waiting_time": waiting_time,
                    "road_id": road_id,
                }
            )
        return wait_times

    def compute_two_round_rollout_reward(
        self,
        *,
        initial_tls_snapshot: dict,
        first_round_trace: List[dict],
        second_round_trace: List[dict],
        selected_phase: int,
        candidate_duration: int,
    ) -> dict:
        """兼容旧调用点的薄壳封装。

        现在真正的 reward 逻辑已经下沉到 src.training.rlvr.reward.traffic_reward。
        这个方法只负责把返回值整理成历史接口期望的 {'reward', 'components'} 结构，
        避免 golden_generation / env.step / 旧测试一次性全部改动。
        """
        result = compute_r_traffic(
            initial_snapshot=initial_tls_snapshot,
            first_round_trace=first_round_trace,
            second_round_trace=second_round_trace,
            selected_phase=selected_phase,
            candidate_duration=candidate_duration,
        )
        return {
            "reward": result["r_traffic"],
            "components": result,
        }
    
    def _get_initial_state(self) -> Union[List[int], Dict[str, List[int]]]:
        """生成 reset 前的占位历史状态。

        初始化 wrapper 时尚未调用底层 env.reset()，因此还不知道真实 movement 数。
        为兼容原始 12 movement 场景，这里 fallback 到 12；真正运行时会在 reset()
        中根据底层返回的真实 occupancy 长度重建 self.states。
        """
        if self.is_multi_agent:
            return {
                tid: [0] * len(self.multi_movement_ids.get(tid, []) or range(12))
                for tid in self.tls_id
            }
        else:
            return [0] * len(self.movement_ids or range(12))

    def _reset_state_history(self, occupancy) -> None:
        """按当前场景真实 movement 数重建历史状态窗口。"""
        self.states = deque(maxlen=self.max_states)

        if self.is_multi_agent:
            # 多路口时 occupancy 是 {tls_id: [occ_m0, occ_m1, ...]}。
            # 每个 tls 的列表长度来自该路口真实 movement 数，允许路口之间长度不同。
            zero_state = {
                tid: [0.0] * len(occupancy.get(tid, []))
                for tid in self.tls_id
            }
            for _ in range(self.max_states - 1):
                self.states.append(copy.deepcopy(zero_state))
            self.states.append(copy.deepcopy(occupancy))
        else:
            # 单路口时 occupancy 是 List[float]。
            # 用 max_states-1 帧全零 + 当前 occupancy 作为 episode 初始历史窗口。
            zero_state = [0.0] * len(occupancy)
            for _ in range(self.max_states - 1):
                self.states.append(zero_state.copy())
            self.states.append(list(occupancy))
    
    def get_state(self):
        """返回对外 observation。

        单路口返回 np.ndarray，shape=(max_states, movement_count)。
        多路口因为不同路口 movement_count 可能不同，不能安全拼成规则 ndarray，
        因此保留 list of dicts 的结构。
        """
        # 如果是多智能体，返回 list of dicts (时间序列)
        # 很多 RL 库可能需要把这个展平，或者特定格式。这里维持原始结构。
        if self.is_multi_agent:
            return list(self.states)
        return np.array(self.states, dtype=np.float32)
    
    @property
    def action_space(self):
        # 联合动作空间：phase_id × duration_idx
        # phase_id    ∈ [0, number_phases)
        # duration_idx ∈ [0, len(GREEN_DURATION_CANDIDATES))，映射到实际秒数
        single_space = gym.spaces.Dict({
            'phase_id':    gym.spaces.Discrete(self.number_phases),
            'duration_idx': gym.spaces.Discrete(len(GREEN_DURATION_CANDIDATES))
        })
        if self.is_multi_agent:
            return gym.spaces.Dict({tid: single_space for tid in self.tls_id})
        return single_space
    
    @property
    def observation_space(self):
        """声明 observation 的取值范围和 shape。

        reset 前 movement 数未知时 fallback 到 12；reset 后如果外部再次访问
        observation_space，会使用真实 movement 数。
        """
        if self.is_multi_agent:
             return gym.spaces.Dict({
                tid: gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_states, len(self.multi_movement_ids.get(tid, []) or range(12))),
                    dtype=np.float32,
                )
                for tid in self.tls_id
            })

        state_dim = len(self.movement_ids or range(12))
        
        obs_space = gym.spaces.Box(
            low=np.zeros((self.max_states, state_dim), dtype=np.float32),
            high=np.ones((self.max_states, state_dim), dtype=np.float32),
            shape=(self.max_states, state_dim),
            dtype=np.float32,
        ) 
        return obs_space
    
    # Wrapper
    def state_wrapper(self, state):
        """返回当前每个 movement 的 occupancy。

        Args:
            state: 底层环境返回的完整仿真状态，不是最终对外 observation。
                   这里会从 state['tls'][tls_id] 中抽取 last_step_occupancy。

        Returns:
            occupancy:
              - 单路口：List[float]，长度为该路口真实 movement 数。
              - 多路口：Dict[tls_id, List[float]]。
            can_perform_action:
              当前是否到达信号灯可决策时刻。step() 会一直推进底层仿真，
              直到该值为 True。

        多路口异步模式：任意一个路口 can_perform_action=True 即返回 True，
        由上层（run_eval.py）根据各路口的 can_perform_action 字段决定哪些路口需要决策。
        """
        if self.is_multi_agent:
            occupancy = {}
            # 异步决策：任一路口可执行动作即退出仿真内循环，交由上层按路口过滤
            can_perform_action = False
            for tid in self.tls_id:
                tls_state = state['tls'][tid]
                occupancy[tid] = tls_state['last_step_occupancy']
                if tls_state['can_perform_action']:
                    can_perform_action = True
            return occupancy, can_perform_action
        else:
            occupancy = state['tls'][self.tls_id]['last_step_occupancy']
            can_perform_action = state['tls'][self.tls_id]['can_perform_action']
            return occupancy, can_perform_action
    
    def info_wrapper(self, infos, occupancy):
        """在 info 中加入每个 phase 的占有率。

        occupancy 是当前决策窗口的平均 movement occupancy，不是瞬时 occupancy。
        phase_occ 的计算方式是把同一个 phase 控制的所有 movement occupancy 相加。
        """
        if self.is_multi_agent:
            phase_occ_all = {}
            for tid in self.tls_id:
                tid_occ = occupancy[tid]
                tid_movements = self.multi_movement_ids[tid]
                tid_p2m = self.multi_phase2movements[tid]
                
                movement_occ = {key: value for key, value in zip(tid_movements, tid_occ)}
                phase_occ = {}
                for phase_index, phase_movements in tid_p2m.items():
                    phase_occ[phase_index] = sum([movement_occ[phase] for phase in phase_movements])
                phase_occ_all[tid] = phase_occ
            
            infos['phase_occ'] = phase_occ_all
        else:
            movement_occ = {key: value for key, value in zip(self.movement_ids, occupancy)}
            phase_occ = {}
            for phase_index, phase_movements in self.phase2movements.items():
                phase_occ[phase_index] = sum([movement_occ[phase] for phase in phase_movements])
            
            infos['phase_occ'] = phase_occ
        return infos

    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化 (1) 静态信息; (2) 动态信息
        """
        # 清理上一个 episode/场景遗留的窗口内 occupancy 缓存。
        self.occupancy.clear_elements()

        # 底层完整仿真状态，包含 tls、vehicle、detector 等信息。
        # 后续会从中抽取 movement_ids、phase2movements 和 last_step_occupancy。
        state = self.env.reset()

        if self.is_multi_agent:
            for tid in self.tls_id:
                # 静态结构信息只需 reset 时读取一次，用于解释 obs 维度和 phase 聚合。
                self.multi_movement_ids[tid]   = state['tls'][tid]['movement_ids']
                self.multi_phase2movements[tid] = state['tls'][tid]['phase2movements']
        else:
            # movement_ids 的顺序和 last_step_occupancy 的顺序一一对应。
            self.movement_ids   = state['tls'][self.tls_id]['movement_ids']
            self.phase2movements = state['tls'][self.tls_id]['phase2movements']

        # 处理路口动态信息
        # occupancy 是 reset 时刻的瞬时 movement occupancy。
        occupancy, _ = self.state_wrapper(state=state)

        # 保存 reset 时刻的 TLS 快照，作为下一次 step 的 t0 基准。
        self._last_tls_snapshot = self._capture_current_tls_snapshot(state)

        # 按真实 movement 数重建 5 帧历史：前 max_states-1 帧为 0，最后一帧为 reset occupancy。
        self._reset_state_history(occupancy)

        # 覆盖变量名：这里的 state 已经不再是底层完整仿真状态，
        # 而是 wrapper 对外返回的 observation。
        state = self.get_state()
        return state, {'step_time': 0}

    @staticmethod
    def _decode_action(action_input) -> tuple:
        """将上层传入的动作解码为 (phase_id, green_duration) 元组。

        支持四种输入格式（兼容新旧代码）：
          1. Dict {'phase_id': int, 'duration': int}     → 推荐格式，直接使用实际绿灯秒数
          2. Dict {'phase_id': int, 'duration_idx': int} → 索引格式，转换后使用（向后兼容）
          3. Tuple/List (phase_id, green_duration)        → 直接使用
          4. int                                          → 旧格式，duration 取 FIXED_TIME_GREEN_DURATION
        """
        if isinstance(action_input, dict):
            phase_id = int(action_input['phase_id'])
            if 'duration' in action_input:
                # 推荐格式：直接传入实际绿灯秒数，无需索引转换
                duration = int(action_input['duration'])
            elif 'duration_idx' in action_input:
                # 向后兼容：索引 → 实际秒数
                duration = GREEN_DURATION_CANDIDATES[int(action_input['duration_idx'])]
            else:
                duration = FIXED_TIME_GREEN_DURATION
        elif isinstance(action_input, (tuple, list)) and len(action_input) == 2:
            phase_id, duration = int(action_input[0]), int(action_input[1])
        else:
            phase_id = int(action_input)
            duration = FIXED_TIME_GREEN_DURATION  # 旧格式兼容：27s+3s黄灯=30s整步
        return phase_id, duration

    def step(
        self,
        action: Union[int, Dict, Tuple],
        return_rollout_trace: bool = False,
        trace_tls_ids: Optional[List[str]] = None,
        compute_reward: bool = True,
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """执行一次控制决策，并返回下一个可决策时刻的 observation。

        一次 wrapper.step(action) 内部可能调用多次底层 env.step()：
        只有当 can_perform_action=True 时，才认为本次决策窗口结束。

        Args:
            action:
                上层给定的控制动作。
            return_rollout_trace:
                为 True 时，额外记录该决策窗口内每个 simulation step 的 TLS 截面，
                供 rollout reward 在窗口外做两轮聚合计算。
            trace_tls_ids:
                需要记录 trace 的路口 ID 列表；只记录 reward 相关最小字段，
                避免把完整 states/render_json 全量缓存进内存。
        """
        can_perform_action = False
        initial_tls_snapshot = copy.deepcopy(self._last_tls_snapshot)
        if trace_tls_ids is None:
            if self.is_multi_agent:
                trace_tls_ids = list(self.tls_id)
            else:
                trace_tls_ids = [self.tls_id]
        rollout_trace: Dict[str, List[dict]] = {tid: [] for tid in trace_tls_ids}

        # 将上层动作统一解码为底层可用格式
        # 多路口：{jid: action_input} → {jid: (phase_id, duration)}
        # 单路口：action_input → (phase_id, duration)
        if self.is_multi_agent:
            decoded_action = {tid: self._decode_action(action[tid]) for tid in self.tls_id}
        else:
            decoded_action = self._decode_action(action)

        # NOTE: can_perform_action 当前仿真时间 (sim_step) 等于 预定的下一次动作时间 (sim_step+delta_time) 时，该标志位变为 True。
        while not can_perform_action:
            if self.is_multi_agent:
                step_action = decoded_action
            else:
                step_action = {self.tls_id: decoded_action}

            states, _, truncated, dones, infos, sensor_data = super().step(step_action)

            # states 是底层完整仿真状态；occupancy 是从 states 中抽取出的 movement 占有率。
            occupancy, can_perform_action = self.state_wrapper(state=states)

            tls_states = states.get('tls', {})
            for tid in rollout_trace:
                rollout_trace[tid].append(
                    self._copy_tls_rollout_snapshot(
                        tls_states.get(tid, {}),
                        vehicle_wait_times=self._collect_vehicle_wait_times(),
                    )
                )

            # 暂存当前仿真 step 的 occupancy，等待决策窗口结束后求平均。
            self.occupancy.add_element(occupancy)

        # 循环结束 → can_perform_action=True 截面时刻，读取一次 e2 快照计算 reward
        # render_json 保留完整仿真状态，用于保存 data.json 或给渲染/数据生成模块使用；
        # 它不同于下面返回的 state/obs。
        render_json = states.copy()

        # 保存本轮窗口结束时刻的 TLS 快照，作为下一次 step 的 t0 基准。
        self._last_tls_snapshot = self._capture_current_tls_snapshot(states)

        if compute_reward:
            # reward 统一使用 rlvr.reward.traffic_reward 作为单一来源。
            # 这里的 first_round_trace 就是当前 step 的决策窗口轨迹；
            # rollout 会在外层额外拼接第二轮 FixedTime 轨迹。
            if self.is_multi_agent:
                rewards = {}
                for tid in self.tls_id:
                    phase_id, duration = self._decode_action(action[tid])
                    tid_initial_snapshot = {}
                    if isinstance(initial_tls_snapshot, dict):
                        tid_initial_snapshot = copy.deepcopy(initial_tls_snapshot.get(tid, {}))
                    reward_result = compute_r_traffic(
                        initial_snapshot=tid_initial_snapshot,
                        first_round_trace=rollout_trace.get(tid, []),
                        second_round_trace=[],
                        selected_phase=phase_id,
                        candidate_duration=duration,
                    )
                    rewards[tid] = reward_result["r_traffic"]
            else:
                phase_id, duration = self._decode_action(action)
                reward_result = compute_r_traffic(
                    initial_snapshot=copy.deepcopy(initial_tls_snapshot or {}),
                    first_round_trace=rollout_trace.get(self.tls_id, []),
                    second_round_trace=[],
                    selected_phase=phase_id,
                    candidate_duration=duration,
                )
                rewards = reward_result["r_traffic"]
        else:
            # 主轨迹/非评估路径不需要 reward 计算时，直接返回零值，避免额外开销。
            if self.is_multi_agent:
                rewards = {tid: 0.0 for tid in self.tls_id}
            else:
                rewards = 0.0

        # avg_occupancy 是当前决策窗口内 occupancy 的时间平均值，
        # 这是最终 append 到历史状态 self.states 的一帧。
        avg_occupancy = self.occupancy.calculate_average()

        # phase_occ 也基于 avg_occupancy 计算，表示该决策窗口内每个 phase 的平均压力/占有率。
        infos = self.info_wrapper(infos, occupancy=avg_occupancy)
        infos['3d_data'] = sensor_data
        if return_rollout_trace:
            infos['rollout_trace'] = rollout_trace

        # 历史窗口向前滚动一帧：丢掉最老窗口，加入当前窗口平均 occupancy。
        self.states.append(avg_occupancy)

        # 对外 observation：单路口为 ndarray(max_states, movement_count)。
        state = self.get_state()

        return state, rewards, truncated, dones, infos, render_json
    def close(self) -> None:
        return super().close()
