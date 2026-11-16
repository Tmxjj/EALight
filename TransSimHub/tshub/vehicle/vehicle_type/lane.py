import enum
from .base_vehicle_action import VehicleAction

class LaneActionType(enum.Enum):
    keep_lane = 0
    slow_down = 1
    change_lane_left = 2
    change_lane_right = 3

class LaneAction(VehicleAction):
    def __init__(self, id, vehicle_type, sumo) -> None:
        super().__init__(id, vehicle_type, sumo)

    def execute(self, lane_change:int,
                current_speed:float, current_lane_index:int, current_road_id:str,
                *args, **kwargs,
        ) -> None:
        action = LaneActionType(lane_change)
        if action == LaneActionType.keep_lane:
            target_speed = min(current_speed + 3, 15) # 15m/s -- 54km/h
            self.sumo.vehicle.slowDown(self.veh_id, target_speed, duration=1)
        elif action == LaneActionType.slow_down:
            target_speed = max(current_speed - 3, 2)
            self.sumo.vehicle.slowDown(self.veh_id, target_speed, duration=1)
        elif action == LaneActionType.change_lane_left:
            target_speed = max(current_speed - 2, 2)
            self.sumo.vehicle.slowDown(self.veh_id, target_speed, duration=1)
            self.change_lane(-1, current_lane=current_lane_index, current_edge=current_road_id)
        elif action == LaneActionType.change_lane_right:
            target_speed = max(current_speed - 2, 2)
            self.sumo.vehicle.slowDown(self.veh_id, target_speed, duration=1)
            self.change_lane(1, current_lane=current_lane_index, current_edge=current_road_id)
