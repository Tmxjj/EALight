import enum

class vehicle_action_type(enum.Enum):
    Lane = 'lane'
    """
    Action= ``str``. Discrete lane action from one of
    
    + "keep_lane",
    + "slow_down", 
    + "change_lane_left", and 
    + "change_lane_right".
    """

    Speed = 'speed'
    """
    Action= ``str``. Discrete speed action from one of
    
    + "accelerate",
    + "decelerate",
    + "maintain_speed".
    """

    LaneWithContinuousSpeed = 'lane_continuous_speed'
    """
    Action=(target_speed, lane_change).

    + target_speed: Baseline target speed (controller may give more or less regardless). Type=float.
    + lane_change: Discrete lane change value. Can be one of 
        + -1 : change to right lane
        + 0 : keep to current lane
        + 1 : change to left lane
    """