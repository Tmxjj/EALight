from tshub.sumo_tools.interpolation.values_interpolation import InterpolationValues
from tshub.sumo_tools.interpolation.visualize_interpolations import visualize_interpolations

flow = [10,20,30,20,10] # 每个时间段的车流量, 每分钟多少车
period = [5,5,5,5,5] # 每个时间段的持续时间
smooth_flow = InterpolationValues(values=flow, intervals=period)

# test get_smooth_turndef
smooth_flow_list = smooth_flow.get_smooth_values()
print(smooth_flow_list)

# visualize
visualize_interpolations(values=flow, period=period, smooth_values=smooth_flow_list)