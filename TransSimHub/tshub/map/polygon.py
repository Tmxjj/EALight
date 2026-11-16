from dataclasses import dataclass, fields
from typing import Tuple, Dict, Any

@dataclass
class PolygonInfo:
    id: str
    edge_id: str # 如果是 lane, 则包含所属的 edge
    length: float # 如果是 lane, 则包含 lane 的长度
    polygon_type: str # 多边形的类型, 例如 lane, node
    shape: Tuple[Tuple[float, float]] # 多边形的形状
    # For Map
    building_levels: int # 建筑物是几层
    # For Node
    node_coord: Tuple[float] # 如果是 node, 则记录 node 的坐标
    node_type: str # 如果是 node, 则记录 node 的类型, 例如是否是 traffic_light

    @classmethod
    def create(cls,
               id: str,
               edge_id: str,
               length: float,
               polygon_type: str,
               shape: str,
               building_levels: int,
               node_coord: Tuple[float],
               node_type: str
        ):
        if isinstance(shape, str):
            shape_tuple = tuple([tuple(map(float, point.split(","))) for point in shape.split()])
        else:
            shape_tuple = shape
        polygon = cls(
            id, edge_id, length,
            polygon_type,
            shape_tuple, building_levels,
            node_coord, node_type
        )
        return polygon
    
    def get_features(self) -> Dict[str, Any]:
        output_dict = {}
        for field in fields(self):
            field_name = field.name
            field_value = getattr(self, field_name)
            if field_name != 'sumo':
                output_dict[field_name] = field_value
        return output_dict