import trimesh
from typing import Final

OLD_TRIMESH: Final[bool] = tuple(int(d) for d in trimesh.__version__.split(".")) <= (
    3,
    9,
    29,
)