import sumolib
from typing import List

def get_tlsID_list(network_file) -> List[str]:
    """返回一个路网文件所有的 Traffic Light Signal ID list

    Args:
        network_file (str): network 文件所在的路径

    Returns:
        list: tls id 组成的列表
    """
    tls_list = []
    net = sumolib.net.readNet(network_file)
    for node in net.getNodes():
        if node.getType() == 'traffic_light':
            tls_list.append(node.getID())
    return tls_list
