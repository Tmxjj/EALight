import sumolib
from loguru import logger
from ..sumo_infos.traffic_light_ids import get_tlsID_list

def generate_traffic_lights_additions(network_file: str, output_file: str) -> None:
    """
    生成用于Sumo的交通信号灯添加文件。
    
    Args:
        network_file (str): Sumo网络文件路径。
        output_file (str): 输出的Sumo添加文件路径。
    """
    tls_ids = get_tlsID_list(network_file=network_file)
    types = ['SaveTLSStates', 'SaveTLSProgram', 'SaveTLSSwitchStates', 'SaveTLSSwitchTimes']
    tls_add_xml = sumolib.xml.create_document("additional")

    for event_type in types:
        for tls_id in tls_ids:
            time_event_xml = tls_add_xml.addChild("timedEvent")
            time_event_xml.setAttribute("type", event_type)
            time_event_xml.setAttribute("source", tls_id)
            time_event_xml.setAttribute("dest", f'{event_type}_{tls_id}.out.xml')

    with open(output_file, 'w') as tls_add_file:
        tls_add_file.write(tls_add_xml.toXML())

    logger.info(f'SIM: 完成 TLS Additions 文件写入, {output_file}。')
