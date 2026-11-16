from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path

from tshub.tshub_env3d.vis3d_sumonet_convert.sumonet_to_tshub3d import SumoNet3D

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), terminal_log_level='INFO')

if __name__ == '__main__':
    netxml = path_convert("./sumo/merge.net.xml") # 路网路径
    sumonet_to_3d = SumoNet3D(net_file=netxml)
    sumonet_to_3d.to_glb(glb_dir=path_convert(f"./model/"))