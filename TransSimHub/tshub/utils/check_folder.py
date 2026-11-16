import os
from loguru import logger

def check_folder(folder_path) -> None:
    """判断文件夹是否存在, 如果不存在则进行创建

    Args:
        folder_path (str): 文件夹的路径
    """
    if not os.path.isdir(folder_path): # 判断路径是否存在
        os.makedirs(folder_path, exist_ok=True) # 不存在则创建
        logger.warning('SIM: 文件夹 {} 不存在, 已创建!'.format(folder_path))