"""
本脚本是配置文件所在的脚本
"""
from pprint import pprint


class Config:
    """配置文件类"""
    # 数据集相关的配置
    root = "/home/caolei/ALL_data/VOC2007/"
    image_set = "ALL_train"


OPT = Config()
