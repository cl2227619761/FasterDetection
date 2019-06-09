"""
本脚本是配置文件所在的脚本
"""
#  from pprint import pprint


class Config:
    """配置文件类"""
    # 数据集相关的配置
    root = "/home/caolei/code/ALL_data/VOC2007/"
    image_set = "ALL_train"
    # 数据集标签类别
    box_label_names = ("正常", "异常")


OPT = Config()
