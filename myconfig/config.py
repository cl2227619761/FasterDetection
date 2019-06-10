"""
本脚本是配置文件所在的脚本
"""
#  from pprint import pprint


class Config:
    """配置文件类"""
    # 数据集相关的配置
    # root = "/home/caolei/code/ALL_data/VOC2007/"
    root = "/home/dl/code/caolei/ALL_data/VOC2007/"
    # image_set = "ALL_train"
    image_set = "ALL"
    # 数据集标签类别
    box_label_names = ("__background__", "正常", "异常")

    # 模型训练相关的参数
    num_epochs = 14


OPT = Config()
