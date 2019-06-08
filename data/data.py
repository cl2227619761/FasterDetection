"""
本脚本是关于数据集的生成，用于训练和验证
"""
import os
import sys
sys.path.append("../")

from torchvision.datasets import VisionDataset

from myconfig.config import OPT 


# 定义数据集，目标检测的数据集的准备类似于普通的CNN，也是要继承torch.utils.
# .data的, 里面的__len__和__getitem__方法
class ALLDetection(VisionDataset):
    """仿写源码里面的VOCDetection"""

    def __init__(self,
                 root=OPT.root,
                 image_set=OPT.image_set,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        """
        参数：
            root: Dataset的根目录，例如././VOC2007/
            image_set: 要使用的数据集的名称, 比如train, trainval, val
            transform: 函数，接受PIL image, 返回处理后的对象
            target_transform: 函数，对target作相应的处理
        """
        # 继承了VisionDataset, 具体参看VisionDataset源码(TODO)
        super(ALLDetection, self).__init__(root, transforms,
                                           transform, target_transform)
        self.image_set = image_set
        image_dir = os.path.join(root, "JPEGImages")  # 图像所在的文件夹
        annotation_dir = os.path.join(root, "Annotations")  # 标注文件夹

        if not os.path.isdir(root):
            raise RuntimeError("未发现数据所在目录，请检查!")

        split_dir = os.path.join(root, "ImageSets/Main")
        split_f = os.path.join(split_dir, image_set.rstrip("\n") + ".txt")

        if not os.path.exists(split_f):
            raise ValueError("image_set存在，请重新选择!")

        with open(os.path.join(split_f), "r") as f:
            filenames = [x.strip() for x in f.readlines()]
        # 为了处理那些文件名后缀带.jpg等的名称
        if filenames[0].endswith(".jpg"):
            filenames = [x.strip(".jpg") for x in filenames]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in filenames]
        self.annotations = [os.path.join(annotation_dir, x + ".xml")
                            for x in filenames]
        assert (len(self.images) == len(self.annotations))


def main():
    """调试用"""
    dataset = ALLDetection()


if __name__ == "__main__":
    main()

