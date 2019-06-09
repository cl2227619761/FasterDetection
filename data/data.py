"""
本脚本是关于数据集的准备
"""
import os
import sys
import collections
import xml.etree.ElementTree as ET

import torch
from torchvision.datasets import VisionDataset
from PIL import Image

sys.path.append("../")
try:
    from myconfig.config import OPT
except Exception:
    raise


# 定义数据集，目标检测的数据集的准备类似于普通的CNN
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
            image_set: 要使用的数据集的名称比如train, trainval
            transform: 函数，接受PIL image, 返回处理后的对象
            target_transform: 函数，对target作相应的处理
        """
        # 继承了VisionDataset, 具体参看VisionDataset源码(TODO)
        super(ALLDetection, self).__init__(root, transforms, transform,
                                           target_transform)
        self.image_set = image_set
        # 图像所在的文件夹
        image_dir = os.path.join(root, "JPEGImages")
        # 标注文件夹
        annotation_dir = os.path.join(root, "Annotations")
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

        # 图像所在的路径构成的列表
        self.images = [os.path.join(image_dir, x + ".jpg") for x in filenames]
        # 图像对应的标注文件的路径所在的列表
        self.annotations = [
            os.path.join(annotation_dir, x + ".xml") for x in filenames
        ]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        参数：
            index: 图像的索引
        返回：
            (image, target)组成的元组
        """
        img = Image.open(self.images[index]).convert("RGB")
        pre_target = self.parse_xml(
            ET.parse(self.annotations[index]).getroot())
        objs = pre_target["annotation"]["object"]
        # 为了防止出现单个框的时候，将字典转成列表，防止出错
        if isinstance(objs, dict):
            objs = [objs]
        num_objs = len(objs)
        boxes = []
        labels = []
        try:
            for i in range(num_objs):
                xmin = float(objs[i]["bndbox"]["xmin"])
                ymin = float(objs[i]["bndbox"]["ymin"])
                xmax = float(objs[i]["bndbox"]["xmax"])
                ymax = float(objs[i]["bndbox"]["ymax"])
                box_ws = xmax - xmin
                box_hs = ymax - ymin
                if box_ws > 50 and box_hs > 50:
                    boxes.append([xmin, ymin, xmax, ymax])

                    name = objs[i]["name"]
                    labels.append(OPT.box_label_names.index(name))
        except KeyError:
            import ipdb; ipdb.set_trace()

        # 转为torch张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=1920)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=1200)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_xml(self, node):
        """解析xml文件，返回xml文件内容到字典中"""
        ALL_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            ALL_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                ALL_dict[node.tag] = text
        return ALL_dict


def main():
    """调试用"""
    dataset = ALLDetection()
    img, target = dataset[0]
    print(img)
    print(target)


if __name__ == "__main__":
    main()
