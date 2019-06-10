"""
本脚本是关于模型的定义
"""
import sys

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from tools.engine import train_one_epoch, evaluate
import tools.transforms as T
from tools import utils

sys.path.append("../")
try:
    from data.data import ALLDetection
except Exception:
    raise


# 加载预训练模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# 替换分类器适配自定义类别
num_classes = 3  # 2 classes (正常+异常) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


def get_transform(train):
    """数据增强操作"""
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    """训练主函数"""
    # device = torch.device("cpu")
    device = torch.device("cuda")
    dataset = ALLDetection(transforms=get_transform(train=True))
    dataset_test = ALLDetection(transforms=get_transform(train=False))
    # 打乱并且划分数据集
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    num_epochs = 50

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, dataloader,
                        device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, dataloader_test, device=device)
    print("训练完成")


if __name__ == "__main__":
    main()
