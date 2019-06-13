"""
本脚本是关于模型的定义
"""
import sys
import copy

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from tools.engine import train_one_epoch, evaluate
import tools.transforms as T
from tools import utils

sys.path.append("../")
try:
    from data.data import ALLDetection
    from myconfig.config import OPT
except Exception:
    raise


# 加载预训练模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# 替换分类器适配自定义类别
num_classes = OPT.num_classes  # 2 classes (正常+异常) + background
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
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    dataset = ALLDetection(transforms=get_transform(train=True))
    dataset_test = ALLDetection(transforms=get_transform(train=False))
    # 打乱并且划分数据集
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_val = torch.utils.data.Subset(dataset_test, indices[-50:-25])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-25:])

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    num_epochs = OPT.num_epochs

    best_score = 0.0
    best_state_dict = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, dataloader,
                        device, epoch, print_freq=10)
        lr_scheduler.step()
        res = evaluate(model, dataloader_val, device=device)
        mAP = res.coco_eval['bbox'].stats[1]
        if mAP > best_score:
            best_score = mAP
            best_state_dict = copy.deepcopy(model.state_dict())
    print("训练完成")
    print('Best valid mAP: %.4f' % best_score)
    torch.save(best_state_dict, 'best_model3.pth')

    print("测试开始")
    model.load_state_dict(best_state_dict)
    res = evaluate(model, dataloader_test, device=device)
    test_mAP = res.coco_eval['bbox'].stats[1]
    print('test mAP: %.4f' % test_mAP)


if __name__ == "__main__":
    main()
