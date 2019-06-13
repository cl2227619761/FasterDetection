import sys
import time

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw

from nms import box_nms
sys.path.append("../")
try:
    from myconfig.config import OPT
    from data.data import ALLDetection
    import mymodel.tools.transforms as T
    from mymodel.tools import utils
    from mymodel.tools.engine import CocoEvaluator, _get_iou_types
    from mymodel.tools.coco_utils import get_coco_api_from_dataset
except Exception:
    raise


def get_transform(train):
    """数据增强操作"""
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
# device = torch.device("cpu")
device = torch.device("cuda")
dataset = ALLDetection(transforms=get_transform(train=True))
dataset_test = ALLDetection(transforms=get_transform(train=False))
# 打乱并且划分数据集
torch.manual_seed(7)
torch.cuda.manual_seed(7)
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



# 加载预训练模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# 替换分类器适配自定义类别
num_classes = OPT.num_classes  # 2 classes (正常+异常) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# model.load_state_dict()
model_dict = torch.load("../mymodel/best_model3.pth")
model.load_state_dict(model_dict)

model.eval()
model.cuda()
# for idx, case in enumerate(dataset_test):
#     img = case[0]
#     prediction = model([img])
#     box = prediction[0]["boxes"]
#     score = prediction[0]["scores"]
#     label = prediction[0]["labels"]
#     keep = box_nms(box, score)
#     box = box[keep]
#     score = score[keep]
#     label = label[keep]

#     true_box = case[1]["boxes"]
#     true_label = case[1]["labels"]
#     # box1 = box[:, None, :]
#     # res = abs(box1 - box)
#     topil = ToPILImage()
#     image = topil(img)
#     draw = ImageDraw.Draw(image)
#     for i in range(box.shape[0]):
#         if label[i].item() == 1:
#             color = (0, 255, 0)
#         else:
#             color = (255, 0, 0)
#         draw.rectangle(box[i].tolist(), outline=color)
#     for j in range(true_box.shape[0]):
#         if true_label[j].item() == 1:
#             color = (0, 127, 0)
#         else:
#             color = (127, 0, 0)
#         draw.rectangle(true_box[j].tolist(), outline=color)
#     image.save("./pics/%d.png" % idx)

coco = get_coco_api_from_dataset(dataset_test.dataset)
iou_types = _get_iou_types(model)
coco_evaluator = CocoEvaluator(coco, iou_types)
metric_logger = utils.MetricLogger(delimiter="  ")
header = 'Test:'
for image, targets in metric_logger.log_every(dataloader_test, 100, header):
    image = list(img.to(device) for img in image)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    torch.cuda.synchronize()
    model_time = time.time()
    outputs = model(image)

    for output in outputs:
        box = output["boxes"]
        score = output["scores"]
        label = output["labels"]
        keep = box_nms(box, score, threshold=0.8)
        output["boxes"] = box[keep]
        output["scores"] = score[keep]
        output["labels"] = label[keep]


    outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
    model_time = time.time() - model_time
    res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
    evaluator_time = time.time()
    coco_evaluator.update(res)
    evaluator_time = time.time() - evaluator_time
    metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

metric_logger.synchronize_between_processes()
print("Averaged stats:", metric_logger)
coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()

