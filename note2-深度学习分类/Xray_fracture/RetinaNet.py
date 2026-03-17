"""
RetinaNet 踝关节目标检测模型
基于 MS COCO 预训练权重，通过迁移学习定位双侧踝关节 ROI
"""

import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import json


# ─────────────────────────────────────────────
# 1. 数据集定义
# ─────────────────────────────────────────────
class AnkleDetectionDataset(Dataset):
    """
    踝关节检测数据集
    标注格式：COCO JSON，类别仅含 'ankle'（class_id=1）
    """
    def __init__(self, img_dir, annotation_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        with open(annotation_file, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.annotations = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            self.annotations.setdefault(img_id, []).append(ann)
        self.img_ids = list(self.images.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        # 灰度图转 RGB（RetinaNet 需要三通道输入）
        image = Image.open(img_path).convert("RGB")
        image = transforms.ToTensor()(image)

        anns = self.annotations.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # 仅一类：ankle

        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        return image, target


# ─────────────────────────────────────────────
# 2. 模型构建（迁移学习）
# ─────────────────────────────────────────────
def build_retinanet(num_classes=2, pretrained=True):
    """
    加载 COCO 预训练 RetinaNet，替换分类头以适配踝关节单类检测
    num_classes = 背景(0) + ankle(1) = 2
    """
    weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model = retinanet_resnet50_fpn(weights=weights)

    # 替换检测头
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels
    model.head = RetinaNetHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    return model


# ─────────────────────────────────────────────
# 3. 训练流程
# ─────────────────────────────────────────────
def train_retinanet(img_dir, ann_file, num_epochs=20, lr=1e-4, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_retinanet(num_classes=2, pretrained=True).to(device)

    dataset = AnkleDetectionDataset(img_dir, ann_file)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "retinanet_ankle.pth")
    print("模型已保存至 retinanet_ankle.pth")
    return model


# ─────────────────────────────────────────────
# 4. ROI 裁剪与质量过滤
# ─────────────────────────────────────────────
def filter_and_crop_roi(
    model, image_path, score_threshold=0.5,
    min_area=500, save_dir="rois"
):
    """
    对单张 X 线影像进行检测、过滤并裁剪 ROI
    过滤条件：
      - 置信度 < score_threshold → 丢弃
      - 检测框面积过小（不含完整踝关节）→ 丢弃
      - 含踝关节假体的影像需在调用前人工或规则预筛
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    image = Image.open(image_path).convert("RGB")
    img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    img_name   = os.path.splitext(os.path.basename(image_path))[0]

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    roi_paths = []
    for i, (box, score) in enumerate(
        zip(predictions["boxes"], predictions["scores"])
    ):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            print(f"  [跳过] 检测框面积过小: {area}px²")
            continue

        roi = image.crop((x1, y1, x2, y2))
        save_path = os.path.join(save_dir, f"{img_name}_roi_{i}.png")
        roi.save(save_path)
        roi_paths.append(save_path)
        print(f"  [保存] ROI {i}: score={score:.3f}, box=({x1},{y1},{x2},{y2})")

    return roi_paths


# ─────────────────────────────────────────────
# 5. 批量推理入口
# ─────────────────────────────────────────────
def batch_inference(model_path, image_folder, save_dir="rois"):
    model = build_retinanet(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".dcm"))
    ]

    all_rois = {}
    for img_path in image_files:
        print(f"处理: {img_path}")
        rois = filter_and_crop_roi(model, img_path, save_dir=save_dir)
        all_rois[img_path] = rois

    return all_rois


if __name__ == "__main__":
    # 示例：训练
    # train_retinanet("data/images", "data/annotations.json")

    # 示例：批量推理 + ROI 裁剪
    # batch_inference("retinanet_ankle.pth", "data/test_images", save_dir="output_rois")
    pass
