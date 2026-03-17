"""
MobileNetV2 骨折分类模型
任务：腓骨下骨（os subfibulare）vs 撕脱骨折（avulsion fracture）二分类
输入：224×224 单通道灰度 ROI
输出：sigmoid 概率值
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os


# ─────────────────────────────────────────────
# 1. 数据集定义
# ─────────────────────────────────────────────
class FractureDataset(Dataset):
    """
    目录结构：
      root/
        train/  os_subfibulare/  *.png
                avulsion/        *.png
        val/    ...
        test/   ...
    label: 0 = os subfibulare, 1 = avulsion fracture
    """
    def __init__(self, root_dir, split="train", transform=None):
        self.samples   = []
        self.transform = transform
        label_map = {"os_subfibulare": 0, "avulsion": 1}

        for cls_name, label in label_map.items():
            folder = os.path.join(root_dir, split, cls_name)
            for fname in os.listdir(folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (os.path.join(folder, fname), label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # 灰度单通道 → 复制为三通道以匹配 ImageNet 预训练权重
        image = Image.open(img_path).convert("L")
        image = Image.merge("RGB", [image, image, image])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────
# 2. 数据增强 & 预处理
# ─────────────────────────────────────────────
def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ─────────────────────────────────────────────
# 3. 模型构建
# ─────────────────────────────────────────────
class MobileNetV2Classifier(nn.Module):
    """
    MobileNetV2 迁移学习分类器
    ┌─────────────────────────────────────────┐
    │  输入: 224×224×3（灰度复制三通道）       │
    │  主干: MobileNetV2（ImageNet 预训练）    │
    │        17 个 Inverted Residual Blocks    │
    │  池化: GlobalAveragePooling (1280-d)     │
    │  分类: Linear(1280→1) + Sigmoid          │
    └─────────────────────────────────────────┘
    """
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # 提取 features（含 17 个 Inverted Residual）+ 初始卷积
        self.features = backbone.features   # output: (B, 1280, 7, 7)

        # 冻结底层特征（可选，适合极小样本场景）
        if freeze_backbone:
            for param in self.features[:14].parameters():
                param.requires_grad = False

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))   # → (B, 1280, 1, 1)

        # 单神经元分类头
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 1),   # sigmoid 二分类
        )

    def forward(self, x):
        x = self.features(x)           # (B, 1280, 7, 7)
        x = self.gap(x)                # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)        # (B, 1280)
        x = self.classifier(x)         # (B, 1)
        return torch.sigmoid(x).squeeze(1)  # (B,)


# ─────────────────────────────────────────────
# 4. 训练 & 验证
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss  = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += ((preds >= 0.5).float() == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss  = criterion(preds, labels)

            total_loss += loss.item() * images.size(0)
            correct    += ((preds >= 0.5).float() == labels).sum().item()
            all_probs.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    return total_loss / n, correct / n, np.array(all_probs), np.array(all_labels)


def train_mobilenetv2(data_root, num_epochs=50, lr=1e-4, batch_size=16,
                      save_path="mobilenetv2_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MobileNetV2Classifier(pretrained=True).to(device)

    train_ds = FractureDataset(data_root, "train", get_transforms("train"))
    val_ds   = FractureDataset(data_root, "val",   get_transforms("val"))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_dl, criterion, device)

        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val   Loss: {va_loss:.4f} Acc: {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ 最佳模型已保存 (Val Acc: {best_val_acc:.4f})")

    print(f"\n训练完成，最佳验证准确率: {best_val_acc:.4f}")
    return model


# ─────────────────────────────────────────────
# 5. 测试集评估（Acc / Sen / Spe / AUC）
# ─────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt


def evaluate_test_set(model_path, data_root, batch_size=16):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = MobileNetV2Classifier(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    test_ds = FractureDataset(data_root, "test", get_transforms("test"))
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss()
    _, _, probs, labels = evaluate(model, test_dl, criterion, device)
    preds = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    acc = accuracy_score(labels, preds)
    sen = tp / (tp + fn)      # sensitivity / recall
    spe = tn / (tn + fp)      # specificity
    auc = roc_auc_score(labels, probs)

    print("─" * 40)
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Sensitivity : {sen:.4f}")
    print(f"  Specificity : {spe:.4f}")
    print(f"  AUC         : {auc:.4f}")
    print("─" * 40)

    # ROC 曲线
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"MobileNetV2 (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    plt.savefig("roc_mobilenetv2.png", dpi=150)
    plt.show()

    return {"acc": acc, "sen": sen, "spe": spe, "auc": auc}


# ─────────────────────────────────────────────
# 6. SHAP 可解释性可视化
# ─────────────────────────────────────────────
import shap
import cv2


def shap_explain(model_path, sample_images, data_root):
    """
    使用 GradientExplainer 生成 SHAP 热力图
    sample_images: list of image paths（来自测试集）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MobileNetV2Classifier(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tfm       = get_transforms("test")
    bg_ds     = FractureDataset(data_root, "test", tfm)
    bg_loader = DataLoader(bg_ds, batch_size=20, shuffle=True)
    background, _ = next(iter(bg_loader))
    background = background.to(device)

    # SHAP GradientExplainer
    explainer = shap.GradientExplainer(model, background)

    for img_path in sample_images:
        img = Image.open(img_path).convert("L")
        img = Image.merge("RGB", [img, img, img])
        x   = tfm(img).unsqueeze(0).to(device)

        shap_values = explainer.shap_values(x)          # (1, 3, 224, 224)
        shap_map    = np.abs(shap_values[0]).mean(axis=0)  # (224, 224)
        shap_map    = (shap_map - shap_map.min()) / (shap_map.max() + 1e-8)
        heatmap     = cv2.applyColorMap(
            (shap_map * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        fname = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(f"shap_{fname}.png", heatmap)
        print(f"SHAP 热力图已保存: shap_{fname}.png")


if __name__ == "__main__":
    # 训练
    # train_mobilenetv2("data", num_epochs=50)

    # 评估
    # evaluate_test_set("mobilenetv2_best.pth", "data")

    # SHAP 可视化
    # shap_explain("mobilenetv2_best.pth", ["data/test/avulsion/001.png"], "data")
    pass
