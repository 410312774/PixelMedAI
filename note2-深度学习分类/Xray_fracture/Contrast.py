"""
对比骨干网络：ResNet18 / ResNet34 / EfficientNet-B0
与 MobileNetV2 在相同训练策略下进行系统性对比
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights
)
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, roc_curve
)

# 复用 MobileNetV2 代码中的数据集和变换
# from mobilenetv2_classifier import FractureDataset, get_transforms, evaluate


# ─────────────────────────────────────────────
# 1. ResNet18
# ─────────────────────────────────────────────
class ResNet18Classifier(nn.Module):
    """
    ResNet18：18 层残差网络
    残差结构缓解梯度消失，全局平均池化后接二分类头
    ┌──────────────────────────────────────────┐
    │  输入: 224×224×3                          │
    │  主干: ResNet18（ImageNet 预训练）         │
    │        8 个 BasicBlock（含跳接）          │
    │  池化: AdaptiveAvgPool → 512-d            │
    │  分类: Linear(512→1) + Sigmoid            │
    └──────────────────────────────────────────┘
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights  = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # 移除原 FC，保留特征提取部分
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.features(x)          # (B, 512, 1, 1)
        x = torch.flatten(x, 1)       # (B, 512)
        x = self.classifier(x)        # (B, 1)
        return torch.sigmoid(x).squeeze(1)


# ─────────────────────────────────────────────
# 2. ResNet34
# ─────────────────────────────────────────────
class ResNet34Classifier(nn.Module):
    """
    ResNet34：34 层残差网络
    比 ResNet18 更深，特征表达更丰富，适合中等规模数据
    ┌──────────────────────────────────────────┐
    │  输入: 224×224×3                          │
    │  主干: ResNet34（ImageNet 预训练）         │
    │        16 个 BasicBlock                   │
    │  池化: AdaptiveAvgPool → 512-d            │
    │  分类: Linear(512→1) + Sigmoid            │
    └──────────────────────────────────────────┘
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights  = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)

        self.features   = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return torch.sigmoid(x).squeeze(1)


# ─────────────────────────────────────────────
# 3. EfficientNet-B0
# ─────────────────────────────────────────────
class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0：复合缩放策略，兼顾精度与效率
    深度/宽度/分辨率同步扩展，MBConv 模块含 SE 注意力机制
    ┌──────────────────────────────────────────┐
    │  输入: 224×224×3                          │
    │  主干: EfficientNet-B0（ImageNet 预训练） │
    │        MBConv Blocks + SE 注意力          │
    │  池化: AdaptiveAvgPool → 1280-d           │
    │  分类: Linear(1280→1) + Sigmoid           │
    └──────────────────────────────────────────┘
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights  = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.features   = backbone.features     # MBConv 特征提取
        self.avgpool    = backbone.avgpool       # AdaptiveAvgPool2d(1,1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 1),
        )

    def forward(self, x):
        x = self.features(x)           # (B, 1280, 7, 7)
        x = self.avgpool(x)            # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)        # (B, 1280)
        x = self.classifier(x)         # (B, 1)
        return torch.sigmoid(x).squeeze(1)


# ─────────────────────────────────────────────
# 4. 统一训练接口
# ─────────────────────────────────────────────
MODEL_REGISTRY = {
    "resnet18":       ResNet18Classifier,
    "resnet34":       ResNet34Classifier,
    "efficientnet_b0": EfficientNetB0Classifier,
}


def train_model(model_name, data_root, num_epochs=50,
                lr=1e-4, batch_size=16):
    """
    统一训练入口，与 MobileNetV2 使用完全相同的：
      - 数据集划分  - 数据增强  - 损失函数(BCE)
      - 优化器(Adam, lr=1e-4)  - Batch Size(16)
    """
    from mobilenetv2_classifier import FractureDataset, get_transforms, evaluate  # noqa

    assert model_name in MODEL_REGISTRY, \
        f"未知模型: {model_name}，可选: {list(MODEL_REGISTRY.keys())}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MODEL_REGISTRY[model_name](pretrained=True).to(device)

    train_ds = FractureDataset(data_root, "train", get_transforms("train"))
    val_ds   = FractureDataset(data_root, "val",   get_transforms("val"))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    best_val_acc = 0.0
    save_path    = f"{model_name}_best.pth"

    for epoch in range(num_epochs):
        # ── 训练 ──
        model.train()
        tr_loss, tr_correct = 0.0, 0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss  = criterion(preds, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tr_loss    += loss.item() * images.size(0)
            tr_correct += ((preds >= 0.5).float() == labels).sum().item()

        tr_acc = tr_correct / len(train_ds)

        # ── 验证 ──
        va_loss, va_acc, _, _ = evaluate(model, val_dl, criterion, device)

        print(f"[{model_name}] Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Train Loss: {tr_loss/len(train_ds):.4f} Acc: {tr_acc:.4f} | "
              f"Val   Loss: {va_loss:.4f} Acc: {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ {model_name} 最佳模型已保存 (Val Acc: {best_val_acc:.4f})")

    return model, save_path


# ─────────────────────────────────────────────
# 5. 多模型对比评估
# ─────────────────────────────────────────────
def compare_all_models(data_root, model_paths: dict, batch_size=16):
    """
    model_paths = {
        "mobilenetv2":    "mobilenetv2_best.pth",
        "resnet18":       "resnet18_best.pth",
        "resnet34":       "resnet34_best.pth",
        "efficientnet_b0": "efficientnet_b0_best.pth",
    }
    """
    from mobilenetv2_classifier import (               # noqa
        MobileNetV2Classifier, FractureDataset,
        get_transforms, evaluate
    )

    ALL_MODELS = {
        "mobilenetv2":    MobileNetV2Classifier,
        "resnet18":       ResNet18Classifier,
        "resnet34":       ResNet34Classifier,
        "efficientnet_b0": EfficientNetB0Classifier,
    }

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    test_ds   = FractureDataset(data_root, "test", get_transforms("test"))
    test_dl   = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}
    plt.figure(figsize=(7, 6))

    for name, path in model_paths.items():
        model = ALL_MODELS[name](pretrained=False).to(device)
        model.load_state_dict(torch.load(path, map_location=device))

        _, _, probs, labels = evaluate(model, test_dl, criterion, device)
        preds = (probs >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        acc = accuracy_score(labels, preds)
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        auc = roc_auc_score(labels, probs)

        results[name] = {"acc": acc, "sen": sen, "spe": spe, "auc": auc}

        # ROC 曲线叠加
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    # 汇总打印
    print(f"\n{'模型':<18} {'Accuracy':>10} {'Sensitivity':>12} {'Specificity':>12} {'AUC':>8}")
    print("─" * 65)
    for name, m in results.items():
        print(f"{name:<18} {m['acc']:>10.4f} {m['sen']:>12.4f} {m['spe']:>12.4f} {m['auc']:>8.4f}")

    # ROC 图保存
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_comparison.png", dpi=150)
    plt.show()

    return results


# ─────────────────────────────────────────────
# 6. 主程序入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    DATA_ROOT = "data"

    # ── 依次训练三个对比模型 ──
    # for name in ["resnet18", "resnet34", "efficientnet_b0"]:
    #     train_model(name, DATA_ROOT, num_epochs=50)

    # ── 多模型对比评估 ──
    # compare_all_models(DATA_ROOT, {
    #     "mobilenetv2":    "mobilenetv2_best.pth",
    #     "resnet18":       "resnet18_best.pth",
    #     "resnet34":       "resnet34_best.pth",
    #     "efficientnet_b0": "efficientnet_b0_best.pth",
    # })
    pass
