## README.md（可直接放 GitHub）

# MobileNetV2-SE (3D) in PyTorch

A lightweight **MobileNetV2** backbone enhanced with a **channel attention mechanism** (SE, Squeeze-and-Excitation) to improve feature extraction and classification performance on 3D medical imaging data.

## Overview

In this project, we design a novel deep convolutional neural network by integrating the **SE attention module** with the classic lightweight network **MobileNetV2** (MobileNetV2-SE).  
The SE module introduces **channel-wise attention** that dynamically recalibrates feature responses with three components:

- Global pooling  
- Adaptive fully connected structure  
- Weight recalibration  

This lightweight attention module improves representational power without significantly increasing computational cost.

## Key Settings (as used in our method)

- **SE reduction ratio**: 8  
- **Backbone**: default MobileNetV2 inverted residual blocks, adjustable via `width_mult`
- **Initialization**:
  - Convolution layers: Kaiming initialization
  - BatchNorm: weight = 1, bias = 0
  - Linear layers: normal(mean=0, std=0.01), bias = 0
- **Transfer learning**:
  - ImageNet pre-trained MobileNetV2 weights are used for initialization
  - Classification head is reset for downstream tasks
  - Pre-trained head parameters are removed; only feature extractor parameters are retained

## Environment

- Python 3.8  
- PyTorch 2.0  
- GPU: RTX 4060 Ti (training environment)

## Usage

### 1) Create the model

```python
from model import get_model  # adjust import to your filename

model = get_model(
    num_classes=600,
    sample_size=112,
    width_mult=1.0,
    in_channels=3,
    se_reduction=8,
    se_in_blocks=True
)
```

### 2) Forward test

```python
import torch

x = torch.randn(8, 3, 16, 112, 112)  # (N, C, D, H, W)
y = model(x)
print(y.shape)  # (8, 600)
```

### 3) Fine-tuning options

- Fine-tune all parameters:

```python
params = get_fine_tuning_parameters(model, "complete")
```

- Only train the classifier layer:

```python
params = get_fine_tuning_parameters(model, "last_layer")
```

## Notes on Our Pipeline (Method Summary)

To integrate image features from native T1 mapping, ADC and T2* mapping, we use MobileNetV2-SE as a feature extractor (removing the final classification layers), concatenate modality features along the channel dimension, and then feed them into a multi-head self-attention module. The fused representation is finally decoded by an MLP classifier to predict renal fibrosis status.

## Citation

If you use this repository, please cite:

- **MobileNetV2: Inverted Residuals and Linear Bottlenecks**
- **Squeeze-and-Excitation Networks**
