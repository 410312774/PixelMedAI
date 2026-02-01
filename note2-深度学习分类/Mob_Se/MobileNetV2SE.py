import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True),
    )


class SEAttention3D(nn.Module):
    """
    Squeeze-and-Excitation for 3D feature maps: (N, C, D, H, W)
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        assert channels > 0
        assert reduction > 0
        hidden = max(1, channels // reduction)

        self.pool = nn.AdaptiveAvgPool3d(1)  # -> (N, C, 1, 1, 1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se=False, se_reduction=8):
        super().__init__()
        self.stride = stride
        assert isinstance(stride, tuple) and len(stride) == 3

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = (self.stride == (1, 1, 1)) and (inp == oup)
        self.use_se = use_se

        if expand_ratio == 1:
            layers = [
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(oup),
            ]
        else:
            layers = [
                # pw
                nn.Conv3d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(oup),
            ]

        self.conv = nn.Sequential(*layers)
        self.se = SEAttention3D(oup, reduction=se_reduction) if use_se else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        if self.use_res_connect:
            return x + out
        return out


class MobileNetV2SE(nn.Module):
    """
    3D MobileNetV2 + SE channel attention (MobileNetV2-SE).
    """
    def __init__(
        self,
        num_classes=1000,
        sample_size=224,
        width_mult=1.0,
        in_channels=1,
        se_reduction=8,
        se_in_blocks=True,  # True: SE after each block; False: SE only once before classifier
    ):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            # t, c, n, s
            [1,  16, 1, (1, 1, 1)],
            [6,  24, 2, (2, 2, 2)],
            [6,  32, 3, (2, 2, 2)],
            [6,  64, 4, (2, 2, 2)],
            [6,  96, 3, (1, 1, 1)],
            [6, 160, 3, (2, 2, 2)],
            [6, 320, 1, (1, 1, 1)],
        ]

        assert sample_size % 16 == 0, "sample_size should be divisible by 16"

        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        features = []
        # first layer: note your original stride is (1,2,2)
        features.append(conv_bn(in_channels, input_channel, stride=(1, 2, 2)))

        # inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride=stride,
                        expand_ratio=t,
                        use_se=se_in_blocks,
                        se_reduction=se_reduction,
                    )
                )
                input_channel = output_channel

        # last conv
        features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*features)

        # optional: SE only once at the end (if not in blocks)
        self.tail_se = SEAttention3D(self.last_channel, reduction=se_reduction) if (not se_in_blocks) else nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.tail_se(x)
        x = F.adaptive_avg_pool3d(x, 1)  # (N, C, 1, 1, 1)
        x = torch.flatten(x, 1)          # (N, C)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()
    if ft_portion == "last_layer":
        ft_module_names = ["classifier"]
        parameters = []
        for k, v in model.named_parameters():
            if any(n in k for n in ft_module_names):
                parameters.append({"params": v})
            else:
                parameters.append({"params": v, "lr": 0.0})
        return parameters
    raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def get_model(**kwargs):
    return MobileNetV2SE(**kwargs)


if __name__ == "__main__":
    # Example: (N, C, D, H, W)
    model = get_model(num_classes=600, sample_size=112, width_mult=1.0, in_channels=3,
                      se_reduction=8, se_in_blocks=True)

    x = torch.randn(8, 3, 16, 112, 112)
    y = model(x)
    print("Output:", y.shape)  # (8, 600)
