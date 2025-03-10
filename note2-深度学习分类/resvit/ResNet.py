
import os
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv, Norm, Pool
from monai.networks.layers.utils import get_pool_layer
from monai.utils import deprecated_arg
from monai.utils.module import look_up_option

__all__ = ["ResNet", "resnet10", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet200"]

from pixelmed_calc.utils.about_log import logger

SHORTCUT_MAPPING = {'resnet10': 'B', 'resnet18': 'A', 'resnet34': 'A', 'resnet50': 'B', 'resnet101': 'B',
                    'resnet152': 'B', 'resnet200': 'B', }


def get_inplanes():
    return [64, 128, 256, 512]


def get_avgpool():
    return [0, 1, (1, 1), (1, 1, 1)]


def get_conv1(conv1_t_size: int, conv1_t_stride: int):
    return (
        [0, conv1_t_size, (conv1_t_size, 7), (conv1_t_size, 7, 7)],
        [0, conv1_t_stride, (conv1_t_stride, 2), (conv1_t_stride, 2, 2)],
        [0, (conv1_t_size // 2), (conv1_t_size // 2, 3), (conv1_t_size // 2, 3, 3)],
    )


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_planes: int,
            planes: int,
            spatial_dims: int = 3,
            stride: int = 1,
            downsample: Union[nn.Module, partial, None] = None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels.
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for first conv layer.
            downsample: which downsample layer to use.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = norm_type(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            in_planes: int,
            planes: int,
            spatial_dims: int = 3,
            stride: int = 1,
            downsample: Union[nn.Module, partial, None] = None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels (taking expansion into account).
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for second conv layer.
            downsample: which downsample layer to use.
        """

        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_type(planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.conv3 = conv_type(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_type(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet based on: `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? <https://arxiv.org/pdf/1711.09577.pdf>`_.
    Adapted from `<https://github.com/kenshohara/3D-ResNets-PyTorch/tree/master/models>`_.

    Args:
        block: which ResNet block to use, either Basic or Bottleneck.
        layers: how many layers to use.
        block_inplanes: determine the size of planes at each step. Also tunable with widen_factor.
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of input channels for first convolutional layer.
        conv1_t_size: size of first convolution layer, determines kernel and padding.
        conv1_t_stride: stride of first convolution layer.
        no_max_pool: bool argument to determine if to use maxpool layer.
        shortcut_type: which downsample block to use. Options are 'A', 'B', default to 'B'.
            - 'A': using `self._downsample_basic_block`.
            - 'B': kernel_size 1 conv + norm.
        widen_factor: widen output for each layer.
        num_classes: number of output (classifications).
        feed_forward: whether to add the FC layer for the output, default to `True`.

    .. deprecated:: 0.6.0
        ``n_classes`` is deprecated, use ``num_classes`` instead.

    """

    @deprecated_arg("n_classes", since="0.6")
    def __init__(
            self,
            block: Type[Union[ResNetBlock, ResNetBottleneck]],
            layers: List[int],
            block_inplanes: List[int],
            spatial_dims: int = 3,
            in_channels: int = 3,
            conv1_t_size: int = 7,
            conv1_t_stride: int = 1,
            no_max_pool: bool = False,
            shortcut_type: str = "B",
            widen_factor: float = 1.0,
            num_classes: int = 400,
            feed_forward: bool = True,
            n_classes: Optional[int] = None,
            **kwargs
    ) -> None:
        super().__init__()
        # in case the new num_classes is default but you still call deprecated n_classes
        if n_classes is not None and num_classes == 400:
            num_classes = n_classes

        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        avgp_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        block_avgpool = get_avgpool()
        conv1_kernel, conv1_stride, conv1_padding = get_conv1(conv1_t_size, conv1_t_stride)
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = conv_type(
            in_channels,
            self.in_planes,
            kernel_size=conv1_kernel[spatial_dims],
            stride=conv1_stride[spatial_dims],
            padding=conv1_padding[spatial_dims],
            bias=False,
        )
        self.bn1 = norm_type(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = pool_type(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], spatial_dims, shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], spatial_dims, shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], spatial_dims, shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], spatial_dims, shortcut_type, stride=2)
        self.avgpool = avgp_type(block_avgpool[spatial_dims])
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_classes) if feed_forward else None

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int, spatial_dims: int = 3) -> torch.Tensor:
        out: torch.Tensor = get_pool_layer(("avg", {"kernel_size": 1, "stride": stride}), spatial_dims=spatial_dims)(x)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(
            self,
            block: Type[Union[ResNetBlock, ResNetBottleneck]],
            planes: int,
            blocks: int,
            spatial_dims: int,
            shortcut_type: str,
            stride: int = 1,
    ) -> nn.Sequential:

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        downsample: Union[nn.Module, partial, None] = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if look_up_option(shortcut_type, {"A", "B"}) == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    spatial_dims=spatial_dims,
                )
            else:
                downsample = nn.Sequential(
                    conv_type(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride),
                    norm_type(planes * block.expansion),
                )

        layers = [
            block(
                in_planes=self.in_planes, planes=planes, spatial_dims=spatial_dims, stride=stride, downsample=downsample
            )
        ]

        self.in_planes = planes * block.expansion
        for _i in range(1, blocks):
            layers.append(block(self.in_planes, planes, spatial_dims=spatial_dims))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x=x.view(-1,1,32,8,8)
        

        #x = x.view(x.size(0), -1)
        #if self.fc is not None:
        #    x = self.fc(x)

        return x


def _resnet(arch: str,
            block: Type[Union[ResNetBlock, ResNetBottleneck]],
            layers: List[int],
            block_inplanes: List[int],
            pretrained: Union[str, bool],
            progress: bool,
            transfer_learning: bool = True,
            **kwargs: Any,
            ) -> ResNet:
    kwargs.update({'shortcut_type': SHORTCUT_MAPPING[arch]})
    model: ResNet = ResNet(block, layers, block_inplanes, **kwargs)
    if pretrained:
        if isinstance(pretrained, bool):
            model_path = os.path.join(os.environ.get('ONEKEY_HOME'), 'pretrain', f'{arch}.pth')
            if os.path.exists(model_path):
                pretrained = model_path
            else:
                raise NotImplementedError(
                    "Currently not implemented. You need to manually download weights provided by the paper's author"
                    " and load then to the model with `state_dict`. See https://github.com/Tencent/MedicalNet"
                )
        # Author of paper zipped the state_dict on googledrive,
        # so would need to download, unzip and read (2.8gb file for a ~150mb state dict).
        # Would like to load dict from url but need somewhere to save the state dicts.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(pretrained, map_location=device)['state_dict']
        # print(state_dict.keys())
        for key in list(state_dict.keys()):
            if key.startswith('module.'):
                new_key = key[7:]
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
            if key.startswith('fc.') and transfer_learning:
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
        logger.info(f'成功加载{pretrained}模型参数。')

    return model


def resnet10(pretrained: Union[str, bool] = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-10 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet10", ResNetBlock, [1, 1, 1, 1], get_inplanes(), pretrained, progress, **kwargs)


def resnet18(pretrained: Union[str, bool] = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", ResNetBlock, [2, 2, 2, 2], get_inplanes(), pretrained, progress, **kwargs)


def resnet34(pretrained: Union[str, bool] = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", ResNetBlock, [3, 4, 6, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet50(pretrained: Union[str, bool] = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", ResNetBottleneck, [3, 4, 6, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet101(pretrained: Union[str, bool] = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 8 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", ResNetBottleneck, [3, 4, 23, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet152(pretrained: Union[str, bool] = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 8 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", ResNetBottleneck, [3, 8, 36, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet200(pretrained: Union[str, bool] = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-200 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 8 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet200", ResNetBottleneck, [3, 24, 36, 3], get_inplanes(), pretrained, progress, **kwargs)


if __name__ == '__main__':
    model = ResNet(ResNetBottleneck, [2, 2, 2, 2], get_inplanes(), n_classes=2,in_channels=1)
    indata = torch.rand([4, 1, 32, 128, 128])
    outdata=model(indata)
    #outdata=outdata.view(4,1,32,8,8)
    print(outdata.shape)
