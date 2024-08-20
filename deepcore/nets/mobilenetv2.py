import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
import numpy as np
from typing import Dict, Type, Any, Callable, Union, List, Optional
from .lsq_layer import Conv2dLSQ, LinearLSQ, ActLSQ
from .nets_utils import EmbeddingRecorder

class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, groups: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None, dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes

# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def quanconv3x3(in_planes, out_planes, num_bits, stride=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return Conv2dLSQ(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, nbits=num_bits, groups=groups, bias=bias)

def quanconv1x1(in_planes, out_planes, num_bits, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return Conv2dLSQ(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, nbits=num_bits, groups=groups, bias=bias)

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
    
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, bitwidth, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class QuantInvertedResidual(nn.Module):
    def __init__(self, inp, oup, bitwidth, stride, expand_ratio):
        super(QuantInvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                quanconv3x3(hidden_dim, hidden_dim, bitwidth, stride, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                quanconv1x1(hidden_dim, oup, bitwidth, 1, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                quanconv1x1(inp, hidden_dim, bitwidth, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                quanconv3x3(hidden_dim, hidden_dim, bitwidth, stride, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                quanconv1x1(hidden_dim, oup, bitwidth, 1, bias=False),
                nn.BatchNorm2d(oup),
            )
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, bitwidth, num_classes=100, width_mult=1., record_embedding: bool = False):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        norm_layer = nn.BatchNorm2d
        if bitwidth != 32:
            block = QuantInvertedResidual
        else:
            block = InvertedResidual
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 1], # NOTE: change stride 2 -> 1 for CIFAR10/100
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        self.last_channel = _make_divisible(1280 * max(1.0, width_mult), 4 if width_mult == 0.1 else 8)
        layers = [ConvBNReLU(3, input_channel, stride=1, norm_layer=norm_layer)]
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult,  4 if width_mult == 0.1 else 8)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(block(input_channel, output_channel, bitwidth, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        layers.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*layers)
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def get_last_layer(self):
        return self.classifier[1]

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.embedding_recorder(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
def Mobilenetv2(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
    pretrained: bool = False, **kwargs):
    return MobileNetV2(num_classes=num_classes, bitwidth=32, width_mult=0.5, record_embedding=record_embedding)

def QMobilenetv2(bitwidth: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
    pretrained: bool = False, **kwargs):
    return MobileNetV2(num_classes=num_classes, bitwidth=bitwidth, width_mult=0.5, record_embedding=record_embedding)

