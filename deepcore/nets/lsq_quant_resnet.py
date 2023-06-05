import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled, flatten, Tensor
from .nets_utils import EmbeddingRecorder
from torchvision.models import resnet
from .lsq_layer import Conv2dLSQ, LinearLSQ, ActLSQ

# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def quanconv3x3(in_planes, out_planes, num_bits, stride=1):
    """3x3 convolution with padding"""
    return Conv2dLSQ(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, nbits=num_bits)

def quanconv1x1(in_planes, out_planes, num_bits, stride=1):
    """1x1 convolution"""
    return Conv2dLSQ(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, nbits=num_bits)
 
class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bitwidth=4):
        super(BasicBlock, self).__init__()

        self.bias11 = LearnableBias(in_planes)
        self.prelu1 = nn.PReLU(in_planes)
        self.bias12 = LearnableBias(in_planes)
        self.quan1 = ActLSQ(nbits = bitwidth)
        self.conv1 = quanconv3x3(in_planes, planes, bitwidth, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.bias21 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.bias22 = LearnableBias(planes)
        self.quan2 = ActLSQ(nbits = bitwidth)
        self.conv2 = quanconv3x3(planes, planes, bitwidth)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), # need quantization
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.bias31 = LearnableBias(planes)
        self.prelu3 = nn.PReLU(planes)
        self.bias32 = LearnableBias(planes)

    def forward(self, x):

        out = self.bias11(x)
        out = self.prelu1(out)
        out = self.bias12(out)
        out = self.quan1(out)
        out = self.conv1(out)
        out = self.bn1(out)

        out = self.bias21(out)
        out = self.prelu2(out)
        out = self.bias22(out)
        out = self.quan2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(x)
        out = self.bias31(out)
        out = self.prelu3(out)
        out = self.bias32(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bitwidth=4):
        super(Bottleneck, self).__init__()
        self.bias11 = LearnableBias(in_planes)
        self.prelu1 = nn.PReLU(in_planes)
        self.bias12 = LearnableBias(in_planes)
        self.quan1 = ActLSQ(nbits = bitwidth)
        self.conv1 = quanconv1x1(in_planes, planes, bitwidth)
        #self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.bias21 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.bias22 = LearnableBias(planes)
        self.quan2 = ActLSQ(nbits = bitwidth)
        self.conv2 = quanconv3x3(planes, planes, bitwidth, stride=stride)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.bias31 = LearnableBias(planes)
        self.prelu3 = nn.PReLU(planes)
        self.bias32 = LearnableBias(planes)
        self.quan3 = ActLSQ(nbits = planes)
        self.conv3 = quanconv1x1(planes, self.expansion * planes, bitwidth)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.bias01 = LearnableBias(planes * self.expansion)
        self.prelu0 = nn.PReLU(planes * self.expansion)
        self.bias02 = LearnableBias(planes * self.expansion)

    def forward(self, x):

        out = self.bias11(x)
        out = self.prelu1(out)
        out = self.bias12(out)
        out = self.quan1(out)
        out = self.conv1(out)
        out = self.bn1(out)

        out = self.bias21(out)
        out = self.prelu2(out)
        out = self.bias22(out)
        out = self.quan2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.bias31(out)
        out = self.prelu3(out)
        out = self.bias32(out)
        out = self.quan3(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += self.downsample(x)
        out = self.bias01(out)
        out = self.prelu0(out)
        out = self.bias02(out)

        return out


class ResNet_32x32(nn.Module):
    def __init__(self, bitwidth, block, num_blocks, channel=3, num_classes=10, record_embedding: bool = False,
                 no_grad: bool = False):
        super().__init__()

        self.in_planes = 64
        self.n_bit = bitwidth

        self.conv1 = conv3x3(channel, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.linear

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.n_bit))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.embedding_recorder(out)
            out = self.linear(out)
        return out


class ResNet_224x224(nn.Module):
    def __init__(self, bitwidth, block, num_blocks, channel: int, num_classes: int, record_embedding: bool = False,
                 no_grad: bool = False):
        super().__init__()
        self.in_planes = 64
        self.n_bit = bitwidth
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc
    
    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.n_bit))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        with set_grad_enabled(not self.no_grad):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.embedding_recorder(x)
            x = self.fc(x)

        return x


def ResNet(bitwidth: int, arch: str, channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
           pretrained: bool = False):
    arch = arch.lower()
    if pretrained:
        if arch == "resnet18":
            net = ResNet_224x224(bitwidth, BasicBlock, [2, 2, 2, 2], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet34":
            net = ResNet_224x224(bitwidth, BasicBlock, [3, 4, 6, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet50":
            net = ResNet_224x224(bitwidth, Bottleneck, [3, 4, 6, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet101":
            net = ResNet_224x224(bitwidth, Bottleneck, [3, 4, 23, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet152":
            net = ResNet_224x224(bitwidth, Bottleneck, [3, 8, 36, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        else:
            raise ValueError("Model architecture not found.")
        from torch.hub import load_state_dict_from_url
        state_dict = load_state_dict_from_url(resnet.model_urls[arch], progress=True)
        net.load_state_dict(state_dict, strict=False)

        if channel != 3:
            net.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if num_classes != 1000:
            net.fc = nn.Linear(net.fc.in_features, num_classes)

    elif im_size[0] == 224 and im_size[1] == 224:
        if arch == "resnet18":
            net = ResNet_224x224(bitwidth, BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet34":
            net = ResNet_224x224(bitwidth, BasicBlock, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet50":
            net = ResNet_224x224(bitwidth, Bottleneck, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet101":
            net = ResNet_224x224(bitwidth, Bottleneck, [3, 4, 23, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet152":
            net = ResNet_224x224(bitwidth, Bottleneck, [3, 8, 36, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        else:
            raise ValueError("Model architecture not found.")
    elif (channel == 1 and im_size[0] == 28 and im_size[1] == 28) or (
            channel == 3 and im_size[0] == 32 and im_size[1] == 32):
        if arch == "resnet18":
            net = ResNet_32x32(bitwidth, BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet34":
            net = ResNet_32x32(bitwidth, BasicBlock, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet50":
            net = ResNet_32x32(bitwidth, Bottleneck, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet101":
            net = ResNet_32x32(bitwidth, Bottleneck, [3, 4, 23, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet152":
            net = ResNet_32x32(bitwidth, Bottleneck, [3, 8, 36, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad)
        else:
            raise ValueError("Model architecture not found.")
    else:
        raise NotImplementedError("Network Architecture for current dataset has not been implemented.")
    return net


def QResNet18(bitwidth: int, channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
             pretrained: bool = False):
    return ResNet(bitwidth, "resnet18", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def QResNet34(bitwidth: int, channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
             pretrained: bool = False):
    return ResNet(bitwidth, "resnet34", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def QResNet50(bitwidth: int, channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
             pretrained: bool = False):
    return ResNet(bitwidth, "resnet50", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def QResNet101(bitwidth: int, channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
              pretrained: bool = False):
    return ResNet(bitwidth, "resnet101", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def QResNet152(bitwidth: int, channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
              pretrained: bool = False):
    return ResNet(bitwidth, "resnet152", channel, num_classes, im_size, record_embedding, no_grad, pretrained)
