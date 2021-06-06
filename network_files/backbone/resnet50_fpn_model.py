"""
@File : resnet50_fpn_model.py
@Author : CodeCat
@Time : 2021/6/3 下午5:58
"""
import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d

from .feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxpool
from .intermediate_layers_getter import IntermediateLayerGetter


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride, norm_layer=norm_layer))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, extra_blocks=None):
        super(BackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxpool()

        self.body = IntermediateLayerGetter(backbone, return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )

        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def resnet50_fpn_backbone(pretrain_path='',
                          norm_layer=FrozenBatchNorm2d,
                          trainable_layers=3,
                          returned_layers=None,
                          extra_blocks=None):
    resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3], include_top=False, norm_layer=norm_layer)
    if isinstance(norm_layer, FrozenBatchNorm2d):
        for module in resnet_backbone.modules():
            if isinstance(module, FrozenBatchNorm2d):
                module.eps = 0.0

    if pretrain_path != '':
        resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False)

    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # 如果要训练所有层结构，conv1后面还有一个bn1
    if len(layers_to_train) == 5:
        layers_to_train.append('bn1')

    for name, parameter in resnet_backbone.named_parameters():
        # 只训练layers_to_train中的层结构
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxpool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]

    # return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    # in_channel为layer4的输出特征图的channel=2048
    in_channels_stage2 = resnet_backbone.in_channel // 8

    in_channels_list = [in_channels_stage2 * 2 ** (i-1) for i in returned_layers]

    # 通过FPN后得到的每个特征图的channel
    out_channels = 256

    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels, extra_blocks)