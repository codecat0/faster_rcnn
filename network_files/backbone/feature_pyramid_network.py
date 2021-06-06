"""
@File : feature_pyramid_network.py
@Author : CodeCat
@Time : 2021/6/3 下午5:01
"""
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """实现FPN网络"""
    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super(FeaturePyramidNetwork, self).__init__()

        # 用来调整网络结构中最后几个特征图的channel
        self.inner_blocks = nn.ModuleList()

        # 对调整后的特征图使用3x3卷积来得到对应的预测特征图
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            layer_block_module = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
        self.extra_blocks = extra_blocks

    def forward(self, x):
        names = list(x.keys())
        x = list(x.values())

        results = []
        last_inner = self.inner_blocks[-1](x[-1])
        results.append(self.layer_blocks[-1](last_inner))

        for idx in range(len(x)-2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode='nearest')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))

        # 在最后一个特征图对应的预测特征图的基础上生成新的预测特征图
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, names)

        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxpool(nn.Module):
    """在最后一个特征图的基础上作用max_pool2d生成新的预测特征图"""
    def forward(self, x, names):
        names.append('pool')
        x.append(F.max_pool2d(x[-1], 1, 2, 0)) # input, kernel_size, stride, padding
        return x, names