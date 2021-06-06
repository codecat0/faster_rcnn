"""
@File : anchor_utils.py
@Author : CodeCat
@Time : 2021/6/4 下午2:44
"""
import torch
from torch import nn

from typing import List


class AnchorGenerator(nn.Module):
    """anchor生成器"""
    def __init__(self, sizes=((128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)):
        super(AnchorGenerator, self).__init__()
        if not isinstance(sizes, (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None

    def forward(self, image_list, feature_maps):
        # 获取每个预测特征层的尺寸
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])

        # 获取输入图像的尺寸
        image_size = image_list.tensors.shape[-2:]

        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # 计算每个特征层到原始图像的步长
        strides = [[torch.tensor(image_size[0] // g[0], dtype=dtype, device=device),
                   torch.tensor(image_size[1] // g[1], dtype=dtype, device=device)] for g in grid_sizes]

        # 根据提供的sizes和aspect_ratios生成anchors模板
        self.set_cell_anchors(dtype, device)

        # 计算预测特征图对应原始图上的所有anchors坐标
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)

        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])

        # 遍历batch中的每一张图片
        for _ in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # 遍历每层特征图映射回原图的anchor坐标信息
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)

        # 将每张图像的所有预测特征图的anchor坐标信息拼接在一起
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors

    @staticmethod
    def generate_anchors(scales, aspect_ratios, dtype=torch.float32, device=torch.device('cpu')):
        """生成anchors模板"""
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        # 生成的anchors模板都是以（0，0）为中心的
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

        return base_anchors.round()

    def set_cell_anchors(self, dtype, device):
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            if cell_anchors[0].device == device:
                return

        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]

        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        """计算每一个预测特征图上每个滑动窗口的预测目标数"""
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes, strides):
        """计算预测特征图对应原始图上的所有anchors坐标"""
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        # 遍历每一个预测特征图
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # 生成对应于原图的x坐标
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width

            # 生成对应于原图的y坐标
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            # 获取预测特征图上的每个点对应原图上的坐标
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)

            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors