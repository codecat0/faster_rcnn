"""
@File : transfrom.py
@Author : CodeCat
@Time : 2021/6/3 下午3:13
"""
import math
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .image_list import ImageList


class GeneralizedRCNNTransfrom(nn.Module):
    """
    1. 输入图像的标准化
    2. 改变输入图像和标签的尺寸，使其尺寸满足在给定的最小尺寸和最大尺寸之间
    """
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransfrom, self).__init__()
        if not isinstance(min_size, (tuple, list)):
            min_size = (min_size,)
        self.min_size = min_size   # 指定图像的最小尺寸范围
        self.max_size = max_size   # 指定图像的最大尺寸范围
        self.image_mean = image_mean   # 指定图像在标准化处理的均值
        self.image_std = image_std     # 指定图像在标准化处理的方差

    def forward(self, images, targets):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors of shape [C, H, W], got {}".format(image.shape))

            # 对图像进行标准化处理
            image = self.normalize(image)

            # 将图像和对应的boxes缩放到指定范围
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # 记录resize后图像的尺寸，方便后面预测框的可视化
        image_sizes = [img.shape[-2:] for img in images]

        # 将images打包成一个batch
        images = self.batch_images(images)

        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])

        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def normalize(self, image):
        """标准化处理"""
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    @staticmethod
    def torch_chioce(k):
        """在torch操作上实现random.choice"""
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    @staticmethod
    def resize_image(image, self_min_size, self_max_size):
        """缩放图像至指定范围内"""
        im_shape = torch.tensor(image.shape[-2:])

        # 获取图像高宽的最小值和最大值
        min_size = float(torch.min(im_shape))
        max_size = float(torch.max(im_shape))

        # 根据指定的最小值和最大值计算缩放比例
        scale_factor = min(self_min_size/min_size, self_max_size/max_size)

        # 利用双线性插值的方法缩放图片，bilinear只支持4d tensor, image[None]操作可以将[C, H, W] -> [1, C, H, W]
        image = F.interpolate(image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

        return image

    @staticmethod
    def resize_boxes(boxes, original_size, new_size):
        """将boxes参数根据图像的缩放情况进行相应缩放"""
        ratios = [torch.tensor(s, dtype=torch.float32, device=boxes.device) /
                  torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
                  for s, s_orig in zip(new_size, original_size)]

        ratios_height, ratios_width = ratios
        xmin = boxes[:, 0] * ratios_width
        ymin = boxes[:, 1] * ratios_height
        xmax = boxes[:, 2] * ratios_width
        ymax = boxes[:, 3] * ratios_height

        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

    def resize(self, image, target):
        """将图片缩放到指定的大小范围内，并对应缩放boxes信息"""
        h, w = image.shape[-2:]

        if self.training:
            size = float(self.torch_chioce(self.min_size))
        else:
            size = float(self.min_size[-1])

        image = self.resize_image(image, size, float(self.max_size))

        if target is None:
            return image, target

        bbox = target['boxes']
        bbox = self.resize_boxes(bbox, [h, w], image.shape[-2:])
        target['boxes'] = bbox

        return image, target

    @staticmethod
    def max_by_axis(the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        """将一批图像打包成一个batch返回"""

        # 分别计算一个batch中所有图像中最大的channel, height, width
        max_size = self.max_by_axis([list(img.shape) for img in images])

        stride = float(size_divisible)

        # 将图像的高度，宽度调整到32的整数倍，最后特征图会缩小32倍，为了保证特征图是整数
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # [batch_size, channel, height, width]
        batch_shape = [len(images)] + max_size

        # 创建全为0且形状为batch_shape的tensor
        batched_imgs = torch.zeros(size=batch_shape, dtype=images[0].dtype, device=images[0].device)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        """对预测结果进行后处理，将boxes还原到原图像尺度上"""
        if self.training:
            return result

        # 遍历每张图像的预测信息，将boxes信息还原回原尺度
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred['boxes']
            boxes = self.resize_boxes(boxes, im_s, o_im_s)
            result[i]['boxes'] = boxes
        return result