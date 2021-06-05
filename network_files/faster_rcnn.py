"""
@File : faster_rcnn.py
@Author : CodeCat
@Time : 2021/6/5 下午3:15
"""
from collections import OrderedDict
from typing import Tuple, List

import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign

from .roi_head.roi_head import RoIHead
from .roi_head.two_mlp_head import TwoMLPHead
from .roi_head.faster_rcnn_predictor import FasterRCNNPredictor

from .transfrom.transfrom import GeneralizedRCNNTransfrom

from .rpn.anchor_utils import AnchorGenerator
from .rpn.rpn import RPNHead, RegionProposalNetwork


class FasterRCNNBase(nn.Module):
    def __init__(self, transform, backbone, rpn, roi_heads):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None):
        # 存储图像的原始尺寸，方便以后将预测的box信息还原
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # 对图像进行预处理
        images, targets = self.transform(images, targets)

        # 获取图像的特征图
        features = self.backbone(images.tensors)

        # 若只在一层特征图上预测，将feature放入有序字典中，并编号为0
        # 若在多层特征图上预测，backbone输出的就是一个有序字典
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # 将特征图以及target标注信息传入rpn，得到proposals
        # 每个proposals为绝对坐标，且为(x1, y1, x2, y2)格式
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 将rpn生成的数据以及target标注信息传入fast_rcnn后半部分
        detections, detector_losses = self.roi_heads(features, proposals, images.image_size, targets)

        # 对网格的预测结果进行后处理，主要是将预测的box信息还原到原始图像上
        detections = self.transform.postprocess(detections, images.image_size, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections


class FasterRCNN(FasterRCNNBase):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_derecrion_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):
        # 预测特征图的channels
        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )

        # 生成RPN滑动窗口预测网络部分
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = {
            'training': rpn_pre_nms_top_n_train,
            'testing': rpn_pre_nms_top_n_test
        }

        rpn_post_nms_top_n = {
            'training': rpn_post_nms_top_n_train,
            'testing': rpn_post_nms_top_n_test
        }

        # 定义RPN框架
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh
        )

        # Multi-scale RoIAlign pooling
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=[7, 7],
                sampling_ratio=2
            )

        # roi pooling后展平处理两个全连接层部分
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )

        # box_head的预测部分
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FasterRCNNPredictor(
                representation_size,
                num_classes
            )

        roi_heads = RoIHead(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_derecrion_per_img
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]

        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        transform = GeneralizedRCNNTransfrom(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(transform, backbone, rpn, roi_heads)
