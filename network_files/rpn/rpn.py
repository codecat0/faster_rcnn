"""
@File : rpn.py
@Author : CodeCat
@Time : 2021/6/4 下午3:50
"""
import torch
from torch import nn
from torch.nn import functional as F

from . import _utils as det_utils
from . import boxes as box_ops
from .anchor_utils import AnchorGenerator

from typing import Dict


class RPNHead(nn.Module):
    """通过3x3的滑动窗口来计算预测目标概率分数和回归参数"""
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

        # 计算预测的目标概率分数（这里的目标只是前景或者背景）
        self.cls_logits = nn.Conv2d(in_channels=in_channels, out_channels=num_anchors, kernel_size=1, stride=1)

        # 计算预测的目标回归参数
        self.bbox_pred = nn.Conv2d(in_channels=in_channels, out_channels=num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    """
    调整tensor维度的排列顺序，并进行reshape
    :param layer: 预测特征图上预测的目标概率或box回归参数
    :param N: batch_size
    :param A: anchors_num_per_position
    :param C: classes_num or 4
    :param H: height
    :param W: width
    :return: 调整tensor形状，将其reshape为[N, -1, C]
    """
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layer(box_cls, box_regression):
    """将每个预测层的预测信息的tensor的排列顺序和shape进行调整为[N, -1, C]"""
    box_cls_flattened = []
    box_regression_flattened = []
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # [batch_size, anchors_num_per_position * classes_num, height, width]
        N, AxC, H, W = box_cls_per_level.shape

        # [batch_size, anchors_num_per_position * 4, height, width]
        Ax4 = box_regression_per_level.shape[1]

        # anchors_num_per_position
        A = Ax4 // 4

        # classes_num
        C = AxC // A

        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    # torch.cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RegionProposalNetwork(nn.Module):
    """
    实现RPN网络
    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str, int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str, int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancePositiveNegativeSampler,
    }

    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thrsh, score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = box_ops.box_iou

        self.proposal_mather = det_utils.Matcher(
            high_threshold=fg_iou_thresh,
            low_threshold=bg_iou_thresh,
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = det_utils.BalancePositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thrsh
        self.score_thresh = score_thresh
        self.min_size = 1

    def forward(self, images, features, targets):
        # fearures是所有预测特征层组成的Orderedict
        features = list(features.values())

        # 计算每一个特征层上的预测目标概率和box回归参数
        objectness, pred_bbox_deltas = self.head(features)

        # 生成一个batch中所有图像的所有anchors信息
        anchors = self.anchor_generator(images, features)

        # batch_size
        num_images = len(anchors)

        # 计算每个预测特征图对应的anchors数量
        per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in per_level_shape_tensors]

        # 调整预测目标概率和box回归参数的形状
        objectness, pred_bbox_deltas = concat_box_prediction_layer(objectness, pred_bbox_deltas)

        # 将预测的回归参数作用到anchors上得到最终预测box坐标
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        # 删除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        loss = {}
        if self.training:
            assert targets is not None
            # 获取每个anchor最匹配的gt，并将anchor进行分类
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)

            # 结合anchors和对应的gt box，计算回归参数
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)

            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness,
                pred_bbox_deltas,
                labels,
                regression_targets
            )

            losses = {
                'loss_objectness': loss_objectness,
                'loss_rpn_box_reg': loss_rpn_box_reg
            }

            return boxes, losses

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        """获取每张预测特征图上预测概率排前pre_nms_top_n的anchors的索引值"""
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        num_images = proposals.shape[0]
        device = proposals.device

        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        # levels负责记录不同预测特征层上的anchor信息
        # 第一层用0标记，第二层用1标记，依次类推
        levels = [torch.full((n, ), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # 获取每张预测特征图上预测概率排前pre_nms_top_n的anchors的索引值
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        # 遍历每张图像
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # 将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # 删除小预测框
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 删除小概率预测框
            keep = torch.where(torch.ge(scores, self.score_thresh))[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # NMS
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def assign_targets_to_anchors(self, anchors, targets):
        """计算每个anchor最匹配的gt box，并划分正样本，负样本以及丢弃的样本"""
        labels = []
        matched_gt_boxes = []

        # 遍历每张图像的anchors和targets
        for anchors_per_image, targets_per_images in zip(anchors, targets):
            gt_boxes = targets_per_images['boxes']
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0], ), dtype=torch.float32, device=device)
            else:
                # 计算anchors与gt boxes的iou值
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)

                # 计算每个anchor与gt box匹配iou最大的索引 （iou<low_threshold索引置为-1，low_threshold<=iou<=high_threshold索引置为-2）
                matched_idxs = self.proposal_mather(match_quality_matrix)

                # 为了防止索引越界，将其下限设置为0
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                # 记录所有anchors匹配后的标签
                # 正样本：1
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # 负样本：0
                bg_indices = matched_idxs == self.proposal_mather.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # 丢弃的样本：-1
                discard_indices = matched_idxs == self.proposal_mather.BETWEEN_THRESHOLDS
                labels_per_image[discard_indices] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        """计算RPN损失，包括类别损失（前景，背景）、box回归损失"""
        # 按照给定的batch_size_per_image，positive_fraction选择正负样本
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        # 将一个batch中所有正/负样本分别拼接在一起, 并获取标记为正/负样本的索引
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        # 将所有正负样本索引拼接在一起
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # 计算边界框回归损失
        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        # 计算目标预测概率损失
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds],
            labels[sampled_inds]
        )

        return objectness_loss, box_loss