"""
@File : roi_head.py
@Author : CodeCat
@Time : 2021/6/5 上午10:30
"""
import torch

from torch import nn
import torch.nn.functional as F

from ..rpn import _utils as det_utils
from ..rpn import boxes as box_ops

from typing import List, Dict


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """计算Faster-RCNN损失"""
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # 计算类别损失
    classification_loss = F.cross_entropy(class_logits, labels)

    # 获取正样本的索引
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]

    # 获取正样本的标签信息
    labels_pos = labels[sampled_pos_inds_subset]

    # 获取分类类别数
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    # 计算边界框的损失
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    ) / labels.numel()

    return classification_loss, box_loss


class RoIHead(nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancePositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,      # Multi-scale RoIAlign pooling
                 box_head,          # TwoMPLHead
                 box_predictor,     # FasterRCNNPredictor
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 score_thresh,
                 nms_thresh,
                 detection_per_image):
        super(RoIHead, self).__init__()

        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,      # default 0.5
            bg_iou_thresh,      # default 0.5
            allow_low_quality_matches=False
        )

        self.fg_bg_sampler = det_utils.BalancePositiveNegativeSampler(
            batch_size_per_image,       # default 512
            positive_fraction           # default 0.25
        )

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)

        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh    # default 0.05
        self.nms_thresh = nms_thresh        # default 0.5
        self.detection_per_image = detection_per_image      # default 100

    def forward(self, features, proposals, image_shapes, targets):
        if self.training:
            # 划分正负样本，统计对应gt的标签以及边界框回归信息
            proposals, labels, regression_targets = self.select_trainig_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        # 将采集的样本通过Multi-scale RoIAlign pooling层
        box_feature = self.box_roi_pool(features, proposals, image_shapes)

        # 通过roi_pooling后的两层全连接层
        box_feature = self.box_head(box_feature)

        class_logits, box_regression = self.box_predictor(box_feature)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {
                'loss_classifier': loss_classifier,
                'loss_box_reg': loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        'boxes': boxes[i],
                        'labels': labels[i],
                        'scores': scores[i],
                    }
                )
        return result, losses

    def select_trainig_samples(self, proposals, targets):
        # 检查targets数据是否为空，是否包含boxes和labels
        self.check_targets(targets)

        dtype, device = proposals[0].dtype, proposals[0].device

        # 获取gt boxes和对应的标签信息
        gt_boxes = [t['boxes'].to(dtype) for t in targets]
        gt_labels = [t['labels'] for t in targets]

        # 将gt box添加到proposals后面
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # 为每一个proposals匹配对应的gt box，并划分到正负样本中
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        # 按照给定数量和比例采样正负样本
        sampled_inds = self.subsample(labels)

        matched_gt_boxes = []
        num_images = len(proposals)

        # 遍历每张图像
        for img_id in range(num_images):
            # 获取每张图像的样本索引
            img_sampled_inds = sampled_inds[img_id]

            # 获取样本的proposals信息
            proposals[img_id] = proposals[img_id][img_sampled_inds]

            # 获取样本的真实类别信息
            labels[img_id] = labels[img_id][img_sampled_inds]

            # 获取样本的gt box索引信息
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[img_id])

        # 根据gt box和proposals计算回归参数
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, labels, regression_targets

    @staticmethod
    def check_targets(targets):
        assert targets is not None
        assert all(['boxes' in t for t in targets])
        assert all(['labels' in t for t in targets])

    @staticmethod
    def add_gt_proposals(proposals, gt_boxes):
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched_idx = []
        labels = []

        # 遍历每张图像
        for proposals_per_image, gt_boxes_per_image, gt_labels_per_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_per_image.numel() == 0:
                device = proposals_per_image.device
                clamped_matched_idxs_per_image = torch.zeros(
                    (proposals_per_image.shape[0], ), dtype=torch.int64, device=device
                )

                labels_per_image = torch.zeros(
                    (proposals_per_image.shape[0], ), dtype=torch.int64, device=device
                )
            else:
                # 计算proposals与gt boxes的iou值
                match_quality_matrix = box_ops.box_iou(gt_boxes_per_image, proposals_per_image)

                # 计算proposals与每个gt box匹配的iou的最大值，并记录索引
                # iou<low_threshold索引值为-1，low_threshold<=iou<=high_threshold索引值为-2，iou>high_threshold的索引值不做操作
                matched_idxs_per_image = self.proposal_matcher(match_quality_matrix)

                # 防止索引越界，将-1，-2调整为0（后续后进一步处理，-1，-2标签不为第0个gt box的类别）
                clamped_matched_idxs_per_image = matched_idxs_per_image.clamp(min=0)

                # 获取proposals匹配的gt box对应标签
                labels_per_image = gt_labels_per_image[clamped_matched_idxs_per_image]
                labels_per_image = labels_per_image.to(dtype=torch.int64)

                # 将gt box索引为-1的类别设置为0， 负样本
                bg_inds = matched_idxs_per_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_inds] = 0

                # 将gt box索引为-2的类别设置为-1，丢弃样本
                discard_inds = matched_idxs_per_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[discard_inds] = -1

            matched_idx.append(clamped_matched_idxs_per_image)
            labels.append(labels_per_image)
        return matched_idx, labels

    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        # 遍历每张图片
        for img_idx, (pos_inds_per_image, neg_inds_per_image) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_per_image | neg_inds_per_image)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def postprocess_detections(self,
                               class_logits,
                               box_regression,
                               proposals,
                               image_shapes):
        """
        对网络的预测数据进行后处理：
            1. 根据proposal以及预测的回归参数计算最终box坐标
            2. 对预测类别结果进行softmax处理
            3. 裁剪预测的box信息，将越界的坐标调整到图片边界上
            4. 移除所有背景信息
            5. 移除低概率目标
            6. 移除小尺寸目标
            7. 执行nms处理，并按score排序
            8. 根据score排序结果返回前topk个目标
        :param class_logits: 网络预测类别概率信息
        :param box_regression: 网络预测边界框回归参数
        :param proposals: rpn输出的proposals
        :param image_shapes: 打包成batch前每张图像的尺寸
        """
        device = class_logits.device
        # 预测类别数
        num_classes = class_logits.shape[-1]

        # 获取每张图像的预测box数量
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        # 根据proposals以及预测回归参数计算最终box坐标
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # 对预测类别结果进行softmax处理
        pred_scores = F.softmax(class_logits, -1)

        # 根据每张图像的box数量分割结果
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        # 遍历每张图像预测信息
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # 裁剪预测的box信息，防止其越界
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # 为每个预测结果创建标签
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # 移除索引为0的信息，索引为0代表背景
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = scores[:, 1:]

            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # 移除低概率目标
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # 移除小尺寸目标
            keep = box_ops.remove_small_boxes(boxes, min_size=1.)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # NMS处理, 并获取前topk个预测目标
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[:self.detection_per_image]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_labels.append(labels)
            all_scores.append(scores)

        return all_boxes, all_scores, all_labels