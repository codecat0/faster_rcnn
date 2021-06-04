"""
@File : _utils.py
@Author : CodeCat
@Time : 2021/6/4 上午10:38
"""
import torch
import math


class BalancePositiveNegativeSampler(object):
    """对每一个batch中的所有anchors进行采样，确保一定比例的正样本"""
    def __init__(self, batch_size_per_image, positive_fraction):
        """
        :param batch_size_per_image: 每张图片被选中anchors的个数
        :param positive_fraction: 正样本比例
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idx):
        """
        :param matched_idx: 包含-1，0，正值的tensor；-1代表样本可丢弃、0代表负样本、正值表示正样本
        :return: 正样本索引，负样本索引
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idx:
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(num_pos, positive.numel())

            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(num_neg, negative.numel())

            perm_pos = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm_neg = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm_pos]
            neg_idx_per_image = negative[perm_neg]

            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


class BoxCoder(object):
    """用于将边界框编码和解码用于回归训练"""
    def __init__(self, weights, box_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.box_xform_clip = box_xform_clip

    def encode_single(self, reference_boxes, anchors):
        """
        anchor: [x_a, y_a, w_a, h_a];  ground truth: [x, y, w, h]
        F(anchor) = ground truth 其中 F = (t_x, t_y, t_w, t_h)
        平移：x = w_a * t_x + x_a;  y = h_a * t_y + y_a
        缩放：w = w_a * exp(t_w); h = h_a * exp(t_h)
        t_x = (x - x_a)/w_a
        t_y = (y - y_a)/h_a
        t_w = log(w/w_a)
        t_h = log(h/h_a)
        """
        weights = torch.as_tensor(self.weights, dtype=reference_boxes.dtype, device=reference_boxes.device)
        wx, wy, ww, wh = weights

        anchors_x1 = anchors[:, 0].unsqueeze(1)
        anchors_y1 = anchors[:, 1].unsqueeze(1)
        anchors_x2 = anchors[:, 2].unsqueeze(1)
        anchors_y2 = anchors[:, 3].unsqueeze(1)

        reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
        reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
        reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
        reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

        ac_widths, ac_heights = anchors_x2 - anchors_x1, anchors_y2 - anchors_y1
        ac_center_x, an_center_y = anchors_x1 + 0.5 * ac_widths, anchors_y1 + 0.5 * ac_heights

        gt_widths, gt_heights = reference_boxes_x2 - reference_boxes_x1, reference_boxes_y2 - reference_boxes_y1
        gt_center_x, gt_center_y = reference_boxes_x1 + 0.5 * gt_widths, reference_boxes_y1 + 0.5 * gt_heights

        targets_dx = wx * (gt_center_x - ac_center_x) / ac_widths
        targets_dy = wy * (gt_center_y - an_center_y) / ac_heights
        targets_dw = ww * torch.log(gt_widths / ac_widths)
        targets_dh = wh * torch.log(gt_heights / ac_heights)

        targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def encode(self, reference_boxes, anchors):
        """结合anchors和对应的ground truth box计算回归参数"""

        # 统计一个batch中每张图片中的anchors数，方便后面拼接在一起后还原
        boxes_per_image = [len(b) for b in reference_boxes]

        reference_boxes = torch.cat(reference_boxes, dim=0)
        anchors = torch.cat(anchors, dim=1)

        targets = self.encode_single(reference_boxes, anchors)
        return targets.split(boxes_per_image, 0)

    def decode_single(self, reg_codes, anchors):
        """
        anchor: [x_a, y_a, w_a, h_a];  ground truth: [x, y, w, h]
        F(anchor) = ground truth 其中 F = (t_x, t_y, t_w, t_h)
        平移：x = w_a * t_x + x_a;  y = h_a * t_y + y_a
        缩放：w = w_a * exp(t_w); h = h_a * exp(t_h)
        """
        anchors = anchors.to(reg_codes.dtype)

        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        center_x = anchors[:, 0] + 0.5 * widths
        center_y = anchors[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = reg_codes[:, 0::4] / wx
        dy = reg_codes[:, 1::4] / wy
        dw = reg_codes[:, 2::4] / ww
        dh = reg_codes[:, 3::4] / wh

        # 防止exp过大
        dw = torch.clamp(dw, max=self.box_xform_clip)
        dh = torch.clamp(dh, max=self.box_xform_clip)

        pred_center_x = widths[:, None] * dx + center_x[:, None]
        pred_center_y = heights[:, None] * dy + center_y[:, None]
        pred_w = widths[:, None] * torch.exp(dw)
        pred_h = heights[:, None] * torch.exp(dh)

        pred_boxes_x1 = pred_center_x - torch.tensor(0.5, dtype=pred_center_x.dtype, device=pred_w.device) * pred_w
        pred_boxes_y1 = pred_center_y - torch.tensor(0.5, dtype=pred_center_y.dtype, device=pred_h.device) * pred_h
        pred_boxes_x2 = pred_center_x + torch.tensor(0.5, dtype=pred_center_x.dtype, device=pred_w.device) * pred_w
        pred_boxes_y2 = pred_center_y + torch.tensor(0.5, dtype=pred_center_y.dtype, device=pred_h.device) * pred_h

        pred_boxes = torch.stack((pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2), dim=2).flatten(1)
        return pred_boxes

    def decode(self, reg_codes, anchors):
        """根据回归参数和anchors得到预测box"""
        anchors_per_image = [len(a) for a in anchors]
        concat_anchors = torch.cat(anchors, dim=0)

        anchor_sum = 0
        for val in anchors_per_image:
            anchor_sum += val

        pred_boxes = self.decode_single(reg_codes, concat_anchors)

        if anchor_sum > 0:
            pred_boxes = pred_boxes.reshape(anchor_sum, -1, 4)

        return pred_boxes


class Matcher(object):
    """为每一个anchor匹配与之对应的gt box"""
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_mathes = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        计算anchors与每个gtboxes匹配的IOU最大值，并记录索引
        iou<low_threshold索引值为-1；low_threshold<=iou<=high_threshold索引值为-2，iou>high_threshold的索引值不做任何处理
        """

        # match_quality_matrix is M(gt) x N(predicted)
        # M x N 的每一列代表一个anchor与所有gt匹配的iou值，每一行代表一个gt与所有anchor匹配的iou值
        matched_vals, mathes = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_mathes:
            all_mathes = mathes.clone()
        else:
            all_mathes = None

        # 获取iou小于low_threshold和iou处于low_threshold和high_threshold之间的索引值
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) and (matched_vals <= self.high_threshold)

        # iou小于low_threshold的mathes索引置为-1
        mathes[below_low_threshold] = self.BELOW_LOW_THRESHOLD

        # iou处于low_threshold和high_threshold之间的mathes索引置为-2
        mathes[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_mathes:
            self.set_low_quality_mathes_(mathes, all_mathes, match_quality_matrix)

    @staticmethod
    def set_low_quality_mathes_(mathes, all_mathes, match_quality_matrix):
        """将每个gtbox匹配的最大iou对应的anchor添加进来"""
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_paris_of_highest_quality = torch.where(torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None]))
        pre_inds_to_update = gt_pred_paris_of_highest_quality[:, 1]
        mathes[pre_inds_to_update] = all_mathes[pre_inds_to_update]
