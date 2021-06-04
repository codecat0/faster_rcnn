"""
@File : boxes.py
@Author : CodeCat
@Time : 2021/6/4 上午9:24
"""
import torch
import torchvision


def nms(boxes, scores, iou_threshold):
    """对每个box根据IOUshi实行NMS，NMS移除低得分的box和与高得分box的IOU超过iou_threshold的box"""
    return torchvision.ops.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    """对一个batch中的所有boxes进行NMS"""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # 获取所有boxes中的最大的坐标值
    max_coordinate = boxes.max()

    offsets = idxs.to(boxes) * (max_coordinate + 1)

    # boxes加上对应层或类别的偏移量后，保证了不同类别/层之间的box不会有重合的现象
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def remove_small_boxes(boxes, min_size):
    """删除宽高小于指定阈值的索引"""
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]

    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))
    keep = torch.where(keep)[0]
    return keep


def clip_boxes_to_image(boxes, size):
    """裁剪预测的boxes信息，将越界的坐标调整到图片的边界上"""
    dim = boxes.dim()
    boxes_x = boxes[:, 0::2]
    boxes_y = boxes[:, 1::2]
    height, width = size

    # 将x，y坐标范围限制在图像size之中
    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def box_area(boxes):
    """计算box的面积"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """计算box之间的IOU值"""
    area1, area2 = box_area(boxes1), box_area(boxes2)

    # 获取两个box相交的左上角坐标和右下角坐标
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou