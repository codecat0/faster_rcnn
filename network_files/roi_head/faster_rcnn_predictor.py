"""
@File : faster_rcnn_predictor.py
@Author : CodeCat
@Time : 2021/6/5 下午3:10
"""
import torch.nn as nn


class FasterRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FasterRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas