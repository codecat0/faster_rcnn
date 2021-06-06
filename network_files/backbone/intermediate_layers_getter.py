"""
@File : intermediate_layers_getter.py
@Author : CodeCat
@Time : 2021/6/3 下午5:41
"""
import torch.nn as nn

from collections import OrderedDict


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        # 遍历模型的子模块按顺序存入有序字典，并且只保存return_layers中最后一层及之前的结构，丢弃之后不用的结构
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 依次遍历模型（丢弃了后面不用的结构了的）的所有的子模块，并进行正向传播，收集return_layers层的输出，并作为FPN网络的输入
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out