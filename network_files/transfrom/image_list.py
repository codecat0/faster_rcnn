"""
@File : image_list.py
@Author : CodeCat
@Time : 2021/6/3 下午3:14
"""


class ImageList(object):
    def __init__(self, tensors, image_sizes):
        """
        :param tensors: padding后的图像数据
        :param image_sizes: padding前的图像尺寸
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        cast_tensors = self.tensors.to(device)
        return ImageList(cast_tensors, self.image_sizes)