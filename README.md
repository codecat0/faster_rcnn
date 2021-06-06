## Faster-RCNN

### 项目代码参考pytorch官方源码
- https://github.com/pytorch/vision/tree/master/torchvision/models/detection

### 环境配置
```
torch
torchvision
Pillow
numpy 
matplotlib
lxml
Pillow
json
```

### 项目结构
```
|--dataset_preprocess: 自定义DataSet
|--network_files
|      |--transform：数据预处理
|      |--backbone：特征提取
|      |--rpn：用于生成region proposals
|      |--roi_head: 利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置
|--train_resnet50_fpn.py: 训练脚本
|--pascal_voc_classes.json: 数据集类别信息
```

### 数据集：PASCAL VOC2012数据集
- 下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

### 权重信息：
- ResNet50+FPN: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

### 训练方法：
1. 下载`PASCAL VOC2012`数据集，并将其放在项目根目录下
2. 下载`backbone ResNet50+FPN`权重，路径与`train_resnet50_fpn.py`中载入模型参数位置保持一致
3. 运行`train_resnet50_fpn.py`文件