## Faster-RCNN

### 项目代码参考pytorch官方源码
- https://github.com/pytorch/vision/tree/master/torchvision/models/detection

### 环境配置
```
torch
torchvision
pycocotools
Pillow
numpy matplotlib
lxml
Pillow
tqdm
```

### 项目结构
```
|--dataset_preprocess: 自定义DataSet
|--network_files
|      |--transform：数据预处理
|      |--backbone：特征提取
|      |--rpn：用于生成region proposals
|      |--roi_head: 利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置
```

### 数据集：PASCAL VOC2012数据集
- 下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
