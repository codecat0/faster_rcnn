"""
@File : train_resnet50_fpn.py
@Author : CodeCat
@Time : 2021/6/5 下午4:28
"""
import os
import datetime
import argparse
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from .dataset_preprocess import transforms
from .dataset_preprocess.dataset import VOC2012DataSet

from .network_files.faster_rcnn import FasterRCNN
from .network_files.roi_head.faster_rcnn_predictor import FasterRCNNPredictor
from .network_files.backbone.resnet50_fpn_model import resnet50_fpn_backbone


def create_model(num_classes, device):
    backbone = resnet50_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    weights_dict = torch.load("./network_files/backbone/fasterrcnn_resnet50_fpn_coco.pth", map_location=device)
    _, _ = model.load_state_dict(weights_dict, strict=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FasterRCNNPredictor(in_features, num_classes)

    return model


def plot_loss_and_lr(train_loss, learning_rate):
    x = list(range(len(train_loss)))
    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(x, train_loss, 'red', label='loss')
    ax1.set_xlabel('step')
    ax1.set_ylabel('loss')
    ax1.set_title('Train Loss and Lr')
    plt.legend(loc='best')

    ax2 = ax1.twinx()
    ax2.plot(x, learning_rate, 'blue', label='lr')
    ax2.set_ylabel('learing rate')
    ax2.set_xlim(0, len(train_loss))
    plt.legend(loc='best')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    plt.subplots_adjust(right=0.8)
    fig.savefig('./loss_and_lr{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    plt.close()


def main(parser_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transform = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
        ]),
        'val': transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = parser_data.data_path
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_data_set = VOC2012DataSet(VOC_root, data_transform['train'], 'train.txt')
    train_data_loader = DataLoader(
        dataset=train_data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        collate_fn=train_data_set.collate_fn
    )

    val_data_set = VOC2012DataSet(VOC_root, data_transform['val'], 'val.txt')
    val_data_loader = DataLoader(
        dataset=val_data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        collate_fn=val_data_set.collate_fn
    )

    model = create_model(num_classes=parser_data.num_classes+1, device=device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if parser_data.resume != '':
        checkpoint = torch.load(parser_data.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        print('the training process from epoch{}'.format(parser_data.start_epoch))

    train_loss = []
    learing_rate = []

    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        model.train()
        mloss = torch.zeros(1).to(device)
        now_lr = torch.zeros(1).to(device)
        for i, [images, targets] in enumerate(train_data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            mloss = (mloss * i + losses.item()) / (i + 1)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            now_lr = optimizer.param_groups[0]['lr']

        train_loss.append(mloss.item())
        learing_rate.append(now_lr.item())

        # 更新学习率
        lr_scheduler.step()

        # 保存权重信息
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch
        }
        torch.save(save_files, os.path.join(parser_data.output_dir, 'resNetFPN-model-{}.pth'.format(epoch)))

    if len(train_loss) != 0 and len(learing_rate) != 0:
        plot_loss_and_lr(train_loss, learing_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data_path', default='./', help='dataset')
    parser.add_argument('--num_classes', default=20, type=int, help='num_classes')
    parser.add_argument('--output_dir', default='./save_weights', help='path of weights to save')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=15, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size when training')

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)