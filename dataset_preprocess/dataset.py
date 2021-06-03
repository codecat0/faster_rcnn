"""
@File : dataset.py
@Author : CodeCat
@Time : 2021/6/3 上午11:47
"""
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


class VOC2012DataSet(Dataset):
    def __init__(self, voc_root, transfroms, txt_name):
        self.root = os.path.join(voc_root, 'VOCdevkit', 'VOC2012')
        self.img_root = os.path.join(self.root, 'JPEGImages')
        self.annotations_root = os.path.join(self.root, 'Annotations')

        txt_path = os.path.join(self.root, 'ImageSets', 'Main', txt_name)

        with open(txt_path) as f:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + '.xml') for line in f.readlines()]

        json_file = './pascal_voc_classes.json'
        json_file = open(json_file, 'r')
        self.class_dict = json.load(json_file)
        json_file.close()

        self.transfroms = transfroms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation']
        img_path = os.path.join(self.img_root, data['filename'])
        image = Image.open(img_path)
        if image.format != 'JPEG':
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        for obj in data['object']:
            bndbox = obj['bndbox']
            xmin, xmax = float(bndbox['xmin']), float(bndbox['xmax'])
            ymin, ymax = float(bndbox['ymin']), float(bndbox['ymax'])

            if xmax <= xmin or ymax <= ymin:
                raise ValueError("there is a box w/h <= 0 in '{}' xml".format(xml_path))

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj['name']])
            if 'diffcuilt' in obj:
                iscrowd.append(int(obj['difficult']))
            else:
                iscrowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd': iscrowd}

        if self.transfroms is not None:
            image, target = self.transfroms(image, target)

        return image, target

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def get_height_width(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation']
        data_height = int(data['size']['height'])
        data_width = int(data['size']['width'])
        return data_height, data_width

    def coco_index(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation']
        data_height = int(data['size']['height'])                    
        data_width = int(data['size']['width'])

        boxes = []
        labels = []
        iscrowd = []
        for obj in data['object']:
            bndbox = obj['bndbox']
            xmin, xmax = float(bndbox['xmin']), float(bndbox['xmax'])
            ymin, ymax = float(bndbox['ymin']), float(bndbox['ymax'])

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj['name']])
            if 'diffcuilt' in obj:
                iscrowd.append(int(obj['difficult']))
            else:
                iscrowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd': iscrowd}

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))