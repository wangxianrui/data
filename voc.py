import os
import torch
import torch.utils.data
import torchvision
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from .data_aug import AugSequence


class VOCAnnotationTransform(object):
    """
    transform cocoannotations with [xmin, ymin, width, height] to [xmin, ymin, xmax, ymax]
    transform labelindex with [0--19]
    :return np.array
        [[xmin, ymin, xmax, ymax, label], [...], ,, ]
    """

    def __init__(self):
        self.classes_name = ('aeroplane', 'bicycle', 'bird', 'boat',
                             'bottle', 'bus', 'car', 'cat', 'chair',
                             'cow', 'diningtable', 'dog', 'horse',
                             'motorbike', 'person', 'pottedplant',
                             'sheep', 'sofa', 'train', 'tvmonitor')

    def __call__(self, target, width, height):
        scale = np.array([width, height, width, height])
        res = []
        for obj in target.iter('object'):
            bndbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bbox = []
            for i, pt in enumerate(pts):
                bbox.append(int(bndbox.find(pt).text))
            name = obj.find('name').text.lower().strip()
            label = self.classes_name.index(name)
            final_box = list(np.array(bbox) / scale)
            final_box.append(label)
            res.append(final_box)
        if len(res) == 0:
            res.append([0.0, 0, 0, 0, 0])
        return np.array(res).astype(np.float32)


class VOCDetection(torch.utils.data.Dataset):
    def __init__(self, data_root, imgfiles, input_size, is_training):
        self.data_root = data_root
        self.imgfiles = imgfiles
        self.input_size = input_size
        self.is_training = is_training
        self.augmentation = AugSequence()
        self.target_transform = VOCAnnotationTransform()
        self.img_paths = []
        for imgfile in imgfiles:
            year = '2007' if '2007' in imgfile else '2012'
            for line in open(os.path.join(data_root, imgfile)):
                self.img_paths.append(os.path.join(data_root, 'VOC' + year, 'JPEGImages', line.strip() + '.jpg'))

    def __getitem__(self, index):
        '''
            'img_id': img_id,
            'height': origion image height,
            'width': origion image width,
            'img': tensor.size(3,input_size)
            'target': np.array with [[xmin, ymin, xmax, ymax, label],...]
        '''
        img_path = self.img_paths[index]
        ann_path = img_path.replace('JPEGImages', 'Annotations').replace('jpg', 'xml')
        # img
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        img = cv2.resize(img, self.input_size)
        # target
        voc_ann = ET.parse(ann_path).getroot()
        target = self.target_transform(voc_ann, width, height)
        # augmentation
        if self.is_training and target.any():
            img, target = self.augmentation(img, target)

        # # augmentation test
        # pt1 = (int(target[0, 0] * 512), int(target[0, 1] * 512))
        # pt2 = (int(target[0, 2] * 512), int(target[0, 3] * 512))
        # img = img.astype(np.uint8)
        # print(pt1, pt2)
        # cv2.rectangle(img, pt1, pt2, (255, 0, 0))
        # cv2.imshow('win', img)
        # cv2.waitKey()
        # #
        # normalize img

        img = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(img)
        return {
            'img_id': int(os.path.splitext(img_path)[0][-6:]),
            'height': height,
            'width': width,
            'img': img,
            'target': target,
        }

    def __len__(self):
        return len(self.img_paths)


def batch_collate(batch):
    '''
    dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    '''
    batch_sample = dict()
    for key in batch[0].keys():
        batch_sample[key] = list()
        for sample in batch:
            batch_sample[key].append(sample[key])
    batch_sample['img'] = torch.stack(batch_sample['img'], 0)
    return batch_sample


'''
create dataloader
dataset = VOCDetection(data_root='/home/wxrui/DATA/VOCdevkit',
                       imgfiles=['VOC2007/ImageSets/Main/trainval.txt',
                                 'VOC2012/ImageSets/Main/trainval.txt'],
                       input_size=(512, 512), is_training=True)
dataloader = torch.utils.data.DataLoader(dataset, 4, shuffle=True, collate_fn=batch_collate)
sample = dataloader.__iter__().__next__()
'''
