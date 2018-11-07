import os
import torch
import torch.utils.data
import torchvision
import cv2
import numpy as np
from pycocotools.coco import COCO
from .data_aug import AugSequence


def get_labelmap():
    labelmap = {
        "none_of_the_above": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "11": 11,
        "13": 12,
        "14": 13,
        "15": 14,
        "16": 15,
        "17": 16,
        "18": 17,
        "19": 18,
        "20": 19,
        "21": 20,
        "22": 21,
        "23": 22,
        "24": 23,
        "25": 24,
        "27": 25,
        "28": 26,
        "31": 27,
        "32": 28,
        "33": 29,
        "34": 30,
        "35": 31,
        "36": 32,
        "37": 33,
        "38": 34,
        "39": 35,
        "40": 36,
        "41": 37,
        "42": 38,
        "43": 39,
        "44": 40,
        "46": 41,
        "47": 42,
        "48": 43,
        "49": 44,
        "50": 45,
        "51": 46,
        "52": 47,
        "53": 48,
        "54": 49,
        "55": 50,
        "56": 51,
        "57": 52,
        "58": 53,
        "59": 54,
        "60": 55,
        "61": 56,
        "62": 57,
        "63": 58,
        "64": 59,
        "65": 60,
        "67": 61,
        "70": 62,
        "72": 63,
        "73": 64,
        "74": 65,
        "75": 66,
        "76": 67,
        "77": 68,
        "78": 69,
        "79": 70,
        "80": 71,
        "81": 72,
        "82": 73,
        "84": 74,
        "85": 75,
        "86": 76,
        "87": 77,
        "88": 78,
        "89": 79,
        "90": 80
    }
    return labelmap


class COCOAnnotationTransform(object):
    """
    transform cocoannotations with [xmin, ymin, width, height] to [xmin, ymin, xmax, ymax]
    transform labelindex with [0--90] to [0--79]
    :return np.array
        [[xmin, ymin, xmax, ymax, label], [...], ,, ]
    """

    def __init__(self):
        self.label_map = get_labelmap()

    def __call__(self, target, width, height):
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            bbox = obj['bbox']  # xmin, ymin, width, height
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            label_idx = self.label_map[str(obj['category_id'])] - 1  # 0--79
            final_box = list(np.array(bbox) / scale)
            final_box.append(label_idx)
            res.append(final_box)
        if len(res) == 0:
            res.append([0.0, 0, 0, 0, 0])
        return np.array(res).astype(np.float32)


class COCODataset(torch.utils.data.Dataset):
    '''
     this Dataset just support coco2014 dataset now  !!!!!
    '''

    def __init__(self, data_root, jsonfile, input_size, is_training):
        self.data_root = data_root
        self.coco = COCO(os.path.join(data_root, jsonfile))
        self.input_size = input_size
        self.is_training = is_training
        self.augmentation = AugSequence()
        self.target_transform = COCOAnnotationTransform()
        self.img_ids = self.coco.getImgIds()

    def __getitem__(self, index):
        '''
        'img_id': img_id,
        'height': origion image height,
        'width': origion image width,
        'img': tensor.size(3,input_size)
        'target': np.array with [[xmin, ymin, xmax, ymax, label],...]
        '''
        img_id = self.img_ids[index]
        coco_img = self.coco.loadImgs(ids=img_id)[0]
        width = coco_img['width']
        height = coco_img['height']
        # img
        img_path = coco_img["file_name"]
        if 'train' in img_path:
            img_path = os.path.join(self.data_root, 'images/train2014', img_path)
        else:
            img_path = os.path.join(self.data_root, 'images/val2014', img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, self.input_size)
        # target
        ann_id = self.coco.getAnnIds(imgIds=img_id)
        coco_ann = self.coco.loadAnns(ids=ann_id)
        target = self.target_transform(coco_ann, width, height)
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
            'img_id': img_id,
            'height': height,
            'width': width,
            'img': img,
            'target': target,
        }

    def __len__(self):
        return len(self.img_ids)


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
dataset = COCODataset('/home/wxrui/DATA/coco', 'annotations/train.json', (512, 512), True)
dataloader = torch.utils.data.DataLoader(dataset, 4, shuffle=True, collate_fn=batch_collate)
sample = dataloader.__iter__().__next__()
'''
