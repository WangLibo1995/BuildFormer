import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu

import matplotlib.patches as mpatches
from PIL import Image
import random

CLASSES = ('Building', 'Background')
PALETTE = [[255, 255, 255],  [0, 0, 0]]

ORIGIN_IMG_SIZE = (512, 512)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)


class InriaDataset(Dataset):
    def __init__(self, data_root='data/AerialImageDataset/train/train', mode='train', img_dir='images', mask_dir='masks',
                 img_suffix='.png', mask_suffix='.png', transform=None, mosaic_ratio=0.25,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)
        mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.float32)
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        return img, mask


def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def get_validation_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def get_test_transform():
    test_transform = [
        albu.Normalize()
    ]
    return albu.Compose(test_transform)
