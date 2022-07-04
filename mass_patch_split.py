import glob
import os
import numpy as np
import cv2
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu

import random

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


Building = np.array([255, 255, 255])  # label 0
Clutter = np.array([0, 0, 0]) # label 1
num_classes = 2


# split huge RS image to small patches
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-img-dir", default="data/mass_build/png/train")
    parser.add_argument("--input-mask-dir", default="data/mass_build/png/train_labels")
    parser.add_argument("--output-img-dir", default="data/mass_build/png/train_images")
    parser.add_argument("--output-mask-dir", default="data/mass_build/png/train_masks")
    parser.add_argument("--mode", type=str, default='train')

    return parser.parse_args()


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = Building
    mask_rgb[np.all(mask_convert == 1, axis=0)] = Clutter

    return mask_rgb


def rgb2label(label):
    label_seg = np.zeros(label.shape[:2], dtype=np.uint8)
    label_seg[np.all(label == Building, axis=-1)] = 0
    label_seg[np.all(label == Clutter, axis=-1)] = 1

    return label_seg


def image_augment(image, mask, mode='train'):
    image_list = []
    mask_list = []
    image_width, image_height = image.shape[1], image.shape[0]
    mask_width, mask_height = mask.shape[1], mask.shape[0]
    assert image_height == mask_height and image_width == mask_width
    if mode == 'train':
        hflip = albu.HorizontalFlip(p=1)(image=image.copy(), mask=mask.copy())
        img_h, mask_h = hflip['image'], hflip['mask']

        vflip = albu.VerticalFlip(p=1)(image=image.copy(), mask=mask.copy())
        img_v, mask_v = vflip['image'], vflip['mask']

        image_list_train = [image, img_h, img_v]
        mask_list_train = [mask, mask_h, mask_v]
        for i in range(len(image_list_train)):
            mask_tmp = rgb2label(mask_list_train[i])
            image_list.append(image_list_train[i])
            mask_list.append(mask_tmp)
    else:
        mask = rgb2label(mask.copy())
        image_list.append(image)
        mask_list.append(mask)
    return image_list, mask_list


def patch_format(inp):
    (img_path, mask_path, imgs_output_dir, masks_output_dir, mode) = inp

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    id = os.path.splitext(os.path.basename(img_path))[0]
    assert img.shape == mask.shape

    if mode == 'train':
        mask_tmp = np.zeros(mask.shape[:2], dtype=np.uint8)
        mask_tmp[np.all(img == [255, 255, 255], axis=-1)] = 1
        mask_c = mask_tmp[np.newaxis, :, :]
        mask[np.all(mask_c == 1, axis=0)] = [0, 0, 0]
        img[np.all(img == [255, 255, 255], axis=-1)] = [0, 0, 0]

    image_list, mask_list = image_augment(image=img.copy(), mask=mask.copy(), mode=mode)
    assert len(image_list) == len(mask_list)
    for m in range(len(image_list)):
        img = image_list[m]
        mask = mask_list[m]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out_img_path = os.path.join(imgs_output_dir, "{}_{}.png".format(id, m))
        cv2.imwrite(out_img_path, img)

        out_mask_path = os.path.join(masks_output_dir, "{}_{}.png".format(id, m))
        cv2.imwrite(out_mask_path, mask)


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    input_img_dir = args.input_img_dir
    input_mask_dir = args.input_mask_dir
    img_paths = glob.glob(os.path.join(input_img_dir, "*.png"))
    mask_paths = glob.glob(os.path.join(input_mask_dir, "*.png"))
    img_paths.sort()
    mask_paths.sort()

    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir

    mode = args.mode

    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)

    inp = [(img_path, mask_path, imgs_output_dir, masks_output_dir, mode)
           for img_path, mask_path in zip(img_paths, mask_paths)]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))


