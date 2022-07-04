import glob
import os
import numpy as np
import cv2
from PIL import Image
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomResizedCrop,
                                    RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", default="data/whubuilding/train/masks_origin")
    parser.add_argument("--output-mask-dir", default="data/whubuilding/train/masks")
    return parser.parse_args()


def rgb_to_label(mask):
    h, w = mask.shape[0], mask.shape[1]
    label = np.zeros(shape=(h, w), dtype=np.uint8)
    label[np.all(mask == [0, 0, 0], axis=-1)] = 1
    label[np.all(mask == [255, 255, 255], axis=-1)] = 0
    return label


def patch_format(inp):
    (mask_path, masks_output_dir) = inp
    # print(mask_path, masks_output_dir)
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    mask = cv2.imread(mask_path)
    label = rgb_to_label(mask)
    out_mask_path = os.path.join(masks_output_dir, "{}.png".format(mask_filename))
    cv2.imwrite(out_mask_path, label)


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    masks_dir = args.mask_dir
    masks_output_dir = args.output_mask_dir
    mask_paths = glob.glob(os.path.join(masks_dir, "*.png"))
    # print(mask_paths)

    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)

    inp = [(mask_path, masks_output_dir) for mask_path in mask_paths]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))


