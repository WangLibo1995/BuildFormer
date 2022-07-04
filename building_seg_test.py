import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def label_to_rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 0]
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label_to_rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb images", action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + '.ckpt'), config=config)
    model.cuda()
    model.eval()
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input['img'].cuda())
            image_ids = input["img_id"]
            if 'gt_semantic_seg' in input.keys():
                masks_true = input['gt_semantic_seg']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            # input_images['features'] NxCxHxW C=3
            predictions = raw_predictions.argmax(dim=1)
            # print('preds shape', predictions[0,:,:])

            for i in range(raw_predictions.shape[0]):
                raw_mask = predictions[i].cpu().numpy()
                mask = raw_mask

                # print(mask.shape)
                if 'gt_semantic_seg' in input.keys():
                    evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                mask_name = image_ids[i]
                results.append((mask, str(args.output_path / mask_name), args.rgb))
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))
    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    precision = evaluator.Precision()
    recall = evaluator.Recall()
    for class_name, class_iou, class_f1 in zip(config.CLASSES, iou_per_class, f1_per_class):
        print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
    print('F1:{}, mIOU:{}, OA:{}, P:{}, R:{}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA,
                                                     np.nanmean(precision[:-1]), np.nanmean(recall[:-1])))


if __name__ == "__main__":
    main()
