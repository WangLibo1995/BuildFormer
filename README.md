## [Building extraction with vision transformer](https://ieeexplore.ieee.org/document/9808187) 

*IEEE Transactions on Geoscience and Remote Sensing*, 2022, [Libo Wang](https://WangLibo1995.github.io), Shenghui Fang, Xiaoliang Meng, [Rui Li](https://lironui.github.io/).

[Research Gate link](https://www.researchgate.net/publication/361583918_Building_extraction_with_vision_transformer)

## Introduction

This project (BuildFormer) is an extension of our [GeoSeg](https://github.com/WangLibo1995/GeoSeg), which mainly focuses on building extraction from remote sensing images.

  
## Folder Structure

Prepare the following folders to organize this repo:
```none
airs
├── BuildFormer (code)
├── pretrain_weights (save the pretrained weights like vit, swin, etc)
├── model_weights (save the model weights)
├── fig_results (save the masks predicted by models)
├── lightning_logs (CSV format training logs)
├── data
│   ├── AerialImageDataset (i.e. INRIA)
│   │   ├── train
│   │   │   ├── val_images (splited original images, ID 1-5 of each city)
│   │   │   ├── vak_masks (splited original masks, ID 1-5 of each city)
│   │   │   ├── train_images (splited original images, the other IDs)
│   │   │   ├── train_masks (splited original masks,  the other IDs)
│   │   │   ├── train
│   │   │   │   ├── images (processed)
│   │   │   │   ├── masks (processed)
│   │   │   ├── val
│   │   │   │   ├── images (processed)
│   │   │   │   ├── masks (processed)
│   │   │   │   ├── masks_gt (processed, for visualization)
│   ├── mass_build
│   │   ├── png
│   │   │   ├── train (original images)
│   │   │   ├── train_labels (original masks, RGB format)
│   │   │   ├── train_images (processed images)
│   │   │   ├── train_masks (processed masks, unit8 format)
│   │   │   ├── val (original images)
│   │   │   ├── val_labels (original masks, RGB format)
│   │   │   ├── val_images (processed images)
│   │   │   ├── val_masks (processed masks, unit8 format)
│   │   │   ├── test (original images)
│   │   │   ├── test_labels (original masks, RGB format)
│   │   │   ├── test_images (processed images)
│   │   │   ├── test_masks (processed masks, unit8 format)
│   ├── whubuilding
│   │   ├── train
│   │   │   ├── images (original images)
│   │   │   ├── masks_origin (original masks)
│   │   │   ├── masks (converted masks)
│   │   ├── val (the same with train)
│   │   ├── test (the same with test)
│   │   ├── train_val (Merge train and val)
```

## Install

Open the folder **airs** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r BuildFormer/requirements.txt
```

## Data Preprocessing

Download the [WHU Aerial](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html), [Massachusetts](https://www.cs.toronto.edu/~vmnih/data/), [INRIA](https://project.inria.fr/aerialimagelabeling/) building datasets and split them by **Folder Structure**.

**WHU**

```
python BuildFormer/whubuilding_mask_convert.py \
--mask-dir "data/whubuilding/train/masks_origin" \
--output-mask-dir "data/whubuilding/train/masks" 
```

```
python BuildFormer/whubuilding_mask_convert.py \
--mask-dir "data/whubuilding/val/masks_origin" \
--output-mask-dir "data/whubuilding/val/masks" 
```

```
python BuildFormer/whubuilding_mask_convert.py \
--mask-dir "data/whubuilding/test/masks_origin" \
--output-mask-dir "data/whubuilding/test/masks" 
```

**Massachusetts**

```
python BuildFormer/mass_patch_split.py \
--input-img-dir "data/mass_build/png/train" \
--input-mask-dir "data/mass_build/png/train_labels" \
--output-img-dir "data/mass_build/png/train_images" \
--output-mask-dir "data/mass_build/png/train_masks" \
--mode "train"
```

```
python BuildFormer/mass_patch_split.py \
--input-img-dir "data/mass_build/png/val" \
--input-mask-dir "data/mass_build/png/val_labels" \
--output-img-dir "data/mass_build/png/val_images" \
--output-mask-dir "data/mass_build/png/val_masks" \
--mode "val"
```

```
python BuildFormer/mass_patch_split.py \
--input-img-dir "data/mass_build/png/test" \
--input-mask-dir "data/mass_build/png/test_labels" \
--output-img-dir "data/mass_build/png/test_images" \
--output-mask-dir "data/mass_build/png/test_masks" \
--mode "val"
```

**INRIA**

```
python BuildFormer/inria_patch_split.py \
--input-img-dir "data/AerialImageDataset/train/train_images" \
--input-mask-dir "data/AerialImageDataset/train/train_masks" \
--output-img-dir "data/AerialImageDataset/train/train/images" \
--output-mask-dir "data/AerialImageDataset/train/train/masks" \
--mode "train"
```

```
python BuildFormer/inria_patch_split.py \
--input-img-dir "data/AerialImageDataset/train/val_images" \
--input-mask-dir "data/AerialImageDataset/train/val_masks" \
--output-img-dir "data/AerialImageDataset/train/val/images" \
--output-mask-dir "data/AerialImageDataset/train/val/masks" \
--mode "val"
```

## Training

```
python BuildFormer/train_supervision.py -c BuildFormer/config/whubuilding/buildformer.py
```

```
python BuildFormer/train_supervision.py -c BuildFormer/config/massbuilding/buildformer.py
```

```
python BuildFormer/train_supervision.py -c BuildFormer/config/inriabuilding/buildformer.py
```



## Testing

```
python BuildFormer/building_seg_test.py -c BuildFormer/config/whubuilding/buildformer.py -o fig_results/whubuilding/buildformer --rgb -t 'lr'
```

```
python BuildFormer/building_seg_test.py -c BuildFormer/config/massbuilding/buildformer.py -o fig_results/massbuilding/buildformer --rgb -t 'lr'
```

```
python BuildFormer/building_seg_test.py -c BuildFormer/config/inriabuilding/buildformer.py -o fig_results/inriabuilding/buildformer --rgb -t 'lr'
```

## Citation

If you find this project useful in your research, please consider citing our paper：

[Building extraction with vision transformer](https://ieeexplore.ieee.org/document/9808187)

## Acknowledgement

- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)