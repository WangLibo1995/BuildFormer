## [Building extraction with vision transformer](https://ieeexplore.ieee.org/document/9808187) 

*IEEE Transactions on Geoscience and Remote Sensing*, 2022, [Libo Wang](https://WangLibo1995.github.io), Shenghui Fang, [Rui Li](https://lironui.github.io/), Xiaoliang Meng

## Introduction

This project (BuildFormer) is an extension of our [GeoSeg](https://github.com/WangLibo1995/GeoSeg), which mainly focuses on building extraction from remote sensing images. **Code is on the way!**

  
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
│   ├── AerialImageDataset
│   │   ├── train
│   │   │   ├── train_images (splited original images)
│   │   │   ├── train_masks (splited original masks)
│   │   │   ├── val_images (splited original images)
│   │   │   ├── vak_masks (splited original masks)
│   │   │   ├── train
│   │   │   │   ├── images (processed)
│   │   │   │   ├── masks (processed)
│   │   │   ├── val
│   │   │   │   ├── images (processed)
│   │   │   │   ├── masks (processed)
│   │   │   │   ├── masks_gt (processed, for visualization)
│   ├── Massbuilding
│   │   ├── png
│   ├── whubuilding

```

## Install

Open the folder **airs** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r BuildFormer/requirements.txt
```

## Pretrained Weights

[Baidu Disk](https://pan.baidu.com/s/1foJkxeUZwVi5SnKNpn6hfg) : 1234

