# Minimalist and High-Performance Semantic Segmentation with Plain Vision Transformers

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/minimalist-and-high-performance-semantic/semantic-segmentation-on-pascal-context)](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-context?p=minimalist-and-high-performance-semantic)

The official implementation of [our paper](https://arxiv.org/abs/2310.12755).

## Install and Usage

Please follow [ViT-Adapter](https://github.com/czczup/ViT-Adapter/tree/main/segmentation) to prepare the environment and datasets.

Don't forget to convert the pre-trained weights of BEiT and BEiTv2 with beit2mmseg.py in tools.

## Training

Please use the following commands. We fix random seeds to reduce randomness.

To train base models on ADE20K with 4gpus:

```
sh ./tools/dist_train.sh configs/ade/mask2former_beit_base_parallel_separate_slim_640_80k_ade20k_ss.py 4 --seed 0
```

To train large models on ADE20K with 8gpus:

```
sh ./tools/dist_train.sh configs/ade/mask2former_beit_large_parallel_separate_slim_640_80k_ade20k_ss.py 8 --seed 0
```

To train base models on Pascal Context with 4gpus:

```
sh ./tools/dist_train.sh configs/pascal/mask2former_beit_base_parallel_separate_slim_480_20k_pascal_ss.py 4 --seed 10
```

To train large models on Pascal Context with 4gpus:

```
sh ./tools/dist_train.sh configs/pascal/mask2former_beit_large_parallel_separate_slim_480_20k_pascal_ss.py 4 --seed 10
```

To train large models on COCO-Stuff 164K with 8gpus:

```
sh ./tools/dist_train.sh configs/coco164k/mask2former_beit_large_parallel_separate_slim_640_80k_coco164_ss.py 8 --seed 0
```

## Pre-trained Models

Coming soon!

## Acknowledgement

The code is largely based on [ViT-Adapter](https://github.com/czczup/ViT-Adapter/tree/main/segmentation) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).
