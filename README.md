# Minimalist and High-Performance Semantic Segmentation with Plain Vision Transformers

## Install and Usage

Please follow [ViT-Adapter](https://github.com/czczup/ViT-Adapter/tree/main/segmentation) to prepare the environment and datasets.

## Training

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
sh ./tools/dist_train.sh configs/pascal/mask2former_beit_large_parallel_separate_slim_480_20k_pascal_ss.py 4 --seed 0
```

To train large models on COCO-Stuff 164K with 8gpus:

```
sh ./tools/dist_train.sh configs/coco164k/mask2former_beit_large_parallel_separate_slim_640_80k_coco164_ss.py 8 --seed 0
```

## Acknowledgement

[ViT-Adapter](https://github.com/czczup/ViT-Adapter/tree/main/segmentation), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
