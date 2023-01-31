# Apply Central Difference Convolutional Network (CDCN) for Face Anti Spoofing

An adaptation from [CDCN-Face-Anti-Spoofing.pytorch](https://github.com/voqtuyen/CDCN-Face-Anti-Spoofing.pytorch) and [face-anti-spoofing](https://github.com/laoshiwei/face-anti-spoofing), add script to build custom data and fix bugs

## Dependence

```bash
pip install -r requirements.txt
```

RTX3060 Laptop

## Dataset

get original data from [here](https://github.com/voqtuyen/CDCN-Face-Anti-Spoofing.pytorch/tree/master/data/nuaa) 

OR if you‘d like to train on custom data, use custom_data.py to add/create your custom dataset

before run custom_data.py, labeled images should be put in data/custom. Label format is like 0_xxxx.jpg

## Train

tune params in config yaml file and run train.py

to see log visualization: 

```
tensorboard --logdir=experiments/log --port=8008
```

## Inference

single image inference by inference.py

![mtcnn face](https://github.com/lrioxh/CDCN.pytorch/blob/main/data/inference/mtcnn%20face.png)

## Notes

#### difference from origin [CDCN](https://github.com/ZitongYu/CDCN): 

- The original version uses living face depth map estimated by PRNet and spoofing depth map is set to 0
  In this version, spoofing depth map is also set to 0, but the living face depth map is set to 1 directly

- due to the difference about depth map, scores are calculated differently

#### difference from other version: 

- add custom data & param θ=0.7 as in origin paper；
- add pre-processing before inference
- save optimal weights instead of latest
- add api for pre-generated depth maps. If you want to train with the depth map estimated by PRNet, put the depth map you generated from live face into data/train/depth and add a column "depth/xxxx" to the .csv corresponding to the depth map file. then set "depth_map_default" to 0 and train

