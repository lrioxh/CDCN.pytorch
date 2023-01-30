# Apply Central Difference Convolutional Network (CDCN) for Face Anti Spoofing

An adaptation from [CDCN-Face-Anti-Spoofing.pytorch](https://github.com/voqtuyen/CDCN-Face-Anti-Spoofing.pytorch) and [face-anti-spoofing](https://github.com/laoshiwei/face-anti-spoofing)

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

single image inference by inference.py![mtcnn face](https://github.com/lrioxh/CDCN.pytorch/blob/main/data/inference/mtcnn%20face.png)

