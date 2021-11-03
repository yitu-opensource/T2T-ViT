# Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet, [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Tokens-to-Token_ViT_Training_Vision_Transformers_From_Scratch_on_ImageNet_ICCV_2021_paper.html)

### Update:
2021/03/11: update our new results. Now our T2T-ViT-14 with 21.5M parameters can reach 81.5% top1-acc with 224x224 image resolution, and 83.3\% top1-acc with 384x384 resolution. 

2021/02/21: T2T-ViT can be trained on most of common GPUs: 1080Ti, 2080Ti, TiTAN V, V100 stably with '--amp' (Automatic Mixed Precision). In some specifical GPU like Tesla T4, 'amp' would cause NAN loss when training T2T-ViT. If you get NAN loss in training, you can disable amp by removing '--amp' in the [training scripts](https://github.com/yitu-opensource/T2T-ViT#train).

2021/01/28: release codes and upload most of the pretrained models of T2T-ViT.

<p align="center">
<img src="https://github.com/yitu-opensource/T2T-ViT/blob/main/images/f1.png">
</p>

## Reference
If you find this repo useful, please consider citing:
```
@InProceedings{Yuan_2021_ICCV,
    author    = {Yuan, Li and Chen, Yunpeng and Wang, Tao and Yu, Weihao and Shi, Yujun and Jiang, Zi-Hang and Tay, Francis E.H. and Feng, Jiashi and Yan, Shuicheng},
    title     = {Tokens-to-Token ViT: Training Vision Transformers From Scratch on ImageNet},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {558-567}
}
```

Our codes are based on the [official imagenet example](https://github.com/pytorch/examples/tree/master/imagenet) by [PyTorch](https://pytorch.org/) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman)


## 1. Requirements

[timm](https://github.com/rwightman/pytorch-image-models), pip install timm==0.3.4

torch>=1.4.0

torchvision>=0.5.0

pyyaml

data prepare: ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## 2. T2T-ViT Models


| Model    | T2T Transformer | Top1 Acc | #params | MACs |  Download|
| :---     |   :---:         |  :---:   |  :---:  | :---: |  :---:   | 
| T2T-ViT-14   |  Performer  |   81.5   |  21.5M  | 4.8G  | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.5_T2T_ViT_14.pth.tar)| 
| T2T-ViT-19   |  Performer  |   81.9   |  39.2M  | 8.5G  | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.9_T2T_ViT_19.pth.tar)| 
| T2T-ViT-24   |  Performer  |   82.3   |  64.1M  | 13.8G  | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.3_T2T_ViT_24.pth.tar)| 
| T2T-ViT-14, 384   |  Performer  |   83.3   |  21.7M  |   | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/83.3_T2T_ViT_14.pth.tar)|
| T2T-ViT-24, Token Labeling   |  Performer  |   84.2   |  65M  |   | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/84.2_T2T_ViT_24.pth.tar)| 
| T2T-ViT_t-14 | Transformer |   81.7   |  21.5M  | 6.1G | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.7_T2T_ViTt_14.pth.tar)  | 
| T2T-ViT_t-19 | Transformer |   82.4   |  39.2M  | 9.8G  | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.4_T2T_ViTt_19.pth.tar) | 
| T2T-ViT_t-24 | Transformer |   82.6   |  64.1M  | 15.0G| [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.6_T2T_ViTt_24.pth.tar) | 

The 'T2T-ViT-14, 384' means we train T2T-ViT-14 with image size of 384 x 384.

The 'T2T-ViT-24, Token Labeling' means we train T2T-ViT-24 with [Token Labeling](https://github.com/zihangJiang/TokenLabeling).

The three lite variants of T2T-ViT (Comparing with MobileNets):
| Model    | T2T Transformer | Top1 Acc | #params | MACs |  Download|
| :---     |   :---:         |  :---:   |  :---:  | :---: |  :---:   | 
| T2T-ViT-7   |  Performer  |   71.7   |  4.3M   | 1.1G  | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/71.7_T2T_ViT_7.pth.tar)| 
| T2T-ViT-10   |  Performer  |   75.2   |  5.9M   | 1.5G  | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/75.2_T2T_ViT_10.pth.tar)| 
| T2T-ViT-12   |  Performer  |   76.5   |  6.9M   | 1.8G  | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/76.5_T2T_ViT_12.pth.tar)  |


### Usage
The way to use our pretrained T2T-ViT:
```
from models.t2t_vit import *
from utils import load_for_transfer_learning 

# create model
model = t2t_vit_14()

# load the pretrained weights
load_for_transfer_learning(model, /path/to/pretrained/weights, use_ema=True, strict=False, num_classes=1000)  # change num_classes based on dataset, can work for different image size as we interpolate the position embeding for different image size.
```


## 3. Validation

Test the T2T-ViT-14 (take Performer in T2T module),

Download the [T2T-ViT-14](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.5_T2T_ViT_14.pth.tar), then test it by running:

```
CUDA_VISIBLE_DEVICES=0 python main.py path/to/data --model t2t_vit_14 -b 100 --eval_checkpoint path/to/checkpoint
```
The results look like:

```
Test: [   0/499]  Time: 2.083 (2.083)  Loss:  0.3578 (0.3578)  Acc@1: 96.0000 (96.0000)  Acc@5: 99.0000 (99.0000)
Test: [  50/499]  Time: 0.166 (0.202)  Loss:  0.5823 (0.6404)  Acc@1: 85.0000 (86.1569)  Acc@5: 99.0000 (97.5098)
...
Test: [ 499/499]  Time: 0.272 (0.172)  Loss:  1.3983 (0.8261)  Acc@1: 62.0000 (81.5000)  Acc@5: 93.0000 (95.6660)
Top-1 accuracy of the model is: 81.5%

```

Test the three lite variants: T2T-ViT-7, T2T-ViT-10, T2T-ViT-12 (take Performer in T2T module),

Download the [T2T-ViT-7](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/71.7_T2T_ViT_7.pth.tar), [T2T-ViT-10](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/75.2_T2T_ViT_10.pth.tar) or [T2T-ViT-12](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/76.5_T2T_ViT_12.pth.tar), then test it by running:

```
CUDA_VISIBLE_DEVICES=0 python main.py path/to/data --model t2t_vit_7 -b 100 --eval_checkpoint path/to/checkpoint
```

Test the model T2T-ViT-14, 384 with 83.3\% top-1 accuracy: 
```
CUDA_VISIBLE_DEVICES=0 python main.py path/to/data --model t2t_vit_14 --img-size 384 -b 100 --eval_checkpoint path/to/T2T-ViT-14-384 
```


## 4. Train

Train the three lite variants: T2T-ViT-7, T2T-ViT-10 and T2T-ViT-12 (take Performer in T2T module):

If only 4 GPUs are available,

```
CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 path/to/data --model t2t_vit_7 -b 128 --lr 1e-3 --weight-decay .03 --amp --img-size 224
```

The top1-acc in 4 GPUs would be slightly lower than 8 GPUs (around 0.1%-0.3% lower).

If 8 GPUs are available: 
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model t2t_vit_7 -b 64 --lr 1e-3 --weight-decay .03 --amp --img-size 224
```


Train the T2T-ViT-14 and T2T-ViT_t-14 (run on 4 or 8 GPUs):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 path/to/data --model t2t_vit_14 -b 128 --lr 1e-3 --weight-decay .05 --amp --img-size 224
```

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model t2t_vit_14 -b 64 --lr 5e-4 --weight-decay .05 --amp --img-size 224
```
If you want to train our T2T-ViT on images with 384x384 resolution, please use '--img-size 384'.


Train the T2T-ViT-19, T2T-ViT-24 or T2T-ViT_t-19, T2T-ViT_t-24:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model t2t_vit_19 -b 64 --lr 5e-4 --weight-decay .065 --amp --img-size 224
```

## 5. Transfer T2T-ViT to CIFAR10/CIFAR100

| Model        |  ImageNet | CIFAR10 |  CIFAR100| #params| 
| :---         |    :---:  | :---:   |  :---:   |   :---:  | 
| T2T-ViT-14   |   81.5    |[98.3](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/cifar10_t2t-vit_14_98.3.pth)  | [88.4](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/cirfar100_t2t-vit-14_88.4.pth) | 21.5M    | 
| T2T-ViT-19   |   81.9    |[98.4](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/cifar10_t2t-vit_19_98.4.pth)  | [89.0](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/cifar100_t2t-vit-19_89.0.pth) |39.2M     | 

We resize CIFAR10/100 to 224x224 and finetune our pretrained T2T-ViT-14/19 to CIFAR10/100 by running:

```
CUDA_VISIBLE_DEVICES=0,1 transfer_learning.py --lr 0.05 --b 64 --num-classes 10 --img-size 224 --transfer-learning True --transfer-model /path/to/pretrained/T2T-ViT-19
```

## 6. Visualization

Visualize the image features of ResNet50, you can open and run the [visualization_resnet.ipynb](https://github.com/yitu-opensource/T2T-ViT/blob/main/visualization_resnet.ipynb) file in jupyter notebook or jupyter lab; some results are given as following:

<p align="center">
<img src="https://github.com/yitu-opensource/T2T-ViT/blob/main/images/resnet_conv1.png" width="600" height="300"/>
</p>

Visualize the image features of ViT, you can open and run the [visualization_vit.ipynb](https://github.com/yitu-opensource/T2T-ViT/blob/main/visualization_vit.ipynb) file in jupyter notebook or jupyter lab; some results are given as following:

<p align="center">
<img src="https://github.com/yitu-opensource/T2T-ViT/blob/main/images/vit_block1.png" width="600" height="300"/>
</p>

Visualize attention map, you can refer to this [file](https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb). A simple example by visualizing the attention map in attention block 4 and 5 is:


<p align="center">
<img src="https://github.com/yitu-opensource/T2T-ViT/blob/main/images/attention_visualization.png" width="600" height="400"/>
</p>


