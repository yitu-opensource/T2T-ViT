# Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet, [arxiv](https://arxiv.org/abs/2101.11986)

### Update: 2020/02/10:

1. Update the codes of feature visualization and attention map. 

2. I will update token_performer.py in this week, then you can run our T2T-ViT in your 12G GPUs.




<p align="center">
<img src="https://github.com/yitu-opensource/T2T-ViT/blob/main/images/f1.png">
</p>

Our codes are based on the [official imagenet example](https://github.com/pytorch/examples/tree/master/imagenet) by [PyTorch](https://pytorch.org/) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman)


## Requirements
[timm](https://github.com/rwightman/pytorch-image-models), pip install timm

torch>=1.4.0

torchvision>=0.5.0

pyyaml


## T2T-ViT Models


| Model    | T2T Transformer | Top1 Acc | #params |  Download|
| :---     |   :---:         |  :---:   |  :---:  |  :---:  | 
| T2T-ViT_t-14 | Transformer |   80.7   |  21.5M | [here](https://drive.google.com/file/d/1GG_hOMwC_ceDt_FqlESQ8QhCHATLfIJC/view?usp=sharing)  | 
| T2T-ViT_t-19 | Transformer |   81.75   |  39.0M | [here](https://drive.google.com/file/d/1GdTwGuvZKiZTs4euAmEvRwT_czDOKKqJ/view?usp=sharing) | 
| T2T-ViT_t-24 | Transformer |   82.2   |  64.1M | [here](https://drive.google.com/file/d/1Edw9jFasXFl5LVrRvJ44vMuQXOlvbDJP/view?usp=sharing) | 
| T2T-ViT-7    |  Performer  |   71.2   |  4.2M  | [here](https://drive.google.com/file/d/1nmp77cSrGfE1CeW_aUAFihfxmz4AWAcT/view?usp=sharing)| 
| T2T-ViT-10   |  Performer  |   74.1   |  5.8M  | [here](https://drive.google.com/file/d/1mn4Qyl-WfmytDSB530Nb0ie3Y5DMCzM_/view?usp=sharing)  | 
| T2T-ViT-12   |  Performer  |   75.5   |  6.8M  | [here](https://drive.google.com/file/d/1LMnlAFJsKnQLfbqX0vYs4n30H4DfXuI8/view?usp=sharing)  | 


## Test

Test the T2T-ViT_t-14 (take transformer in T2T transformer),

Download the [T2T-ViT_t-14](https://drive.google.com/file/d/1GG_hOMwC_ceDt_FqlESQ8QhCHATLfIJC/view?usp=sharing), then test it by running:

```
CUDA_VISIBLE_DEVICES=0 python main.py path/to/data --model T2t_vit_t_14 -b 100 --eval_checkpoint path/to/checkpoint
```

Test the T2T-ViT_t-24 (take transformer in T2T transformer),

Download the [T2T-ViT_t-24](https://drive.google.com/file/d/1Edw9jFasXFl5LVrRvJ44vMuQXOlvbDJP/view?usp=sharing), then test it by running:

```
CUDA_VISIBLE_DEVICES=0 python main.py path/to/data --model T2t_vit_t_24 -b 100 --eval_checkpoint path/to/checkpoint
```

## Train

Train the T2T-ViT_t-14 (take transformer in T2T transformer):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model T2t_vit_t_14 -b 64 --lr 5e-4 --weight-decay .05 --img-size 224
```

Train the T2T-ViT_t-24 (take transformer in T2T transformer):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model T2t_vit_t_24 -b 64 --lr 5e-4 --weight-decay .05 --img-size 224
```

## Visualization

Visualize the image features of ResNet50, you can open and run the [visualization-resnet.ipynb](https://github.com/yitu-opensource/T2T-ViT/blob/8cf18b1c99f8622292a897242240c31f87ac4489/visualization_resnet.ipynb) file in jupyter notebook or jupyter lab; some results are given as following:

<p align="center">
<img src="https://github.com/yitu-opensource/T2T-ViT/blob/main/images/resnet_conv1.png" width="600" height="300"/>
</p>

Visualize the image features of ViT, you can open and run the [visualization-vit.ipynb](https://github.com/yitu-opensource/T2T-ViT/blob/8cf18b1c99f8622292a897242240c31f87ac4489/visualization-vit.ipynb) file in jupyter notebook or jupyter lab; some results are given as following:

<p align="center">
<img src="https://github.com/yitu-opensource/T2T-ViT/blob/main/images/vit_block1.png" width="600" height="300"/>
</p>

Visualize attention map, you can refer to this [file](https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb). A simple example by visualizing the attention map in attention block 4 and 5 is:


<p align="center">
<img src="https://github.com/yitu-opensource/T2T-ViT/blob/main/images/attention_visualization.png" width="600" height="400"/>
</p>



Updating...

## Reference
If you find this repo useful, please consider citing:
```
@misc{yuan2021tokenstotoken,
    title={Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet},
    author={Li Yuan and Yunpeng Chen and Tao Wang and Weihao Yu and Yujun Shi and Francis EH Tay and Jiashi Feng and Shuicheng Yan},
    year={2021},
    eprint={2101.11986},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
