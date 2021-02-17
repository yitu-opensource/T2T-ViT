# Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet, [arxiv](https://arxiv.org/abs/2101.11986)

### Update: 2020/02/14:

Update token_performer.py, now t2t-vit-7 can be trained with 4 GPUs with 12G memory,other T2T-ViT also can be trained with 4 or 8 GPUs.




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
| T2T-ViT-7    |  Performer  |   71.3   |  4.3M  | [here](https://drive.google.com/file/d/1gTvmvUhdjTNJpgEKJ-iqEhChdKWCFU3M/view?usp=sharing)| 
| T2T-ViT-10   |  Performer  |   74.0   |  5.9M  | [here](https://drive.google.com/file/d/1s_cTYsUcPWhhdDXxn4CvpA-G7u-OpGgX/view?usp=sharing)| 
| T2T-ViT-12   |  Performer  |   75.6   |  6.9M  | [here](https://drive.google.com/file/d/1uldU_G3oawOF8hWuZEGRuL1lxjbU58Ly/view?usp=sharing)  |
| T2T-ViT-14   |  Performer  |   80.6   |  21.5M | [here](https://drive.google.com/file/d/1zTXtcGwIS_AmPqhUDACYDITDmnNP2yLI/view?usp=sharing)| 
| T2T-ViT-19   |  Performer  |   81.4   |  39.0M | [here](https://drive.google.com/file/d/1uXOXQ44wNvHOpQxL39jkpcexJv5wH6DG/view?usp=sharing)| 
| T2T-ViT-24   |  Performer  |   81.8   |  64.1M | [here](https://drive.google.com/file/d/1XujowogyGVR81EsUsYAwlJeeZjERrOd0/view?usp=sharing)| 
| T2T-ViT_t-14 | Transformer |   80.7   |  21.5M | [here](https://drive.google.com/file/d/1GG_hOMwC_ceDt_FqlESQ8QhCHATLfIJC/view?usp=sharing)  | 
| T2T-ViT_t-19 | Transformer |   81.75  |  39.0M | [here](https://drive.google.com/file/d/1GdTwGuvZKiZTs4euAmEvRwT_czDOKKqJ/view?usp=sharing) | 
| T2T-ViT_t-24 | Transformer |   82.2   |  64.1M | [here](https://drive.google.com/file/d/1Edw9jFasXFl5LVrRvJ44vMuQXOlvbDJP/view?usp=sharing) | 



## Test

Test the T2T-ViT-7 or T2T-ViT-12 (take Performer in T2T transformer),

Download the [T2T-ViT-7](https://drive.google.com/file/d/1gTvmvUhdjTNJpgEKJ-iqEhChdKWCFU3M/view?usp=sharing) or [T2T-ViT-12](https://drive.google.com/file/d/1uldU_G3oawOF8hWuZEGRuL1lxjbU58Ly/view?usp=sharing), then test it by running:

```
CUDA_VISIBLE_DEVICES=0 python main.py path/to/data --model T2t_vit_7 -b 100 --eval_checkpoint path/to/checkpoint
```

Test the T2T-ViT-14 (take Performer in T2T transformer),

Download the [T2T-ViT-14](https://drive.google.com/file/d/1zTXtcGwIS_AmPqhUDACYDITDmnNP2yLI/view?usp=sharing), then test it by running:

```
CUDA_VISIBLE_DEVICES=0 python main.py path/to/data --model T2t_vit_14 -b 100 --eval_checkpoint path/to/checkpoint
```

Test the T2T-ViT_t-24 (take Transformer in T2T transformer),

Download the [T2T-ViT_t-24](https://drive.google.com/file/d/1Edw9jFasXFl5LVrRvJ44vMuQXOlvbDJP/view?usp=sharing), then test it by running:

```
CUDA_VISIBLE_DEVICES=0 python main.py path/to/data --model T2t_vit_t_24 -b 100 --eval_checkpoint path/to/checkpoint
```

## Train

Train the T2T-ViT-7 and T2T-ViT-12 (take Performer in T2T transformer):

If only 4 GPUs are available,

```
CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 path/to/data --model T2t_vit_7 -b 128 --lr 1e-3 --weight-decay .03 --cutmix 0.0 --reprob 0.25 --img-size 224
```

The top1-acc in 4 GPUs would be slightly lower than 8 GPUs (around 0.1%-0.3% lower).

If 8 GPUs are available: 
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model T2t_vit_7 -b 64 --lr 1e-3 --weight-decay .03 --cutmix 0.0 --reprob 0.25 --img-size 224
```


Train the T2T-ViT-14 or T2T-ViT-19 or T2T-ViT-24 (take Performer in T2T transformer):

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model T2t_vit_14 -b 64 --lr 5e-4 --weight-decay .05 --img-size 224
```


Train the T2T-ViT_t-14, T2T-ViT_t-19 or T2T-ViT_t-24 (take Transformer in T2T transformer):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model T2t_vit_t_14 -b 64 --lr 5e-4 --weight-decay .05 --img-size 224
```



## Visualization

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
