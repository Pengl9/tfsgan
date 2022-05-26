# TFS_GAN: A Time-Frequency Semantic GAN Framework for Imbalanced Classification Using Radio Signals


This is the repo for Ubicomp 2022 paper: " TFS_GAN: A Time-Frequency Semantic GAN Framework for Imbalanced Classification Using Radio Signals ". We will iteratively update the TFS_GAN source code later.

## Installation

Use following command to install dependencies (python3.7 with pip installed):
```
pip3 install -r requirement.txt
```

If having trouble installing PyTorch, follow the original guidance (https://pytorch.org/).
Notably, the code is tested with ```cudatoolkit version 10.2```.

## Pre-training on ImageNet

Download [ImageNet](https://image-net.org/challenges/LSVRC/2012/) dataset under [ImageNet Folder]. Go to the path "[ImageNet Folder]/val" and use [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) to build sub-folders.

To train single-crop HCSC on 8 Tesla-V100-32GB GPUs for 200 epochs, run:
```
python3 -m torch.distributed.launch --master_port [your port] --nproc_per_node=8 \
pretrain.py [your ImageNet Folder]
```

To train multi-crop HCSC on 8 Tesla-V100-32GB GPUs for 200 epochs, run:
```
python3 -m torch.distributed.launch --master_port [your port] --nproc_per_node=8 \
pretrain.py --multicrop [your ImageNet Folder]
```
## Logs

Our code will be updated iteratively in the future.

#### 2022.05.17

First upload [ **partial code** of tfsgan ]

