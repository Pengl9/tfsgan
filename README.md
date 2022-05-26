# TFS_GAN: A Time-Frequency Semantic GAN Framework for Imbalanced Classification Using Radio Signals


This is the repo for Ubicomp 2022 paper: " TFS_GAN: A Time-Frequency Semantic GAN Framework for Imbalanced Classification Using Radio Signals ". We will iteratively update the TFS_GAN source code later.

## Installation

Use following command to install dependencies (python3.7 with pip installed):
```
pip3 install -r requirement.txt
```

If having trouble installing PyTorch, follow the original guidance (https://pytorch.org/).

Notably, the code is tested with ```TFS_GAN Version 1.0```.

## Pre-training on RF Datasets

Download [RF Datasets](https://cloud.tsinghua.edu.cn/d/87c76946e8e44be0a046/) dataset under [Data Folder]. 

To train single-wifi/uwb/mmwave/rfid for 200 epochs, run:
```
# Data_type in a.py needs to be changed.
python main.py
```

## Logs

Our code will be updated iteratively in the future.

#### 2022.05.17

First upload [ **partial code** of tfsgan ]

