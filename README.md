We release the training and testing code for THUMOS, EPIC-Kitchen 100 (verb, noun), and MultiTHUMOS datasets.

## Operating Systems and Specs
* Ubuntu 18.04.5 LTS
* NVIDIA RTX A6000 
* NVIDIA-SMI 520.61.05 - Driver Version: 520.61.05 - CUDA Version: 11.8

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
6. [Citation](#citation)

## Introduction
Recent studies have emphasized the importance of applying long-term temporal context modeling (TCM) blocks to the extracted video clip features such as employing complex self-attention mechanisms. In this paper, we present the simplest method ever to address this task and argue that the extracted video clip features are already informative to achieve outstanding performance without sophisticated architectures. To this end, we introduce TemporalAverager, which average long-term temporal context modeling while maximizing information from the extracted video clip features with a basic, parameter-free, and local region operating max-pooling block. Picking out only the most critical information for adjacent and local clip embeddings, this block results in a more efficient TAL model. We demonstrate that TemporalAverager outperforms other state-of-the-art methods that utilize long-term TCM such as self-attention on various TAL datasets while requiring significantly fewer parameters and computational resources.
<!-- <div align="center">
  <img src="figures/common_architecture.png" width="1100px"/>
</div> -->
Our method, TemporalAverager, results in the simplest model ever for TAL task that contains minimalist parameters and computational cost for the TAL model. TemporalAverager is effective at modeling temporal contexts, which outperforms the robust baseline, ActionFormer, with 2.8x fewer GMACs and 3x faster inference speed. Especially, when comparing only the backbone time, our proposed method only takes 2.5 ms which is incredibly 8.0x faster than ActionFormer backbone, 20.1 ms.

## Installation
#### a. Install packages
```bash
conda create -n TemporalAverager python=3.9
conda activate TemporalAverager
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
python -m pip install -r requirements.txt
pip install -e ./
```
#### b. Build NMS
Part of NMS is implemented in C++. The code can be compiled by

```shell
cd ./libs/utils; python setup.py install; cd ../..
```
The code should be recompiled every time you update PyTorch.


### Reproduce Our Results on THUMOS
##### Download Features and Annotations
- Download thumos.tar.gz from this [link](https://1drv.ms/u/s!AmoaChPnSuIOmwImo2-7iaRzrrAT?e=lyebbA), md5sum `1f71c37dba55d549e4b02841d0dcf603`.
- Make `data` folder in the current code directory.
- The data folder structure should look like the following:
```bash
# This folder
├── configs
│   └── ........
│   └── temporalmaxer_thumos_i3d.yaml
├── data
│   ├── thumos
│       ├── annotations
│       └── i3d_features
├── eval.py
├── figures
├── libs
    ........
```
- Train and test THUMOS
```bash
# training
./scripts/thumos/train.sh
# testing
./scripts/thumos/test.sh
```
* The results should be:

| Method            |  0.3  |  0.4  |  0.5  |  0.6  |  0.7  |  Avg  |
|-------------------|-------|-------|-------|-------|-------|-------|
| TemporalAverager     | 82.8  | 78.9  | 71.8  | 60.5  | 44.7  | 67.7  |


