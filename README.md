# <p align=center>`LVP for Domain-Adaptive Semantic Segmentation`</p><!-- omit in toc -->

## Table of Contents

  * [Introduction](#1-introduction)
  * [Environment Setup](#2-Environment-Setup)
  * [Dataset](#3-Dataset-Setup)
  * [Framework Structure](#4-Framework-Structure)
  * [Acknowledgements](#5-Acknowledgements)


## Code Availability Statement
This code is associated with a paper currently under review. To comply with the review process, the code will be made publicly available once the paper is accepted. 

We appreciate your understanding and patience. Once the code is released, we will warmly welcome any feedback and suggestions. Please stay tuned for our updates!

## Code Implementation Statement

As discussed in [8](https://github.com/lhoyer/MIC/issues/8), [54](https://github.com/lhoyer/MIC/issues/54) and [63](https://github.com/lhoyer/MIC/issues/63), **our method inherits the instability of [MIC](https://github.com/lhoyer/MIC).** :cry:

To address this, we will release two versions of the code: one designed to produce more stable results with lower standard deviation, and another that achieves higher performance albeit with greater variance. :smiley:

Note, however, that the mathematical expectation of performance is the same for both, i.e., 76.6% mIoU and 68.9% mIoU on GTAV→Cityscapes and SYNTHIA→Cityscapes, respectively. :100:

## 1. Introduction

 🔥 Pending


## 2. Environment Setup

First, please install cuda version 11.0.3 available at [https://developer.nvidia.com/cuda-11-0-3-download-archive](https://developer.nvidia.com/cuda-11-0-3-download-archive). It is required to build mmcv-full later.

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/TIP-seg
source ~/venv/TIP-seg/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Further, please download the MiT weights from SegFormer using the
following script. If problems occur with the automatic download, please follow
the instructions for a manual download within the script.

```shell
sh tools/download_checkpoints.sh
```

## 3. Dataset Setup

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.


The final folder structure should look like this:

```none
LVP
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
├── ...
```

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```

## 4. Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).


##  5. Acknowledgements

TIP is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
