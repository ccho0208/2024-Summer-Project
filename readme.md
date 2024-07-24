# AI Summer Lecture Notes and Projects

This repository contains materials and projects related to deep neural networks. 

## Table of Contents

- [Introduction](#introduction)
- [Development Environment](#set_up)
- [Database](#dataset)
- [Projects](#projects)
- [Resources](#resources)

## Introduction

This repository is dedicated to documenting my journey through a summer project and learning series on deep neural networks. It includes notes, code examples, and projects that I have worked on during the summer. The goal is to provide a comprehensive overview of the topics I have learned throughout the summer of 2024.

## Setting Up Development Environment

### Virtual Environment
1) If the packages (`python`, `pip package manager` and `venv`) are already installed, we create a new virtual environment by making a `~/.venv/venv_torch` directory to hold it:
```
$ python3 -m venv ~/.venv/venv_torch
```
2) Then, we activate the virtual environment using a shell-specific command:
```
$ source ~/.venv/venv_torch/bin/activate
```
3) When the virtual environment is active, the shell prompt is prefixed with `(venv_torch)`.
4) Install packages (`pytorch`, `scikit-learn`, `matplotlib`, etc) within the virtual environment by using `pip`.
```
(venv_torch) $ pip install --upgrade pip
(venv_torch) $ pip install torch torchvision torchaudio
```
5) To exit the virtual environment:
```
(venv_torch) $ deactivate
```

### Software Packages
1) scikit-learn: `pip install -U scikit-learn`
2) matplotlib: `pip install matplotlib`
3) PyYaml: `pip install PyYAML`
4) PyTorch model summary: `pip install pytorch-model-summary`


## Dataset

### 1) CIFAR-10
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. Raw image files can be found here ([link](https://www.kaggle.com/datasets/yiklunchow/cifar10raw)).

### 2) Pascal-VOC

The [Pascal Visual Object Classes (VOC)](http://host.robots.ox.ac.uk/pascal/VOC/) dataset is a benchmark for visual object recognition and detection. It includes 20 object classes and is split into annual challenges with separate training and test sets.

- **VOC 2007**: Contains 9,963 images with 24,640 annotated objects.
- **VOC 2012**: Contains 11,540 images with 27,450 annotated objects.

Each image is annotated with bounding boxes, object classes, and segmentation masks, supporting tasks like image classification, object detection, and segmentation. This dataset is widely used for developing and benchmarking computer vision models.

## Projects

### 1) Image Classification using CNNs ([link](https://github.com/ccho0208/2024-summer-project_Deep-Learning/tree/main/2_proj_image_classification))

The image classification pipeline implements CNN methods such as VGG and ResNet to classify CIFAR-10 image data.

### 2) Object Detection using YOLOv3 ([link](https://github.com/ccho0208/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3))

## Resources

