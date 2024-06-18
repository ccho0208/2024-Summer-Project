# AI Summer Lecture Notes and Projects

This repository contains materials and projects related to deep neural networks. 

## Table of Contents

- [Introduction](#introduction)
- [Development Environment](#set_up)
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


### Colab


## Dataset

### 1) CIFAR-10
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. Raw image files can be found here ([link](https://www.kaggle.com/datasets/yiklunchow/cifar10raw)).

## Projects

### 1) Image Classification using CNNs

The image classification pipeline implements CNN methods such as VGG and ResNet to classify CIFAR-10 image data. 

## Resources

