# Lecture Series on Object Detection Techniques

## Overview
This repository contains lecture series PDFs on two advanced topics in the field of object detection using Convolutional Neural Networks (CNNs). The lectures cover foundational techniques and innovative methods in object detection, detailing their methodologies, implementations, and performance considerations.

## Performance Metrics

### 1. Precision and Recall ([link](https://github.com/ccho0208/2024-summer-project_Deep-Learning/blob/main/3_lectures_detection/240620%20-%20Precision%20and%20Recall.pdf))

### 2. mAP: mean Average Precision ([link](https://github.com/ccho0208/2024-summer-project_Deep-Learning/blob/main/3_lectures_detection/240622%20-%20mAP.pdf))
- A survey on performance metrics for object-detection algorithms ([link](https://www.youtube.com/watch?v=c45jSJ3WGds&list=PLoEMreTa9CNm18TPHIYm3t2CLIqxLxzYD&index=1))
- mAP (mean Average Precision) [link](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

## Topics Covered

### 1. Region-Based Convolutional Neural Networks (R-CNNs, [link](https://arxiv.org/pdf/1311.2524))
The first topic in this lecture series delves into R-CNNs, the pioneering approach that utilized CNNs for object detection. Key points include:
- **Region Proposals**: R-CNNs use selective search to generate approximately 2000 region proposals per image.
- **Feature Extraction**: The model employs AlexNet for feature extraction from each proposed region.
- **Classification**: Features extracted are classified using a Support Vector Machine (SVM).

### 2. Spatial Pyramid Pooling Networks (SPPNs, [link](https://arxiv.org/pdf/1406.4729))
The second topic focuses on Spatial Pyramid Pooling Networks (SPPNs), which enhance the efficiency and flexibility of object detection. Key points include:
- **Single Pass CNN**: Unlike R-CNNs, SPPNs run the CNN only once over the entire image.
- **Fixed-Length Representations**: SPPNs use spatial pyramid pooling to convert region proposals into fixed-length representations, suitable for fully connected networks.
- **Flexibility in Spatial Size**: This method allows the model to handle objects of varying spatial sizes without needing to warp or crop them to a fixed size.

## Files Included
- `Lecture_R-CNN.pdf`: Detailed notes and explanations on R-CNNs, including their architecture, implementation, and performance analysis.
- `Lecture_SPPN.pdf`: Comprehensive coverage of SPPNs, with emphasis on their innovations and advantages over traditional R-CNNs.
