# Lecture Series on Object Detection Techniques

## Overview
This repository contains lecture series PDFs on two advanced topics in the field of object detection using Convolutional Neural Networks (CNNs). The lectures cover foundational techniques and innovative methods in object detection, detailing their methodologies, implementations, and performance considerations.

## Topics Covered

### 1. Region-Based Convolutional Neural Networks (R-CNNs, [link](https://arxiv.org/pdf/1311.2524))
The first topic in this lecture series delves into R-CNNs, the pioneering approach that utilized CNNs for object detection. Key points include:
- **Region Proposals**: R-CNNs use selective search to generate approximately 2000 region proposals per image.
- **Feature Extraction**: The model employs AlexNet for feature extraction from each proposed region.
- **Classification**: Features extracted are classified using a Support Vector Machine (SVM).
- **Performance**: While R-CNNs significantly improved object detection accuracy, they are computationally intensive, with processing times of about 13 seconds per image on a GPU and 53 seconds per image on a CPU.

### 2. Spatial Pyramid Pooling Networks (SPPNs)
The second topic focuses on Spatial Pyramid Pooling Networks (SPPNs), which enhance the efficiency and flexibility of object detection. Key points include:
- **Single Pass CNN**: Unlike R-CNNs, SPPNs run the CNN only once over the entire image.
- **Fixed-Length Representations**: SPPNs use spatial pyramid pooling to convert region proposals into fixed-length representations, suitable for fully connected networks.
- **Flexibility in Spatial Size**: This method allows the model to handle objects of varying spatial sizes without needing to warp or crop them to a fixed size.

## Files Included
- `Lecture_R-CNN.pdf`: Detailed notes and explanations on R-CNNs, including their architecture, implementation, and performance analysis.
- `Lecture_SPPN.pdf`: Comprehensive coverage of SPPNs, with emphasis on their innovations and advantages over traditional R-CNNs.
