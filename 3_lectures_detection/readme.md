# Lecture Series on Object Detection Techniques

## Overview
This repository contains lecture series PDFs on two advanced topics in the field of object detection using Convolutional Neural Networks (CNNs). The lectures cover foundational techniques and innovative methods in object detection, detailing their methodologies, implementations, and performance considerations.

## Performance Metrics

### 1. Precision and Recall ([link](https://github.com/ccho0208/2024-summer-project_Deep-Learning/blob/main/3_lectures_detection/0a%20-%20Precision%20and%20Recall.pdf))

### 2. mAP: mean Average Precision ([link](https://github.com/ccho0208/2024-summer-project_Deep-Learning/blob/main/3_lectures_detection/0b%20-%20mAP.pdf))
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

### 3. Fast R-CNN ([link](https://arxiv.org/pdf/1504.08083))
The third topic discusses Fast R-CNN, an improved version of R-CNN that streamlines the object detection process. Key points include:
- **ROI Projection**: Fast R-CNN projects Region of Interest (ROI) proposals onto a convolutional feature map.
- **ROI Pooling**: The method uses ROI pooling layers to convert variable-sized ROIs into fixed-size feature vectors.
- **Classification and Regression**: These fixed-size feature vectors are fed into a softmax classifier (with \( k+1 \) classes) and a regressor for bounding box prediction.
- **Loss Function**: The loss function is defined as:

  $$
  \text{loss} = L_{\text{cls}}(p,u) + \lambda [u \geq 1] L_{\text{loc}}(t,n)
  $$

  where \( $L_{\text{cls}}$ \) is the classification loss and \( $L_{\text{loc}}$ \) is the localization loss.

### 4. Faster R-CNN ([link](https://arxiv.org/pdf/1506.01497))
The fourth topic covers Faster R-CNN, which introduces a deep learning-based approach for region proposal generation. Key points include:
- **Region Proposal Network (RPN)**: Faster R-CNN integrates a Region Proposal Network that generates region proposals, making the detection process end-to-end.
- **Anchor Boxes**: The use of anchor boxes in the RPN helps in predicting bounding boxes of different scales and aspect ratios.
- **Pipeline**: The overall pipeline consists of feature extraction -> region proposal -> bounding boxes, streamlining the detection process and significantly improving speed and accuracy.

### 5. YOLO ([link](https://arxiv.org/pdf/1506.02640))
- YOLO algorithm (Andrew Ng) ([link](https://www.youtube.com/watch?v=9s_FpMpdYW8))
- What is YOLO algorithm? ([link](https://www.youtube.com/watch?v=ag3DLKsl2vk))
- Evolution of YOLO algorithm ([link](https://encord.com/blog/yolo-object-detection-guide/))

## Files Included
- `Lecture_R-CNN.pdf`: Detailed notes and explanations on R-CNNs, including their architecture, implementation, and performance analysis.
- `Lecture_SPPN.pdf`: Comprehensive coverage of SPPNs, with emphasis on their innovations and advantages over traditional R-CNNs.
- `Lecture_Fast_R-CNN.pdf`: Insights into Fast R-CNN, explaining its ROI projection, pooling layers, classification and regression, and loss function.
- `Lecture_Faster_R-CNN.pdf`: Explanation of Faster R-CNN, focusing on its Region Proposal Network, anchor boxes, and streamlined pipeline.
