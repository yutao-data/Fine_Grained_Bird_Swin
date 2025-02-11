# Fine-Grained Bird Image Classification: A Two-Stage Transformer-Based Approach

## Overview
This repository contains the implementation of a fine-grained bird image classification model. The project focuses on distinguishing bird species using a novel **two-stage transformer-based approach**, leveraging Swin and Multi-Granularity Part Sampling Attention (MPSA) models.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [References](#references)

## Introduction
Fine-grained image classification (FGVC) is a challenging task due to high intra-class variation and inter-class similarity. We address these challenges using a two-stage model:
- **Stage 1**: A Swin Transformer model performs an initial classification.
- **Stage 2**: The MPSA model refines predictions for visually similar bird species (e.g., blackbirds and crows).

This project was part of a **Kaggle competition** and aims to improve classification accuracy using deep learning and fine-grained attention techniques.

## Dataset
We use a subset of the **Caltech-UCSD Birds-200-2011** dataset, which contains labeled images of 20 different bird species. Some challenges in the dataset include:
- **Low resolution**: Images have a maximum of 500x500 pixels.
- **Background interference**: Some birds blend into the background.
- **Class imbalance**: Certain species have fewer training images.

### Sample Data Distribution
| Label                 | Train | Validation |
|-----------------------|-------|------------|
| American Crow        | 53    | 7          |
| Red-winged Blackbird | 53    | 7          |
| Indigo Bunting      | 57    | 3          |
| ...                 | ...   | ...        |

## Methodology
Our approach consists of:
1. **Preprocessing**: 
   - Background removal and cropping (DeepLabV3 & Faster R-CNN).
   - Geometric transformations (rotation, flipping, cropping).
   - Color augmentations (contrast adjustments, random noise).
2. **Model Architecture**:
   - **Swin Transformer**: Hierarchical self-attention for general classification.
   - **MPSA Model**: Multi-granularity part sampling attention for refining hard-to-classify species.
3. **Training Strategy**:
   - Optimizer: AdamW.
   - Loss Function: CrossEntropyLoss.
   - Mixed precision training.
   - Early stopping to prevent overfitting.

## Implementation Details
- **Primary Model**: Swin Transformer (Pre-trained on CUB-200-2011)
- **Secondary Model**: MPSA Model (Trained on visually similar species)
- **Libraries Used**:
  - `torch`, `torchvision`
  - `albumentations` for data augmentation
  - `opencv` for preprocessing
  - `transformers` for Swin model

## Results
The proposed approach achieved high classification accuracy:

| Model                        | Validation Accuracy | Test Accuracy |
|------------------------------|---------------------|--------------|
| AlexNet                      | 75.00%             | -            |
| ResNet18                     | 79.66%             | -            |
| VGG16                        | 85.17%             | -            |
| ViT-base                     | 88.35%             | 84.50%       |
| Swin-base                     | 95.76%             | 86.50%       |
| **Swin Pre-trained**         | **97.46%**         | **88.00%**   |
| MPSA                         | 95.23%             | 87.50%       |
| Two-Stage (Swin + MPSA)      | -                   | 87.00%       |

## Installation
To install dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
### Train the Model
```bash
python main.py --train
```

### Test the Model
```bash
python main.py --test
```

### Evaluate Performance
```bash
python evaluate.py
```

## Future Work
- Incorporate **self-supervised learning** (DINOv2) to improve feature extraction.
- Explore **semi-supervised** techniques to leverage unlabeled data.
- Experiment with **efficient vision transformers** to reduce computation time.

## References
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [Multi-Granularity Part Sampling Attention](https://arxiv.org/abs/2210.07181)
- [Caltech-UCSD Birds-200-2011 Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

## Authors
Yutao Chen - Xianyun Zhuang - Gabriel Lozano  
(Centrale Supélec)
