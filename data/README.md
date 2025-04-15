<div style="font-size:2em; font-weight:bold; text-align:center; margin-top:20px;">Data Directory</div>

## 1. Overview

This directory contains the MNIST dataset files and sample images extracted for visualization purposes.

## 2. Directory Structure

```
data/
├── mnist/              # Raw and preprocessed MNIST data
└── mnist_samples/      # Sample images extracted from the MNIST dataset
```

## 3. MNIST Dataset

The MNIST dataset is a large collection of handwritten digits commonly used for training various image processing systems. It contains:
- 60,000 training images
- 10,000 test images
- Each image is a 28×28 pixel grayscale image of a handwritten digit (0-9)

The dataset is automatically downloaded when running the scripts or notebooks in this project.

## 4. Data Preprocessing

When loaded through our scripts, the MNIST data undergoes the following preprocessing steps:
- Normalization (pixel values scaled to range [0, 1])
- Reshaping to include channel dimension (28×28×1)
- One-hot encoding of labels
- Optional train/validation split

## 5. Sample Images

The `mnist_samples/` directory contains sample images extracted from the dataset for each digit class. These samples are used for visualization in the notebooks and generated during the data preparation step. 