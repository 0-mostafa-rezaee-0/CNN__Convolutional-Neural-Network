<div style="font-size:2em; font-weight:bold; text-align:center; margin-top:20px;">Figures Directory</div>

## 1. Overview

This directory contains visualizations generated during the CNN training and evaluation process.

## 2. Directory Contents

```
figures/
├── mnist_samples.png       # Grid of sample MNIST digits
├── confusion_matrix.png    # Confusion matrix of model predictions
├── training_history.png    # Training/validation accuracy and loss over epochs
├── feature_maps.png        # Visualization of CNN feature maps
└── prediction_samples.png  # Examples of correct and incorrect predictions
```

## 3. Visualization Descriptions

### 3.1 MNIST Samples (mnist_samples.png)
A grid visualization of sample digits from the MNIST dataset, showing examples from each class (0-9).

### 3.2 Confusion Matrix (confusion_matrix.png)
A heatmap showing the model's classification performance. Each cell (i,j) represents the number of samples from class i that were classified as class j. The diagonal elements represent correct classifications.

### 3.3 Training History (training_history.png)
Plots of the model's accuracy and loss on both training and validation sets over the training epochs. This visualization helps in understanding the learning progress and identifying issues like overfitting.

### 3.4 Feature Maps (feature_maps.png)
Visualizations of the feature maps learned by different convolutional layers in the CNN. These show what patterns and features the model is detecting at different layers.

### 3.5 Prediction Samples (prediction_samples.png)
A visualization showing examples of correct and incorrect predictions made by the model, helping to understand the model's strengths and limitations. 