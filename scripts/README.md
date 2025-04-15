<div style="font-size:2em; font-weight:bold; text-align:center; margin-top:20px;">Scripts Directory</div>

## Table of Contents 
<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#1-overview"><i><b>1. Overview</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#2-directory-contents"><i><b>2. Directory Contents</b></i></a>
</div>
&nbsp;

<details>
  <summary><a href="#3-script-descriptions"><i><b>3. Script Descriptions</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#31-data_preppy">3.1. data_prep.py</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#32-extract_sample_imagespy">3.2. extract_sample_images.py</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#33-train_cnnpy">3.3. train_cnn.py</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#34-visualize_featurespy">3.4. visualize_features.py</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#35-analyze_predictionspy">3.5. analyze_predictions.py</a><br>
  </div>
</details>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#4-common-parameters"><i><b>4. Common Parameters</b></i></a>
</div>
&nbsp;

## 1. Overview

This directory contains Python scripts for each major function of the MNIST digit recognition project.

## 2. Directory Contents

```
scripts/
├── analyze_predictions.py   # Detailed analysis of model predictions
├── data_prep.py             # Download and preprocess MNIST data
├── extract_sample_images.py # Extract sample images for visualization
├── train_cnn.py             # Train the CNN model
└── visualize_features.py    # Generate feature map visualizations
```

## 3. Script Descriptions

### 3.1 data_prep.py

Downloads and preprocesses the MNIST dataset:
- Downloads MNIST data if not already available
- Normalizes the pixel values to [0, 1]
- Reshapes the data to include the channel dimension
- Splits the data into training, validation, and test sets
- Saves the processed data for later use

Usage:
```bash
python scripts/data_prep.py
```

### 3.2 extract_sample_images.py

Extracts sample images from the MNIST dataset for visualization:
- Loads the MNIST dataset
- Selects representative samples from each digit class
- Saves the sample images to the data/mnist_samples directory
- Generates a grid visualization of the samples

Usage:
```bash
python scripts/extract_sample_images.py
```

### 3.3 train_cnn.py

Trains the CNN model on the MNIST dataset:
- Loads the preprocessed MNIST data
- Defines the CNN architecture
- Implements data augmentation
- Sets up early stopping and model checkpointing
- Trains the model and saves the best and final versions
- Generates training history visualizations

Usage:
```bash
python scripts/train_cnn.py
```

### 3.4 visualize_features.py

Generates visualizations of the CNN feature maps:
- Loads the trained model
- Selects sample input images
- Extracts feature maps from various layers
- Creates and saves visualizations of the feature maps
- Helps in understanding what patterns the CNN is detecting

Usage:
```bash
python scripts/visualize_features.py
```

### 3.5 analyze_predictions.py

Provides detailed analysis of the trained model's predictions:
- Generates comprehensive confusion matrix visualizations
- Creates classification metrics report with per-digit performance
- Visualizes prediction confidence distribution
- Identifies and visualizes the most commonly confused digit pairs
- Outputs a detailed text report of the analysis findings

Usage:
```bash
python scripts/analyze_predictions.py --model_path models/mnist_cnn_best.h5
```

The script will generate multiple visualizations in the figures directory and a detailed text report.

## 4. Common Parameters

Most scripts accept the following command-line parameters:
- `--data_dir`: Directory for the MNIST data (default: 'data/mnist')
- `--output_dir`: Directory for output files (default: varies by script)
- `--batch_size`: Batch size for processing (default: 32)
- `--random_seed`: Random seed for reproducibility (default: 42)

You can get help on the parameters for any script using:
```bash
python scripts/<script_name>.py --help
``` 