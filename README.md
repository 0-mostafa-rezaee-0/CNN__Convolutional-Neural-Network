# CNN_MNIST_Dockerized

<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">CNN-Based MNIST Digit Recognition</div>

<div style="text-align:center; margin-bottom:30px;">A comprehensive project for digit recognition using Convolutional Neural Networks in Docker</div>

## Table of Contents 
<details>
  <summary><a href="#1-project-overview"><i><b>1. Project Overview</b></i></a></summary>
</details>
&nbsp;

<details>
  <summary><a href="#2-educational-objectives"><i><b>2. Educational Objectives</b></i></a></summary>
</details>
&nbsp;

<details>
  <summary><a href="#3-prerequisites"><i><b>3. Prerequisites</b></i></a></summary>
</details>
&nbsp;

<details>
  <summary><a href="#4-project-structure"><i><b>4. Project Structure</b></i></a></summary>
</details>
&nbsp;

<details>
  <summary><a href="#5-getting-started"><i><b>5. Getting Started</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#51-docker-setup">5.1. Docker Setup</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#52-running-the-project">5.2. Running the Project</a><br>
  </div>
</details>
&nbsp;

<details>
  <summary><a href="#6-project-components"><i><b>6. Project Components</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#61-data-preparation">6.1. Data Preparation</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#62-cnn-architecture">6.2. CNN Architecture</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#63-model-training">6.3. Model Training</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#64-visualizations">6.4. Visualizations</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#65-utility-scripts">6.5. Utility Scripts</a><br>
  </div>
</details>
&nbsp;

<details>
  <summary><a href="#7-learning-exercises"><i><b>7. Learning Exercises</b></i></a></summary>
</details>
&nbsp;

<details>
  <summary><a href="#8-common-issues"><i><b>8. Common Issues</b></i></a></summary>
</details>
&nbsp;

<details>
  <summary><a href="#9-resources-for-further-learning"><i><b>9. Resources for Further Learning</b></i></a></summary>
</details>
&nbsp;

<details>
  <summary><a href="#10-license"><i><b>10. License</b></i></a></summary>
</details>
&nbsp;

## 1. Project Overview

This project demonstrates implementing a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. The entire project is containerized using Docker to ensure a consistent environment for development and execution.

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (60,000 for training and 10,000 for testing), each 28×28 pixels in size. Using a CNN, we'll build a model that can accurately classify these digits with over 99% accuracy.

## 2. Educational Objectives

This project serves as a learning tool for:

- Understanding CNN architectures and their advantages over standard ANNs for image processing
- Implementing and training a CNN using TensorFlow/Keras
- Visualizing and interpreting CNN feature maps
- Comparing performance between CNN and traditional ANN approaches
- Learning Docker containerization for reproducible machine learning environments

## 3. Prerequisites

- Docker and Docker Compose installed on your system
- Basic knowledge of Python and neural networks
- Understanding of image classification concepts

## 4. Project Structure

```
.
+---Dockerfile                <-- Docker configuration for environment setup
+---docker-compose.yml        <-- Docker Compose configuration for service definition
+---requirements.txt          <-- Python dependencies for the project
+---start.sh                  <-- Startup script for Docker container
|
+---data/                     <-- Data directory
|   +---mnist/                <-- Raw and processed MNIST data (generated at runtime)
|   +---mnist_samples/        <-- Sample images extracted from MNIST for visualization
|   +---README.md             <-- Documentation for the dataset
|
+---figures/                  <-- Visualizations and plots
|   +---mnist_samples.png     <-- Grid of sample MNIST digits
|   +---confusion_matrix.png  <-- Model performance visualization
|   +---training_history.png  <-- Training/validation metrics over time
|   +---feature_maps.png      <-- CNN feature map visualizations
|   +---prediction_samples.png <-- Examples of model predictions
|   +---README.md             <-- Documentation for visualizations
|
+---notebooks/                <-- Jupyter notebooks for interactive learning
|   +---CNN_MNIST-data.ipynb  <-- Main notebook for the project
|   +---exploratory_analysis.ipynb <-- Basic data exploration notebook
|   +---README.md             <-- Documentation for notebooks
|
+---models/                   <-- Saved model files
|   +---mnist_cnn_best.h5     <-- Best model based on validation accuracy
|   +---mnist_cnn_final.h5    <-- Final trained model
|   +---README.md             <-- Documentation for models
|
+---scripts/                  <-- Python scripts
    +---data_prep.py          <-- Download and preprocess MNIST data
    +---extract_sample_images.py <-- Extract sample images for visualization
    +---train_cnn.py          <-- Train the CNN model
    +---visualize_features.py <-- Generate feature map visualizations
    +---README.md             <-- Documentation for scripts
```

## 5. Getting Started

### 5.1 Docker Setup

Clone this repository and navigate to the project directory:

```bash
git clone <repository-url>
cd CNN_MNIST_Dockerized
```

### 5.2 Running the Project

Build and run the Docker container:

```bash
docker-compose up
```

Access Jupyter Lab in your browser at: http://localhost:8888

#### 5.2.1 Running the Complete Pipeline

To run the entire CNN MNIST pipeline from data preparation to visualization:

```bash
# Inside the Docker container
./run_pipeline.sh
```

This script will:
1. Prepare the MNIST dataset
2. Extract and visualize sample images
3. Train the CNN model with data augmentation
4. Visualize feature maps from the trained model

Additionally, you can run an optional comparison between CNN and traditional ANN:

```bash
# Run the pipeline with CNN vs ANN comparison
./run_pipeline.sh --compare
```

This will add an additional step that trains both model types and generates a detailed performance comparison.

Alternatively, you can run each script individually as described in the [Scripts Documentation](scripts/README.md).

## 6. Project Components

### 6.1 Data Preparation

The MNIST dataset is automatically downloaded and preprocessed when running the scripts. The data is normalized and split into training, validation, and test sets.

### 6.2 CNN Architecture

Our CNN model consists of:
- Multiple convolutional layers with appropriate filter sizes
- Max pooling layers for downsampling
- Dropout for regularization
- Dense layers for classification
- Softmax output for 10 digit classes (0-9)

### 6.3 Model Training

The model is trained with:
- Data augmentation (rotation, zoom, shift)
- Categorical cross-entropy loss
- Adam optimizer
- Early stopping and model checkpointing
- Validation accuracy and loss monitoring

### 6.4 Visualizations

The project includes visualizations for:
- Sample digits from the MNIST dataset
- Training and validation metrics
- Confusion matrix of model predictions
- CNN feature maps
- Examples of correct and incorrect predictions

### 6.5 Utility Scripts

The project includes utility scripts for maintenance and monitoring:
- Project structure validation
- Project status summary generation

These scripts make it easier to verify the project's integrity and get a quick overview of the current state. For details, see the [Utils Documentation](utils/README.md).

## 7. Learning Exercises

1. Modify the CNN architecture and observe how it affects performance
2. Compare the CNN results with a simple multi-layer perceptron (MLP)
   - Use the built-in comparison tool: `python utils/model_comparison.py`
   - Analyze why CNNs perform better for image classification tasks
3. Experiment with different data augmentation techniques
4. Visualize feature maps from different layers to understand what the CNN learns

## 8. Common Issues

- **Docker Memory Issues**: If Docker crashes, try increasing the memory allocation in Docker settings
- **Training Performance**: For faster training, ensure Docker has access to GPU resources if available
- **Jupyter Lab Connection**: If you can't connect to Jupyter Lab, check that port 8888 is not in use by another application

## 9. Resources for Further Learning

- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [Deep Learning Book by Ian Goodfellow](https://www.deeplearningbook.org/)

## 10. License

This project is licensed under the MIT License - see the LICENSE file for details.