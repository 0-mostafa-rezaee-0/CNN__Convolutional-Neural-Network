# CNN-Based MNIST Digit Recognition Project Prompt

## Task Overview

Create a comprehensive, well-documented project for MNIST digit recognition using Convolutional Neural Networks (CNN) within a Docker container environment. The project should follow a clear, educational structure suitable for students learning about CNNs, deep learning, and containerization.

## Project Requirements

### Core Functionality
1. Download and preprocess the MNIST dataset
2. Create a CNN model architecture suitable for digit recognition
3. Train the model with appropriate techniques (early stopping, model checkpoints)
4. Visualize training progress and results
5. Save and evaluate the trained model
6. Provide interactive notebooks for exploration

### Technical Specifications
1. Create a Docker-based development environment with all necessary dependencies
2. Use TensorFlow/Keras for implementing the CNN
3. Develop reusable Python scripts for each major function
4. Generate comprehensive visualizations for model understanding
5. Document all components thoroughly with README files

### Project Structure
Follow this specific directory structure:

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

## Specific Implementation Details

### 1. Docker Environment
- Base on Python 3.9 or higher
- Include TensorFlow, Keras, NumPy, Pandas, Matplotlib, Scikit-learn, and Seaborn
- Configure Jupyter Notebook access on port 8888
- Include a startup script to simplify container launch

### 2. CNN Architecture
- Create a model with:
  - Input layer accepting 28×28×1 grayscale images
  - Multiple convolutional layers with appropriate filter sizes
  - Max pooling layers for downsampling
  - Dropout for regularization
  - Dense layers for classification
  - Softmax output for 10 digit classes
- Use ReLU activation for convolutional and dense layers

### 3. Training Approach
- Implement data augmentation (rotation, zoom, shift)
- Use categorical cross-entropy loss
- Apply Adam optimizer
- Implement early stopping and model checkpointing
- Monitor validation accuracy and loss
- Save the best model during training

### 4. Visualization Requirements
- Generate confusion matrix heatmap
- Plot training and validation accuracy/loss
- Visualize CNN feature maps to show learned features
- Create a grid of sample digits from each class
- Show examples of correct and incorrect predictions

### 5. Documentation Standards
- Each directory must have a comprehensive README.md file
- READMEs should include:
  - Title and overview
  - Detailed table of contents with collapsible sections when appropriate
  - File descriptions and purposes
  - Usage instructions
  - Implementation details
- Document code with detailed comments
- Maintain consistent styling across all documentation

## Expected Project Structure for README Files

### Main README.md
- Title displayed as `<h1>` in a center-aligned div
- Table of contents with collapsible sections for items with subheadings
- Sections including Project Overview, Educational Objectives, Prerequisites, Project Structure, Getting Started, Project Components, Learning Exercises, Common Issues, Resources for Further Learning, and License

### Directory-Specific READMEs
- Follow the same styling as the main README
- Include specific details about the files in that directory
- Provide usage examples relevant to the directory's purpose
- Feature tables of contents appropriate to each file's length and complexity

## Educational Focus
The project should serve as a learning tool for:
- Understanding CNN architectures and their advantages over standard ANNs for image processing
- Visualizing and interpreting CNN feature maps
- Comparing performance between CNN and traditional ANN approaches
- Best practices for model training, validation, and testing
- Docker containerization for reproducible machine learning environments

## Deliverables
1. Complete project structure with all files and directories
2. Working Docker container with all dependencies
3. Executable Python scripts for all core functionality
4. Interactive Jupyter notebooks with explanations
5. Comprehensive documentation in README files
6. Visualization outputs for model understanding
7. Trained CNN models with >99% accuracy on MNIST

## Comparison with ANN Model
The project should explicitly highlight:
- Performance differences between CNN and ANN approaches
- Why CNNs work better for image recognition tasks
- Computational trade-offs between the two approaches
- Feature extraction capabilities of convolutional layers
- Visualization of learned features at different layers 