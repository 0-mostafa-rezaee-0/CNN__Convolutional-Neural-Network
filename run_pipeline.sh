#!/bin/bash

# CNN MNIST Pipeline Runner
# This script runs the entire pipeline for the CNN MNIST project:
# 1. Data preparation
# 2. Extract sample images
# 3. Train the CNN model
# 4. Visualize feature maps
# 5. (Optional) Compare CNN with traditional ANN

# Default settings
RUN_COMPARISON=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --compare)
      RUN_COMPARISON=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--compare]"
      echo "  --compare: Run CNN vs ANN comparison after the main pipeline"
      exit 1
      ;;
  esac
done

echo "===== CNN MNIST Pipeline ====="
echo "Starting pipeline execution..."

# Step 1: Data preparation
echo -e "\n[1/4] Preparing MNIST data..."
python scripts/data_prep.py
if [ $? -ne 0 ]; then
    echo "Error: Data preparation failed."
    exit 1
fi

# Step 2: Extract sample images
echo -e "\n[2/4] Extracting sample images for visualization..."
python scripts/extract_sample_images.py
if [ $? -ne 0 ]; then
    echo "Error: Sample image extraction failed."
    exit 1
fi

# Step 3: Train the CNN model
echo -e "\n[3/4] Training the CNN model..."
python scripts/train_cnn.py --use_data_augmentation
if [ $? -ne 0 ]; then
    echo "Error: Model training failed."
    exit 1
fi

# Step 4: Visualize feature maps
echo -e "\n[4/4] Visualizing CNN feature maps..."
python scripts/visualize_features.py
if [ $? -ne 0 ]; then
    echo "Error: Feature map visualization failed."
    exit 1
fi

echo -e "\n===== Main Pipeline completed successfully! ====="

# Step 5 (Optional): Run CNN vs ANN comparison
if [ "$RUN_COMPARISON" = true ]; then
    echo -e "\n===== Running CNN vs ANN Comparison ====="
    echo "This will train both a CNN and a traditional ANN model for comparison..."
    python utils/model_comparison.py --epochs 10
    if [ $? -ne 0 ]; then
        echo "Error: Model comparison failed."
        exit 1
    fi
    echo -e "\n===== Comparison completed successfully! ====="
fi

echo "Results are available in the following directories:"
echo "- Models: ./models/"
echo "- Visualizations: ./figures/"
echo "- Processed Data: ./data/" 