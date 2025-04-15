#!/bin/bash

# Check if data directories exist
if [ ! -d "/app/data/mnist" ]; then
    echo "Creating data directories..."
    mkdir -p /app/data/mnist
    mkdir -p /app/data/mnist_samples
fi

# Check if models directory exists
if [ ! -d "/app/models" ]; then
    echo "Creating models directory..."
    mkdir -p /app/models
fi

# Check if figures directory exists
if [ ! -d "/app/figures" ]; then
    echo "Creating figures directory..."
    mkdir -p /app/figures
fi

# Run Jupyter lab by default
echo "Starting Jupyter Lab..."
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
