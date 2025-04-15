#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and visualize sample images from the MNIST dataset.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract sample images from MNIST dataset')
    parser.add_argument('--data_dir', type=str, default='data/mnist',
                        help='Directory to load and store MNIST data')
    parser.add_argument('--samples_dir', type=str, default='data/mnist_samples',
                        help='Directory to store sample images')
    parser.add_argument('--figures_dir', type=str, default='figures',
                        help='Directory to store visualizations')
    parser.add_argument('--samples_per_class', type=int, default=5,
                        help='Number of samples to extract per class')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def load_mnist_data():
    """Load the MNIST dataset."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


def extract_samples(x_data, y_data, samples_per_class, random_seed):
    """
    Extract sample images from each class.
    
    Args:
        x_data: Image data
        y_data: Labels
        samples_per_class: Number of samples to extract per class
        random_seed: Random seed for reproducibility
        
    Returns:
        dict: Dictionary with class labels as keys and lists of sample images as values
    """
    np.random.seed(random_seed)
    samples = {}
    
    # Extract samples for each class (0-9)
    for digit in range(10):
        # Find indices of images for this digit
        indices = np.where(y_data == digit)[0]
        
        # Randomly select samples_per_class samples
        selected_indices = np.random.choice(indices, samples_per_class, replace=False)
        
        # Store the selected samples
        samples[digit] = [x_data[i] for i in selected_indices]
    
    return samples


def save_samples(samples, samples_dir):
    """
    Save extracted sample images to disk.
    
    Args:
        samples: Dictionary with class labels as keys and lists of sample images as values
        samples_dir: Directory to save samples
    """
    for digit, images in samples.items():
        digit_dir = os.path.join(samples_dir, str(digit))
        create_directory(digit_dir)
        
        for i, img in enumerate(images):
            plt.imsave(
                os.path.join(digit_dir, f"sample_{i}.png"),
                img,
                cmap='gray'
            )
    
    print(f"Saved sample images to {samples_dir}")


def create_grid_visualization(samples, figures_dir):
    """
    Create and save a grid visualization of sample images.
    
    Args:
        samples: Dictionary with class labels as keys and lists of sample images as values
        figures_dir: Directory to save visualization
    """
    samples_per_class = len(samples[0])
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot samples in a grid
    for i, digit in enumerate(range(10)):
        for j, image in enumerate(samples[digit]):
            plt.subplot(10, samples_per_class, i * samples_per_class + j + 1)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            if j == 0:
                plt.title(f"Digit: {digit}", fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(figures_dir, 'mnist_samples.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created grid visualization: {output_file}")


def main():
    """Main function to extract and visualize MNIST samples."""
    args = parse_args()
    
    # Create output directories
    create_directory(args.samples_dir)
    create_directory(args.figures_dir)
    
    # Load MNIST data
    x_train, y_train, x_test, y_test = load_mnist_data()
    
    # Extract sample images from each class
    samples = extract_samples(
        x_train, y_train, args.samples_per_class, args.random_seed
    )
    
    # Save samples to disk
    save_samples(samples, args.samples_dir)
    
    # Create grid visualization
    create_grid_visualization(samples, args.figures_dir)


if __name__ == '__main__':
    main() 