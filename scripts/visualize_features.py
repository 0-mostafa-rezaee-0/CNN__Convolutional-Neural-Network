#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize feature maps from a trained CNN model.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize CNN feature maps')
    parser.add_argument('--data_dir', type=str, default='data/mnist',
                        help='Directory with MNIST data')
    parser.add_argument('--model_path', type=str, default='models/mnist_cnn_best.h5',
                        help='Path to the trained model')
    parser.add_argument('--figures_dir', type=str, default='figures',
                        help='Directory to save visualizations')
    parser.add_argument('--samples_per_class', type=int, default=1,
                        help='Number of samples per class to visualize')
    return parser.parse_args()


def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def load_data(data_dir):
    """Load test data for visualization."""
    # Try to load preprocessed data
    if os.path.exists(os.path.join(data_dir, 'x_test.npy')):
        x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    else:
        # Load from Keras
        from tensorflow.keras.datasets import mnist
        
        (_, _), (x_test, y_test) = mnist.load_data()
        
        # Normalize and reshape
        x_test = x_test.astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # One-hot encode (not needed for this script, but keeping consistent)
        from tensorflow.keras.utils import to_categorical
        y_test = to_categorical(y_test, 10)
    
    return x_test, y_test


def get_sample_images(x_test, y_test, samples_per_class):
    """Get sample images for each digit class."""
    samples = []
    
    # Get indices for each class
    y_test_classes = np.argmax(y_test, axis=1)
    
    # For each digit class (0-9)
    for digit in range(10):
        # Find indices for this digit
        indices = np.where(y_test_classes == digit)[0]
        
        # Take the first few samples
        for i in range(min(samples_per_class, len(indices))):
            samples.append(x_test[indices[i]])
    
    return np.array(samples)


def create_feature_extractor(model, layer_names):
    """Create a model that outputs the feature maps for specified layers."""
    outputs = [model.get_layer(name).output for name in layer_names]
    return tf.keras.Model(inputs=model.inputs, outputs=outputs)


def plot_feature_maps(sample_images, feature_maps, layer_names, figures_dir):
    """Plot and save feature map visualizations."""
    n_layers = len(layer_names)
    n_samples = len(sample_images)
    
    plt.figure(figsize=(15, 10))
    
    # For each sample image
    for s in range(n_samples):
        # Plot the original image
        plt.subplot(n_samples, n_layers + 1, s * (n_layers + 1) + 1)
        plt.imshow(sample_images[s].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if s == 0:
            plt.title('Input Image')
        
        # For each layer
        for l in range(n_layers):
            # Get feature map for this layer and sample
            feature_map = feature_maps[l][s]
            
            # Create a composite image of feature maps
            # Take the first 16 channels or all if fewer
            n_channels = min(16, feature_map.shape[-1])
            
            # Calculate grid dimensions
            grid_size = int(np.ceil(np.sqrt(n_channels)))
            
            # Create a composite image
            composite = np.zeros((
                grid_size * feature_map.shape[0],
                grid_size * feature_map.shape[1]
            ))
            
            # Fill the composite image with feature maps
            for i in range(n_channels):
                row = i // grid_size
                col = i % grid_size
                channel = feature_map[:, :, i]
                
                # Normalize the channel for better visualization
                channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-7)
                
                # Add to composite
                composite[
                    row * feature_map.shape[0]:(row + 1) * feature_map.shape[0],
                    col * feature_map.shape[1]:(col + 1) * feature_map.shape[1]
                ] = channel
            
            # Plot the composite image
            plt.subplot(n_samples, n_layers + 1, s * (n_layers + 1) + l + 2)
            plt.imshow(composite, cmap='viridis')
            plt.axis('off')
            if s == 0:
                plt.title(f'Layer: {layer_names[l]}\n({feature_map.shape[-1]} filters)')
    
    plt.suptitle('CNN Feature Maps Visualization', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(figures_dir, 'feature_maps.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Feature maps visualization saved to {output_file}")


def main():
    """Main function to visualize CNN feature maps."""
    args = parse_args()
    
    # Create output directory
    create_directory(args.figures_dir)
    
    # Load data
    x_test, y_test = load_data(args.data_dir)
    
    # Get sample images
    sample_images = get_sample_images(x_test, y_test, args.samples_per_class)
    
    # Load the trained model
    model = load_model(args.model_path)
    
    # Define layers to visualize (choose convolutional layers)
    layer_names = []
    for layer in model.layers:
        if 'conv' in layer.name:
            layer_names.append(layer.name)
    
    # If too many layers, select a few representative ones
    if len(layer_names) > 4:
        selected_indices = np.linspace(0, len(layer_names) - 1, 4, dtype=int)
        layer_names = [layer_names[i] for i in selected_indices]
    
    print(f"Visualizing feature maps for layers: {layer_names}")
    
    # Create feature extractor model
    feature_extractor = create_feature_extractor(model, layer_names)
    
    # Get feature maps for sample images
    feature_maps = feature_extractor.predict(sample_images)
    
    # Ensure feature_maps is a list
    if not isinstance(feature_maps, list):
        feature_maps = [feature_maps]
    
    # Plot and save feature map visualizations
    plot_feature_maps(sample_images, feature_maps, layer_names, args.figures_dir)


if __name__ == '__main__':
    main() 