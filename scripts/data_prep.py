#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation script for MNIST dataset.
Downloads, preprocesses, and saves the MNIST dataset.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MNIST data preparation script')
    parser.add_argument('--data_dir', type=str, default='data/mnist',
                        help='Directory to store the dataset')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='Fraction of training data to use for validation')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def load_and_preprocess_data(validation_split, random_seed):
    """
    Load and preprocess the MNIST dataset.
    
    Returns:
        tuple: (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape images to include channel dimension (28x28x1)
    x_train_full = x_train_full.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Split training data into train and validation sets
    if validation_split > 0:
        np.random.seed(random_seed)
        indices = np.random.permutation(len(x_train_full))
        val_size = int(len(x_train_full) * validation_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        x_train = x_train_full[train_indices]
        y_train = y_train_full[train_indices]
        x_val = x_train_full[val_indices]
        y_val = y_train_full[val_indices]
    else:
        x_train = x_train_full
        y_train = y_train_full
        x_val = None
        y_val = None
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    if y_val is not None:
        y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"Training set: {x_train.shape[0]} images")
    if x_val is not None:
        print(f"Validation set: {x_val.shape[0]} images")
    print(f"Test set: {x_test.shape[0]} images")
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def save_data(data_dir, x_train, y_train, x_val, y_val, x_test, y_test):
    """Save preprocessed data to disk."""
    print(f"Saving preprocessed data to {data_dir}...")
    np.save(os.path.join(data_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
    
    if x_val is not None and y_val is not None:
        np.save(os.path.join(data_dir, 'x_val.npy'), x_val)
        np.save(os.path.join(data_dir, 'y_val.npy'), y_val)
    
    print("Data preprocessing and saving complete.")


def main():
    """Main function to load, preprocess, and save the MNIST dataset."""
    args = parse_args()
    
    # Create data directory
    create_directory(args.data_dir)
    
    # Load and preprocess the data
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_data(
        args.validation_split, args.random_seed
    )
    
    # Save the preprocessed data
    save_data(args.data_dir, x_train, y_train, x_val, y_val, x_test, y_test)


if __name__ == '__main__':
    main() 