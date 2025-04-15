#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare performance between CNN and traditional ANN models on MNIST dataset.
This script trains both model types and generates performance comparisons.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare CNN and ANN models on MNIST')
    parser.add_argument('--data_dir', type=str, default='data/mnist',
                        help='Directory with MNIST data')
    parser.add_argument('--figures_dir', type=str, default='figures',
                        help='Directory to save comparison visualizations')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Maximum number of epochs for training')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def load_data(data_dir):
    """Load preprocessed MNIST data."""
    print("Loading preprocessed MNIST data...")
    
    # Check if preprocessed data exists
    if (os.path.exists(os.path.join(data_dir, 'x_train.npy')) and
        os.path.exists(os.path.join(data_dir, 'y_train.npy'))):
        
        # Load preprocessed data
        x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    else:
        # Use Keras to load the data directly
        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.utils import to_categorical
        
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape images to include channel dimension (28x28x1)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # One-hot encode labels
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
    
    print(f"Training set: {x_train.shape[0]} images")
    print(f"Test set: {x_test.shape[0]} images")
    
    return x_train, y_train, x_test, y_test


def create_ann_model(input_shape):
    """Create a traditional ANN model for MNIST classification."""
    # Flatten the input shape
    flattened_size = np.prod(input_shape)
    
    model = Sequential([
        # Flatten layer to convert 2D input to 1D
        Flatten(input_shape=input_shape),
        
        # Hidden layers
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        
        # Output layer
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_cnn_model(input_shape):
    """Create a CNN model for MNIST classification."""
    model = Sequential([
        # First convolutional layer
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
               input_shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second convolutional layer
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    """Train the model and measure training time."""
    # Create early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    
    # Measure training time
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    # Calculate number of parameters
    parameter_count = model.count_params()
    
    return history, test_accuracy, training_time, parameter_count


def plot_comparison(ann_history, cnn_history, ann_test_acc, cnn_test_acc, 
                   ann_time, cnn_time, ann_params, cnn_params, figures_dir):
    """Plot comparison between ANN and CNN models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot training accuracy
    axes[0, 0].plot(ann_history.history['accuracy'], label='ANN Train')
    axes[0, 0].plot(ann_history.history['val_accuracy'], label='ANN Validation')
    axes[0, 0].plot(cnn_history.history['accuracy'], label='CNN Train')
    axes[0, 0].plot(cnn_history.history['val_accuracy'], label='CNN Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot training loss
    axes[0, 1].plot(ann_history.history['loss'], label='ANN Train')
    axes[0, 1].plot(ann_history.history['val_loss'], label='ANN Validation')
    axes[0, 1].plot(cnn_history.history['loss'], label='CNN Train')
    axes[0, 1].plot(cnn_history.history['val_loss'], label='CNN Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot test accuracy comparison
    model_types = ['ANN', 'CNN']
    accuracies = [ann_test_acc, cnn_test_acc]
    
    axes[1, 0].bar(model_types, accuracies, color=['blue', 'orange'])
    axes[1, 0].set_title('Test Accuracy Comparison')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim([min(accuracies) - 0.05, 1.0])
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        axes[1, 0].text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    # Plot training time comparison
    times = [ann_time, cnn_time]
    
    axes[1, 1].bar(model_types, times, color=['blue', 'orange'])
    axes[1, 1].set_title('Training Time Comparison')
    axes[1, 1].set_ylabel('Time (seconds)')
    
    # Add time values on top of bars
    for i, t in enumerate(times):
        axes[1, 1].text(i, t + 1, f'{t:.2f}s', ha='center')
    
    # Add parameter counts as annotations
    plt.figtext(0.5, 0.01, f'ANN Parameters: {ann_params:,} | CNN Parameters: {cnn_params:,}', 
               ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.suptitle('CNN vs Traditional ANN on MNIST Dataset', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_file = os.path.join(figures_dir, 'cnn_vs_ann_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison visualization saved to {output_file}")
    
    # Create a summary table
    summary = {
        'ANN': {
            'Test Accuracy': f'{ann_test_acc:.4f}',
            'Training Time': f'{ann_time:.2f}s',
            'Parameters': f'{ann_params:,}'
        },
        'CNN': {
            'Test Accuracy': f'{cnn_test_acc:.4f}',
            'Training Time': f'{cnn_time:.2f}s',
            'Parameters': f'{cnn_params:,}'
        },
        'Difference': {
            'Test Accuracy': f'{cnn_test_acc - ann_test_acc:.4f}',
            'Training Time': f'{cnn_time - ann_time:.2f}s',
            'Parameters': f'{cnn_params - ann_params:,}'
        }
    }
    
    return summary


def print_summary_table(summary):
    """Print a formatted summary table."""
    print("\n===== CNN vs ANN Performance Summary =====")
    print(f"{'Metric':<20} {'ANN':<15} {'CNN':<15} {'Difference':<15}")
    print("-" * 65)
    
    for metric in ['Test Accuracy', 'Training Time', 'Parameters']:
        print(f"{metric:<20} {summary['ANN'][metric]:<15} {summary['CNN'][metric]:<15} {summary['Difference'][metric]:<15}")


def main():
    """Main function to run the comparison."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # Create output directory
    create_directory(args.figures_dir)
    
    # Load data
    x_train, y_train, x_test, y_test = load_data(args.data_dir)
    
    print("\n===== Training Traditional ANN Model =====")
    # Create and train ANN model
    ann_model = create_ann_model((28, 28, 1))
    ann_model.summary()
    ann_history, ann_test_acc, ann_time, ann_params = train_model(
        ann_model, x_train, y_train, x_test, y_test, args.epochs, args.batch_size
    )
    
    print("\n===== Training CNN Model =====")
    # Create and train CNN model
    cnn_model = create_cnn_model((28, 28, 1))
    cnn_model.summary()
    cnn_history, cnn_test_acc, cnn_time, cnn_params = train_model(
        cnn_model, x_train, y_train, x_test, y_test, args.epochs, args.batch_size
    )
    
    # Plot comparison
    summary = plot_comparison(
        ann_history, cnn_history, ann_test_acc, cnn_test_acc, 
        ann_time, cnn_time, ann_params, cnn_params, args.figures_dir
    )
    
    # Print summary table
    print_summary_table(summary)


if __name__ == '__main__':
    main() 