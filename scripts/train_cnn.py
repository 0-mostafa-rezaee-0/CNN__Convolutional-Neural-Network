#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a CNN model on the MNIST dataset.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from sklearn.metrics import confusion_matrix


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CNN model on MNIST')
    parser.add_argument('--data_dir', type=str, default='data/mnist',
                        help='Directory with MNIST data')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--figures_dir', type=str, default='figures',
                        help='Directory to save visualizations')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Maximum number of epochs for training')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_data_augmentation', action='store_true',
                        help='Use data augmentation during training')
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
        
        # Check if validation data exists
        if (os.path.exists(os.path.join(data_dir, 'x_val.npy')) and
            os.path.exists(os.path.join(data_dir, 'y_val.npy'))):
            x_val = np.load(os.path.join(data_dir, 'x_val.npy'))
            y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        else:
            x_val = None
            y_val = None
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
        
        # No validation set in this case
        x_val = None
        y_val = None
    
    print(f"Training set: {x_train.shape[0]} images")
    if x_val is not None:
        print(f"Validation set: {x_val.shape[0]} images")
    print(f"Test set: {x_test.shape[0]} images")
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def create_cnn_model():
    """Create and compile the CNN model."""
    model = Sequential([
        # First convolutional layer
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
               input_shape=(28, 28, 1)),
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


def create_data_generator():
    """Create a data generator for data augmentation."""
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )


def train_model(model, x_train, y_train, x_val, y_val, args):
    """Train the CNN model."""
    # Create model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.models_dir, 'mnist_cnn_best.h5'),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    # Create early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=args.patience,
        restore_best_weights=True,
        verbose=1
    )
    
    callbacks = [checkpoint_callback, early_stopping]
    
    # Use validation data if available, otherwise use validation_split
    if x_val is not None and y_val is not None:
        validation_data = (x_val, y_val)
        validation_split = 0.0
    else:
        validation_data = None
        validation_split = 0.1
    
    # Train with or without data augmentation
    if args.use_data_augmentation:
        print("Using data augmentation...")
        datagen = create_data_generator()
        datagen.fit(x_train)
        
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=args.batch_size),
            epochs=args.epochs,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks
        )
    else:
        history = model.fit(
            x_train, y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks
        )
    
    return model, history


def save_model(model, models_dir):
    """Save the final trained model."""
    model_path = os.path.join(models_dir, 'mnist_cnn_final.h5')
    model.save(model_path)
    print(f"Final model saved to {model_path}")


def evaluate_model(model, x_test, y_test):
    """Evaluate the model on test data."""
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    return test_loss, test_accuracy


def plot_training_history(history, figures_dir):
    """Plot and save training history visualization."""
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(figures_dir, 'training_history.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history visualization saved to {output_file}")


def plot_confusion_matrix(model, x_test, y_test, figures_dir):
    """Plot and save confusion matrix visualization."""
    # Get model predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the figure
    output_file = os.path.join(figures_dir, 'confusion_matrix.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix visualization saved to {output_file}")


def plot_prediction_samples(model, x_test, y_test, figures_dir, num_samples=10):
    """Plot and save examples of model predictions."""
    # Get model predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Find correct and incorrect predictions
    correct_indices = np.where(y_pred_classes == y_true_classes)[0]
    incorrect_indices = np.where(y_pred_classes != y_true_classes)[0]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot correct predictions
    for i in range(min(num_samples, len(correct_indices))):
        idx = correct_indices[i]
        plt.subplot(4, 5, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_true_classes[idx]}\nPred: {y_pred_classes[idx]}", 
                  color='green', fontsize=10)
        plt.axis('off')
    
    # Plot incorrect predictions
    for i in range(min(num_samples, len(incorrect_indices))):
        idx = incorrect_indices[i]
        plt.subplot(4, 5, i + num_samples + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_true_classes[idx]}\nPred: {y_pred_classes[idx]}", 
                  color='red', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Correct (top) and Incorrect (bottom) Predictions', y=0.98)
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(figures_dir, 'prediction_samples.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Prediction samples visualization saved to {output_file}")


def main():
    """Main function to train and evaluate CNN model."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # Create output directories
    create_directory(args.models_dir)
    create_directory(args.figures_dir)
    
    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.data_dir)
    
    # Create and compile model
    model = create_cnn_model()
    model.summary()
    
    # Train model
    model, history = train_model(model, x_train, y_train, x_val, y_val, args)
    
    # Save the final model
    save_model(model, args.models_dir)
    
    # Evaluate model
    evaluate_model(model, x_test, y_test)
    
    # Create visualizations
    plot_training_history(history, args.figures_dir)
    plot_confusion_matrix(model, x_test, y_test, args.figures_dir)
    plot_prediction_samples(model, x_test, y_test, args.figures_dir)


if __name__ == '__main__':
    main() 