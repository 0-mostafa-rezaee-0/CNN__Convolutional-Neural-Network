#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a summary of the CNN MNIST project status.
This script collects information about data, models, and visualizations.
"""

import os
import json
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def format_size(size_bytes):
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def get_data_summary():
    """Get summary of the data directory."""
    data_dir = 'data'
    data_summary = {
        'mnist_data_available': os.path.exists('data/mnist/x_train.npy'),
        'sample_images_available': os.path.exists('data/mnist_samples'),
        'data_files': []
    }
    
    if data_summary['mnist_data_available']:
        data_files = [
            'x_train.npy', 'y_train.npy', 
            'x_test.npy', 'y_test.npy',
            'x_val.npy', 'y_val.npy'  # These might not exist
        ]
        
        for file in data_files:
            file_path = os.path.join('data/mnist', file)
            if os.path.exists(file_path):
                size = format_size(os.path.getsize(file_path))
                modified = datetime.datetime.fromtimestamp(
                    os.path.getmtime(file_path)
                ).strftime('%Y-%m-%d %H:%M:%S')
                
                data_summary['data_files'].append({
                    'name': file,
                    'size': size,
                    'modified': modified
                })
    
    # Count sample images
    if data_summary['sample_images_available']:
        sample_count = 0
        for digit in range(10):
            digit_dir = os.path.join('data/mnist_samples', str(digit))
            if os.path.exists(digit_dir):
                sample_count += len([f for f in os.listdir(digit_dir) if f.endswith('.png')])
        
        data_summary['sample_count'] = sample_count
    
    return data_summary

def get_model_summary():
    """Get summary of the trained models."""
    models_dir = 'models'
    model_summary = {
        'best_model_available': os.path.exists('models/mnist_cnn_best.h5'),
        'final_model_available': os.path.exists('models/mnist_cnn_final.h5'),
        'models': []
    }
    
    model_files = ['mnist_cnn_best.h5', 'mnist_cnn_final.h5']
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        if os.path.exists(model_path):
            size = format_size(os.path.getsize(model_path))
            modified = datetime.datetime.fromtimestamp(
                os.path.getmtime(model_path)
            ).strftime('%Y-%m-%d %H:%M:%S')
            
            # Try to load model and get performance
            try:
                model = load_model(model_path)
                model_info = {
                    'name': model_file,
                    'size': size,
                    'modified': modified,
                    'layers': len(model.layers),
                    'parameters': model.count_params()
                }
                
                # Try to evaluate if test data is available
                if os.path.exists('data/mnist/x_test.npy') and os.path.exists('data/mnist/y_test.npy'):
                    x_test = np.load('data/mnist/x_test.npy')
                    y_test = np.load('data/mnist/y_test.npy')
                    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
                    model_info['test_accuracy'] = float(accuracy)
                    model_info['test_loss'] = float(loss)
                
                model_summary['models'].append(model_info)
            except Exception as e:
                model_summary['models'].append({
                    'name': model_file,
                    'size': size,
                    'modified': modified,
                    'error': str(e)
                })
    
    return model_summary

def get_visualization_summary():
    """Get summary of visualization files."""
    figures_dir = 'figures'
    visualization_summary = {
        'available_visualizations': []
    }
    
    if os.path.exists(figures_dir):
        for file in os.listdir(figures_dir):
            if file.endswith('.png'):
                file_path = os.path.join(figures_dir, file)
                size = format_size(os.path.getsize(file_path))
                modified = datetime.datetime.fromtimestamp(
                    os.path.getmtime(file_path)
                ).strftime('%Y-%m-%d %H:%M:%S')
                
                visualization_summary['available_visualizations'].append({
                    'name': file,
                    'size': size,
                    'modified': modified
                })
    
    return visualization_summary

def generate_summary():
    """Generate a complete project summary."""
    summary = {
        'generated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data': get_data_summary(),
        'models': get_model_summary(),
        'visualizations': get_visualization_summary()
    }
    
    return summary

def print_summary(summary):
    """Print summary in a readable format."""
    print("===== CNN MNIST Project Summary =====")
    print(f"Generated at: {summary['generated_at']}\n")
    
    # Data summary
    print("=== Data ===")
    data = summary['data']
    if data['mnist_data_available']:
        print("✅ MNIST data is available")
        for file in data['data_files']:
            print(f"  - {file['name']} ({file['size']}, modified: {file['modified']})")
    else:
        print("❌ MNIST data is not available")
    
    if data['sample_images_available']:
        print(f"✅ Sample images are available ({data.get('sample_count', 'unknown')} samples)")
    else:
        print("❌ Sample images are not available")
    
    # Model summary
    print("\n=== Models ===")
    models = summary['models']
    if models['best_model_available'] or models['final_model_available']:
        for model in models['models']:
            print(f"✅ {model['name']} ({model['size']}, modified: {model['modified']})")
            if 'parameters' in model:
                print(f"  - Layers: {model['layers']}, Parameters: {model['parameters']:,}")
            if 'test_accuracy' in model:
                print(f"  - Test accuracy: {model['test_accuracy']:.4f}, Test loss: {model['test_loss']:.4f}")
            if 'error' in model:
                print(f"  - Error loading model: {model['error']}")
    else:
        print("❌ No trained models available")
    
    # Visualization summary
    print("\n=== Visualizations ===")
    visualizations = summary['visualizations']
    if visualizations['available_visualizations']:
        for viz in visualizations['available_visualizations']:
            print(f"✅ {viz['name']} ({viz['size']}, modified: {viz['modified']})")
    else:
        print("❌ No visualizations available")

def main():
    """Main function to generate and display project summary."""
    try:
        summary = generate_summary()
        print_summary(summary)
        
        # Save summary to JSON file
        with open('project_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to project_summary.json")
    except Exception as e:
        print(f"Error generating summary: {e}")

if __name__ == "__main__":
    main() 