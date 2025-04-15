#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze predictions made by the CNN model on MNIST test set.
This script provides detailed analysis of model predictions including:
- Confusion matrix
- Classification report
- Examples of correct and incorrect predictions
- Analysis of most confusing digit pairs
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze CNN model predictions on MNIST')
    parser.add_argument('--data_dir', type=str, default='data/mnist',
                        help='Directory with MNIST data')
    parser.add_argument('--model_path', type=str, default='models/mnist_cnn_best.h5',
                        help='Path to the trained model')
    parser.add_argument('--figures_dir', type=str, default='figures',
                        help='Directory to save analysis visualizations')
    parser.add_argument('--output_file', type=str, default='model_analysis.txt',
                        help='File to save text analysis results')
    return parser.parse_args()


def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def load_data(data_dir):
    """Load test data for analysis."""
    print("Loading test data...")
    
    # Check if preprocessed data exists
    if os.path.exists(os.path.join(data_dir, 'x_test.npy')):
        x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        return x_test, y_test
    else:
        print("Error: Test data not found. Run data_prep.py first.")
        exit(1)


def get_model_predictions(model, x_test):
    """Get model predictions on test data."""
    print("Generating model predictions...")
    
    # Get predicted probabilities
    y_pred_prob = model.predict(x_test)
    
    # Get predicted class indices
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    
    # Get prediction confidence (max probability)
    prediction_confidence = np.max(y_pred_prob, axis=1)
    
    return y_pred_prob, y_pred_classes, prediction_confidence


def analyze_confusion_matrix(y_true_classes, y_pred_classes, figures_dir):
    """Analyze and visualize confusion matrix in detail."""
    print("Analyzing confusion matrix...")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Normalize confusion matrix by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots (raw and normalized)
    plt.figure(figsize=(16, 7))
    
    # Plot raw confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Count)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Plot normalized confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(figures_dir, 'detailed_confusion_matrix.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved detailed confusion matrix to {output_file}")
    
    # Find most confused digit pairs
    confused_pairs = []
    for i in range(10):
        for j in range(10):
            if i != j:
                confused_pairs.append((i, j, cm[i, j], cm_norm[i, j]))
    
    # Sort by normalized confusion value
    confused_pairs.sort(key=lambda x: x[3], reverse=True)
    
    return cm, cm_norm, confused_pairs[:5]  # Return top 5 confused pairs


def analyze_classification_metrics(y_true_classes, y_pred_classes):
    """Generate and analyze classification report."""
    print("Analyzing classification metrics...")
    
    # Generate classification report
    report = classification_report(
        y_true_classes, y_pred_classes, 
        target_names=[f"Digit {i}" for i in range(10)],
        output_dict=True
    )
    
    # Extract metrics per class
    class_metrics = {}
    for i in range(10):
        class_name = f"Digit {i}"
        class_metrics[i] = {
            'precision': report[class_name]['precision'],
            'recall': report[class_name]['recall'],
            'f1-score': report[class_name]['f1-score'],
            'support': report[class_name]['support']
        }
    
    # Find best and worst performing classes
    best_class = max(range(10), key=lambda i: class_metrics[i]['f1-score'])
    worst_class = min(range(10), key=lambda i: class_metrics[i]['f1-score'])
    
    return report, class_metrics, best_class, worst_class


def visualize_prediction_confidence(prediction_confidence, y_true_classes, y_pred_classes, figures_dir):
    """Visualize prediction confidence distribution for correct and incorrect predictions."""
    print("Visualizing prediction confidence...")
    
    # Separate confidences for correct and incorrect predictions
    correct_mask = y_pred_classes == y_true_classes
    correct_confidence = prediction_confidence[correct_mask]
    incorrect_confidence = prediction_confidence[~correct_mask]
    
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.hist(correct_confidence, bins=50, alpha=0.7, label='Correct Predictions', color='green')
    plt.hist(incorrect_confidence, bins=50, alpha=0.7, label='Incorrect Predictions', color='red')
    
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Confidence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics as text
    stats_text = (
        f"Correct Predictions: {len(correct_confidence)}\n"
        f"Incorrect Predictions: {len(incorrect_confidence)}\n"
        f"Avg. Confidence (Correct): {np.mean(correct_confidence):.4f}\n"
        f"Avg. Confidence (Incorrect): {np.mean(incorrect_confidence):.4f}"
    )
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Save the figure
    output_file = os.path.join(figures_dir, 'prediction_confidence.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved prediction confidence visualization to {output_file}")


def visualize_confused_digit_pairs(confused_pairs, x_test, y_true_classes, y_pred_classes, figures_dir):
    """Visualize examples of the most confused digit pairs."""
    print("Visualizing confused digit pairs...")
    
    plt.figure(figsize=(15, len(confused_pairs) * 3))
    
    for i, (true_digit, pred_digit, count, rate) in enumerate(confused_pairs):
        # Find examples of this confusion
        confused_indices = np.where((y_true_classes == true_digit) & 
                                   (y_pred_classes == pred_digit))[0]
        
        # If there are examples, show up to 5
        n_examples = min(5, len(confused_indices))
        
        for j in range(n_examples):
            plt.subplot(len(confused_pairs), 5, i * 5 + j + 1)
            idx = confused_indices[j]
            plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
            plt.title(f"True: {true_digit}, Pred: {pred_digit}")
            plt.axis('off')
        
        # Add label for this row
        plt.figtext(0.01, 1 - (i + 0.5) / len(confused_pairs), 
                  f"True {true_digit} → Pred {pred_digit} (Count: {count}, Rate: {rate:.2f})",
                  va="center", ha="left", fontsize=14)
    
    plt.suptitle('Examples of Most Confused Digit Pairs', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    output_file = os.path.join(figures_dir, 'confused_digit_pairs.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confused digit pairs visualization to {output_file}")


def create_analysis_report(report, class_metrics, best_class, worst_class, confused_pairs, 
                          correct_rate, output_file):
    """Create and save a detailed analysis report."""
    print("Generating analysis report...")
    
    with open(output_file, 'w') as f:
        f.write("===== CNN MNIST Model Analysis Report =====\n\n")
        
        # Overall performance
        f.write("OVERALL PERFORMANCE\n")
        f.write("===================\n")
        f.write(f"Accuracy: {report['accuracy']:.4f}\n")
        f.write(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}\n")
        f.write(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}\n")
        f.write(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}\n")
        f.write(f"Correct Predictions: {correct_rate * 100:.2f}%\n\n")
        
        # Per-class performance
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("=====================\n")
        f.write(f"{'Digit':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
        f.write("-" * 50 + "\n")
        
        for i in range(10):
            metrics = class_metrics[i]
            f.write(f"{i:<6} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                   f"{metrics['f1-score']:<10.4f} {metrics['support']:<10}\n")
        
        f.write("\n")
        
        # Best and worst performing classes
        f.write("PERFORMANCE EXTREMES\n")
        f.write("===================\n")
        f.write(f"Best performing digit: {best_class} "
               f"(F1-Score: {class_metrics[best_class]['f1-score']:.4f})\n")
        f.write(f"Worst performing digit: {worst_class} "
               f"(F1-Score: {class_metrics[worst_class]['f1-score']:.4f})\n\n")
        
        # Most confused digit pairs
        f.write("MOST CONFUSED DIGIT PAIRS\n")
        f.write("========================\n")
        for i, (true_digit, pred_digit, count, rate) in enumerate(confused_pairs):
            f.write(f"{i+1}. True {true_digit} predicted as {pred_digit}: "
                   f"{count} times ({rate:.2%})\n")
    
    print(f"Analysis report saved to {output_file}")


def main():
    """Main function to analyze model predictions."""
    args = parse_args()
    
    # Create output directory
    create_directory(args.figures_dir)
    
    # Load test data
    x_test, y_test = load_data(args.data_dir)
    
    # Convert one-hot encoded labels to class indices
    y_true_classes = np.argmax(y_test, axis=1)
    
    try:
        # Load trained model
        print(f"Loading model from {args.model_path}...")
        model = load_model(args.model_path)
        
        # Get model predictions
        y_pred_prob, y_pred_classes, prediction_confidence = get_model_predictions(model, x_test)
        
        # Calculate correct prediction rate
        correct_mask = y_pred_classes == y_true_classes
        correct_rate = np.mean(correct_mask)
        print(f"Model accuracy: {correct_rate:.4f}")
        
        # Analyze confusion matrix
        cm, cm_norm, confused_pairs = analyze_confusion_matrix(
            y_true_classes, y_pred_classes, args.figures_dir
        )
        
        # Analyze classification metrics
        report, class_metrics, best_class, worst_class = analyze_classification_metrics(
            y_true_classes, y_pred_classes
        )
        
        # Visualize prediction confidence
        visualize_prediction_confidence(
            prediction_confidence, y_true_classes, y_pred_classes, args.figures_dir
        )
        
        # Visualize confused digit pairs
        visualize_confused_digit_pairs(
            confused_pairs, x_test, y_true_classes, y_pred_classes, args.figures_dir
        )
        
        # Create and save analysis report
        create_analysis_report(
            report, class_metrics, best_class, worst_class, 
            confused_pairs, correct_rate, args.output_file
        )
        
        print("Model analysis completed successfully.")
        
    except Exception as e:
        print(f"Error analyzing model: {e}")
        exit(1)


if __name__ == '__main__':
    main() 