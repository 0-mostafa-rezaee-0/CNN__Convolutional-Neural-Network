<div style="font-size:2em; font-weight:bold; text-align:center; margin-top:20px;">Models Directory</div>

## 1. Overview

This directory stores the trained CNN models from the MNIST digit recognition project.

## 2. Directory Contents

```
models/
├── mnist_cnn_best.h5    # Best model saved during training (based on validation accuracy)
└── mnist_cnn_final.h5   # Final model after completing all training epochs
```

## 3. Model Details

### 3.1 Model Architecture

Both saved models follow this CNN architecture:
- Input layer (28×28×1 grayscale images)
- Multiple convolutional layers with ReLU activation
- Max pooling layers for downsampling
- Dropout layers for regularization
- Dense layers with ReLU activation
- Softmax output layer with 10 units (one per digit class)

### 3.2 Model Files

#### 3.2.1 Best Model (mnist_cnn_best.h5)
This model represents the weights that achieved the highest validation accuracy during training. It was saved using Keras' ModelCheckpoint callback.

#### 3.2.2 Final Model (mnist_cnn_final.h5)
This model contains the weights from the end of the training process after completing all epochs.

## 4. Loading Models

Models can be loaded using TensorFlow/Keras with:

```python
from tensorflow.keras.models import load_model

# Load the best model
model = load_model('models/mnist_cnn_best.h5')

# Or load the final model
model = load_model('models/mnist_cnn_final.h5')
```

## 5. Performance Metrics

The models are trained to achieve over 99% accuracy on the MNIST test set. Detailed metrics including precision, recall, and F1-score can be found in the evaluation notebook. 