# MNIST Neural Network: NumPy and PyTorch

## Overview

This project implements the same multilayer perceptron for MNIST in two ways:

1. From scratch with NumPy, including the forward pass, cross-entropy loss, backpropagation, and mini-batch SGD.
2. With PyTorch using built-in layers, automatic differentiation, and an SGD optimizer.

Both versions use the same dataset split, preprocessing, architecture, and main training hyperparameters. The notebook evaluates the models and compares their accuracy, convergence, runtime, and implementation approach.

## Objectives

- Implement and train a fully connected neural network without an automatic differentiation framework.
- Validate the manual gradient calculations through end-to-end MNIST training.
- Reproduce the architecture with PyTorch under comparable settings.
- Compare measured results from both implementations.

## Dataset and Preprocessing

The project uses the standard MNIST split:

- 60,000 training images
- 10,000 test images
- 28 x 28 grayscale pixels
- 10 output classes representing digits 0–9

Images are converted to `float32`, normalized from `[0, 255]` to `[0, 1]`, and flattened into 784-feature vectors. NumPy and PyTorch use the same preprocessed arrays and train/test split.

## Architecture

```text
Input image: 28 x 28
        |
Flatten: 784 features
        |
Linear: 784 -> 128
        |
      ReLU
        |
Linear: 128 -> 10
        |
Class logits / probabilities
```

The model contains 101,770 trainable parameters.

## Implementations

### NumPy from Scratch

The NumPy implementation includes:

- He-style weight initialization and zero biases
- Linear layers, ReLU, stable softmax, and cross-entropy
- Analytical gradients for both linear layers and ReLU
- Mini-batch SGD with data shuffling each epoch
- Sample-weighted running loss and accuracy tracking

All gradients and updates for `W1`, `b1`, `W2`, and `b2` are implemented explicitly.

### PyTorch

The PyTorch model uses:

- `nn.Linear(784, 128)`
- `nn.ReLU()`
- `nn.Linear(128, 10)`
- `nn.CrossEntropyLoss()`
- `torch.optim.SGD`
- `DataLoader` mini-batches with shuffled training data

PyTorch handles gradient computation through autograd. The training loop still performs the forward pass, loss calculation, optimizer update, and epoch metric aggregation explicitly. Device selection supports CUDA when available and falls back to CPU; the reported run used CPU.

## Training Setup

| Setting | Value |
|---|---:|
| Epochs | 10 |
| Batch size | 128 |
| Learning rate | 0.1 |
| Optimizer | Mini-batch SGD |
| Loss | Cross-entropy |
| Random seed | 42 |
| Reported device | CPU |

The runtime values below were measured in the executed notebook and are environment-dependent.

## Results

| Metric | NumPy | PyTorch |
|---|---:|---:|
| Final training accuracy | 97.44% | 97.38% |
| Test accuracy | 97.09% | 96.90% |
| Training time | 21.87 s | 15.50 s |
| Implementation approach | Manual forward pass, gradients, backpropagation, and updates | Built-in layers, autograd, loss, and optimizer |

Both implementations first reached at least 95% training accuracy in epoch 5. In the measured CPU run, PyTorch trained faster, while the NumPy model finished with slightly higher training and test accuracy.

## Training and Evaluation Visuals

### NumPy Training Curves

![NumPy training loss](results/training_loss_curve.png)

![NumPy training accuracy](results/training_accuracy_curve.png)

### MNIST Test Confusion Matrix

![NumPy confusion matrix](results/confusion_matrix.png)

### NumPy vs PyTorch Convergence

![NumPy vs PyTorch convergence](results/numpy_vs_pytorch_convergence.png)

Additional prediction examples are available in:

- [Correct predictions](results/correct_predictions.png)
- [Incorrect predictions](results/incorrect_predictions.png)

## Setup and Usage

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter from the project root:

```bash
jupyter notebook notebooks/mnist_from_scratch.ipynb
```

Run the notebook cells in order. MNIST is downloaded automatically into `data/` when it is not already present. Training and evaluation plots are written to `results/`.

## Project Structure

```text
.
├── data/
│   └── .gitkeep
├── notebooks/
│   └── mnist_from_scratch.ipynb
├── results/
│   ├── confusion_matrix.png
│   ├── correct_predictions.png
│   ├── incorrect_predictions.png
│   ├── numpy_vs_pytorch_convergence.png
│   ├── training_accuracy_curve.png
│   └── training_loss_curve.png
├── .gitignore
├── README.md
└── requirements.txt
```

The downloaded MNIST files and generated result images are excluded from version control by the current `.gitignore` rules.

## Key Takeaways

- A two-layer NumPy network with explicit backpropagation reached 97.09% test accuracy on MNIST.
- Under equivalent high-level training settings, both implementations crossed 95% training accuracy in the same epoch.
- PyTorch reduced training time from 21.87 seconds to 15.50 seconds in the measured CPU run.
- The NumPy version exposes each gradient and parameter update; the PyTorch version removes that manual gradient bookkeeping while preserving an explicit training loop.
