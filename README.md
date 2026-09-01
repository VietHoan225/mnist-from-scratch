# MNIST Neural Network From Scratch

A small deep learning project focused on understanding how a neural network works under the hood by implementing a multilayer perceptron from scratch with NumPy, then reproducing the same architecture with PyTorch for comparison.

## Architecture

The model uses a simple fully connected neural network:

```text
28x28 image
    ↓
Flatten to 784 features
    ↓
Linear: 784 → 128
    ↓
ReLU
    ↓
Linear: 128 → 10
    ↓
Softmax / Cross-Entropy
```

## Goals

- Load and preprocess the MNIST dataset
- Implement forward propagation manually with NumPy
- Implement backpropagation from scratch
- Train the network using mini-batch gradient descent
- Evaluate the model on the MNIST test set
- Re-implement the same architecture with PyTorch
- Compare the NumPy and PyTorch implementations

## Project Structure

```text
.
├── data/
├── notebooks/
│   └── mnist_from_scratch.ipynb
├── results/
├── .gitignore
├── README.md
└── requirements.txt
```

The MNIST dataset is downloaded automatically into the `data/` directory and is not committed to Git.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the notebook:

```bash
jupyter notebook notebooks/mnist_from_scratch.ipynb
```

## Current Progress

- [x] Project setup
- [x] MNIST data loading
- [x] Data normalization and flattening
- [x] Dataset visualization and shape inspection
- [x] NumPy forward propagation
- [ ] Backpropagation
- [ ] Training loop
- [ ] Model evaluation and visualization
- [ ] PyTorch implementation
- [ ] NumPy vs PyTorch comparison

## Dataset

This project uses the MNIST handwritten digit dataset:

- 60,000 training images
- 10,000 test images
- Image size: `28 x 28`
- 10 classes: digits `0-9`

Pixel values are normalized to the range `[0, 1]`. Images are flattened into 784-dimensional vectors for the fully connected neural network.
