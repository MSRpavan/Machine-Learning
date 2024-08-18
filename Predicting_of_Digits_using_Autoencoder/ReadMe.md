# Predicting Digits using Autoencoder on MNIST

This project delves into the use of Autoencoders for digit recognition on the MNIST dataset. The goal is to compare the effectiveness of two distinct Autoencoder architectures—one based on fully connected (dense) layers and the other on Convolutional Neural Network (CNN) layers—in reconstructing and predicting handwritten digits.

## Dataset

The MNIST dataset is a classic benchmark in machine learning, containing 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is grayscale, with a resolution of 28x28 pixels, resulting in a feature vector of 784 dimensions when flattened.

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

## Project Overview

In this project, I implemented and trained two different Autoencoder models:

1. **Fully Connected Autoencoder**: Utilizes dense layers for both the encoder and decoder.
2. **Convolutional Autoencoder**: Employs convolutional layers in the encoder and transposed convolutional layers in the decoder.

The primary aim was to compare their performance in reconstructing the input images and to determine which architecture is better suited for this task.

## Model Architectures

### 1. Fully Connected Autoencoder

- **Encoder**:
  - 4 fully connected (dense) layers, each followed by a ReLU activation function.
- **Decoder**:
  - Mirrors the encoder with 4 fully connected layers.
  - The final layer applies a Sigmoid activation function to constrain the output values between 0 and 1, matching the pixel intensity range of the input images.
- **Input/Output Shape**: [N, 28, 28]

The fully connected Autoencoder flattens the input image into a 784-dimensional vector, processes it through several dense layers in the encoder, and then reconstructs the image in the decoder.

### 2. Convolutional Autoencoder (CNN)

- **Encoder**:
  - 2 Convolutional layers with `kernel_size=3`, `stride=2`, `padding=1`, each followed by a ReLU activation function.
  - 1 Convolutional layer with `kernel_size=7`, followed by ReLU activation.
- **Decoder**:
  - 1 Transposed Convolutional layer with `kernel_size=7`, followed by ReLU activation.
  - 2 Transposed Convolutional layers with `kernel_size=3`, `stride=2`, `padding=1`, `output_padding=1`, followed by ReLU activation.
  - The final layer applies a Sigmoid activation function.
- **Input/Output Shape**: [N, 28, 28]

The Convolutional Autoencoder retains the spatial structure of the image throughout the encoding and decoding process, leveraging convolutional operations to capture local patterns more effectively.

## Training Details

### Hyperparameters:

- **Batch Size**: 32
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (with a learning rate of 0.001)

### Performance Metrics:

- **Fully Connected Autoencoder**:
  - Minimum Training Loss: 0.0688
- **Convolutional Autoencoder (CNN)**:
  - Minimum Training Loss: 0.0044

### Conclusion: Comparing CNN and Fully Connected Layers

The results clearly indicate that the Convolutional Autoencoder significantly outperforms the Fully Connected Autoencoder. The CNN-based model achieved a much lower minimum loss, indicating that it was better at reconstructing the input images with higher fidelity.

#### Why CNN Performs Better:

- **Spatial Hierarchies**: CNNs are designed to capture spatial hierarchies in data, which is particularly beneficial for image processing tasks. The convolutional layers can effectively detect edges, textures, and other patterns that are spatially local, leading to better reconstruction and feature learning.
- **Parameter Efficiency**: Convolutional layers use fewer parameters compared to fully connected layers, making them more efficient for image data, where the local structure is important.
- **Better Generalization**: The ability of CNNs to generalize well from training data to unseen data makes them more robust for tasks like image reconstruction.

In contrast, fully connected layers treat every pixel as independent and do not inherently capture the spatial relationships in the data, making them less effective for this type of task.

## Requirements

To run this project, you will need the following Python packages:

- Python (>=3.6)
- PyTorch
- numpy
- matplotlib

You can install the necessary packages using:

```bash
pip install torch numpy matplotlib
## Contributors

- [M.S.R.Pavan](https://github.com/MSRpavan)
