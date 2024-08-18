# FashionMNIST Classification using ANN

This project aims to classify images of clothing items from the FashionMNIST dataset using an Artificial Neural Network (ANN). The model is built with three dense layers and is optimized using the Adam optimizer. The primary goal is to achieve high accuracy in classifying the images into one of the 10 fashion categories.

## Dataset

The FashionMNIST dataset is a widely used benchmark for image classification. It contains 60,000 training images and 10,000 test images, each of 28x28 pixels, representing 10 different clothing categories:

- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

Each image is grayscale, resulting in 784 features (28x28 pixels) per image.

## Model Architecture

The ANN model used in this project consists of the following layers:

- **Input Layer**: 784 neurons (28x28 pixel values)
- **Hidden Layer 1**: 500 neurons with ReLU activation
- **Hidden Layer 2**: 500 neurons with ReLU activation
- **Hidden Layer 3**: 500 neurons with ReLU activation
- **Output Layer**: 10 neurons (one for each clothing category) with softmax activation

### Hyperparameters:

- **Loss Function**: Cross Entropy Loss
- **Optimizer**: Adam
- **Input Size**: 784
- **Hidden Size**: 500
- **Number of Classes**: 10
- **Number of Epochs**: 10
- **Batch Size**: 32
- **Learning Rate**: 0.001

## Training and Evaluation

During training, the model's performance was evaluated on the training set, and the results were as follows:

- **Minimum Training Loss**: 0.0807

On the test set, the model achieved the following performance:

- **Test Accuracy**: 88.22%

These results indicate that the model is effective in classifying images of clothing items, though there is potential for further improvement through techniques such as hyperparameter tuning or model architecture adjustments.

## Requirements

To run the code, ensure you have the following packages installed:

- Python (>=3.6)
- PyTorch (for building the ANN)
- numpy
- matplotlib

You can install the required packages using:

```bash
pip install torch numpy matplotlib


## Contributors

- [M.S.R.Pavan](https://github.com/MSRpavan)

