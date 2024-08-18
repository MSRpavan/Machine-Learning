# Plant Disease Classification using ResNet-9

This project focuses on classifying plant diseases using images. Leveraging the ResNet-9 architecture, the model is trained to identify 49 different classes of plant diseases and healthy states. The dataset used for this project is sourced from Kaggle.

## Dataset

The dataset used in this project is the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset), which contains images of leaves categorized into 49 different classes, including both diseases and healthy states. 

### Classes:

Some of the classes in the dataset include:

- Cherry_(including_sour)___Powdery_mildew
- Tomato___Spider_mites Two-spotted_spider_mite
- Grape___Esca_(Black_Measles)
- Chili__healthy
- Corn_(maize)___Common_rust_
- Potato___Late_blight
- Apple___Apple_scab
- Strawberry___healthy
- Tomato___Early_blight
- Corn_(maize)___Northern_Leaf_Blight
- ...and 39 other classes.

Each class consists of numerous images that depict either a specific disease or a healthy plant.

## Project Overview

The main objective of this project is to build a deep learning model using ResNet-9 architecture to accurately classify images into their respective disease or healthy categories. The model is trained on a dataset that includes a variety of plant diseases, providing a challenging classification task due to the similarities between some of the disease symptoms.

## Model Architecture

### ResNet-9

ResNet-9 is a simplified version of the ResNet architecture, known for its effectiveness in image classification tasks. The architecture includes:

- **Convolutional Layers**: 
  - Convolutional layers with `kernel_size=5`, `padding=2`.
  - Batch Normalization using `nn.BatchNorm2d(out_channels)`.
  - Max Pooling with `MaxPool2d(2)`.
- **Fully Connected Layers**:
  - The dense layer includes `MaxPool2d(4)` followed by a flattening layer.
  - ReLU activation function is used throughout the network.

### Model Parameters

- **Total Parameters**: 18,259,753
- **Trainable Parameters**: 18,259,753

### Training Details

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Best Training Loss**: 0.1441
- **Training Accuracy**: 95.12%
- **Best Test Loss**: 0.1775
- **Test Accuracy**: 94.65%

The model was trained using the CrossEntropyLoss function, which is suitable for multi-class classification tasks. The Adam optimizer was used for training, providing efficient gradient descent optimization.

## Results

The ResNet-9 model achieved impressive results on this challenging dataset:

- **Training Accuracy**: 95.12%
- **Testing Accuracy**: 94.65%
- **Training Loss**: 0.1441
- **Testing Loss**: 0.1775

These results indicate that the model is highly effective at classifying plant diseases, with only a slight drop in accuracy when moving from the training to the testing phase.

## Conclusion

The ResNet-9 architecture proved to be a powerful tool for plant disease classification, achieving high accuracy on both training and testing datasets. The slight difference between training and testing accuracy suggests that the model generalizes well to unseen data, making it a reliable option for real-world applications.

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

