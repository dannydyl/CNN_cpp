# Project : Convolutional Neural Networks for Object Recognition: Forward Propagation

## Overview

This project focuses on implementing the forward propagation phase of a Convolutional Neural Network (CNN) for object recognition, specifically using the CNN-AK model that identifies ten classes of objects in 32x32x3 color images. The project does not cover the training part (backpropagation) but focuses on the classification part using forward propagation.

## Objectives

- Understand and implement the forward propagation in CNNs.
- Get familiar with the handling of multi-dimensional arrays and tensors in the context of neural networks.
- Utilize provided synthetic data for filter weights/coefficients or use real trained data if available.

## CNN Architecture

The project involves implementing a simplified version of CNN-AK, which includes the following layers and operations:

1. **Input Layer**: Receives the 32x32x3 color images.
2. **Convolutional Layers**: Multiple convolutional layers with ReLU activations and max pooling.
3. **Fully Connected Layer**: Processes the output from the convolutional layers to predict the class probabilities.
4. **Softmax Layer**: Converts the output into probability distribution across ten classes.

## Implementation Details

- The ConvNet class should be implemented with methods to compute the output of each layer.
- Focus on the computational aspects of convolution, ReLU activation, max pooling, fully connected processing, and softmax calculation.
- Use the provided or generated synthetic data for the network's weights and biases.

## Project Stages

1. **Data Preparation**: Generate or use provided synthetic data for network initialization.
2. **Network Construction**: Implement the ConvNet class and its associated methods.
3. **Forward Propagation**: Sequentially compute and pass the data through all layers.
4. **Output Analysis**: Analyze the output probabilities to understand the network's behavior.

## Tools and Technologies

- C++ for the entire implementation.
- Understanding of basic linear algebra, calculus, and probability.
- Familiarity with neural network concepts, especially CNNs.

## Usage

Compile the provided C++ files and run the executable, following the instructions for inputting data or using the provided data sets.

## References

- ConvNetJS CIFAR-10 demo by Andrej Karpathy (provided as the baseline for understanding and implementation).
- CIFAR-10 and CIFAR-100 datasets for understanding the data format and object classes.

## Acknowledgments

- Prof. Murali Subbarao for providing the project framework and guidance.
- Andrej Karpathy for the ConvNetJS demo, which forms the basis of this project's CNN model.
