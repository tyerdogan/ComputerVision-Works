# From Manual Backprop to CNN Optimization

## Overview
This assignment explores the fundamental mechanics of training neural networks and the practical aspects of improving model performance. It is divided into two main sections: understanding the internals of automatic differentiation and optimizing a Convolutional Neural Network (CNN) for image classification.

## Contents

### Part 1: Backprop Your Own Gradients
In this theoretical and practical section, we dive deep into the backpropagation algorithm.
- **Manual Backpropagation**: Manually derive and implement the backward pass for a Cross-Entropy loss function, verifying gradients against PyTorch's autograd engine.
- **Custom Autograd Function**: Implement a custom `torch.autograd.Function` for Cross-Entropy loss, optimizing the backward pass for efficiency.

### Part 2: Architecture Optimization
This section focuses on applied deep learning using the CIFAR-10 dataset.
- **Objective**: Design and train a CNN to achieve a validation accuracy of **>85%**.
- **Techniques Used**:
  - **Data Augmentation**: `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, etc., to improve generalization.
  - **Model Architecture**: Custom CNN with residual-like blocks, Batch Normalization, and Dropout.
  - **Optimization**: Usage of `AdamW` optimizer and `CosineAnnealingWarmRestarts` scheduler.
  - **Regularization**: Label smoothing and weight decay to prevent overfitting.
