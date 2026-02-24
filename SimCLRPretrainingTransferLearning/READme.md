# SimCLR-Pretraining-Transfer-Learning-CIFAR10

## Description
This project demonstrates a self-supervised learning pipeline using SimCLR (Simple Framework for Contrastive Learning of Visual Representations). It pre-trains a ResNet18 backbone on the **Imagenette** dataset and subsequently fine-tunes the model on the **CIFAR-10** dataset for image classification.

## Key Components
1. **Pre-training**: 
   - Uses SimCLR with a contrastive loss function.
   - Dataset: Imagenette (a subset of ImageNet).
   - Model: ResNet18 (without the final classification layer).
   
2. **Fine-tuning**:
   - Adds a linear classification head to the pre-trained backbone.
   - Dataset: CIFAR-10.
   - Task: 10-class image classification.
