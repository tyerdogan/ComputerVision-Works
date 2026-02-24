## 📌 Project: Self-Supervised Classification & Transfer Learning with PyTorch Lightning

This project explores advanced image classification techniques (SeaLife Classification), comparing **Self-Supervised Learning (SimCLR)** against **Transfer Learning (ResNet50)**, all implemented using the **PyTorch Lightning** framework for scalable and modular deep learning.

### 🚀 Key Technologies
- **PyTorch Lightning**: The core framework used to structure the training loops, separating research code from engineering logic.
- **SimCLR (Self-Supervised Learning)**: Implemented a contrastive learning approach to learn visual representations without labels.
- **ResNet Architectures**: Utilized ResNet18 for SimCLR and ResNet50 for Transfer Learning.
- **Weights & Biases (WandB)**: Integrated for experiment tracking, hyperparameter tuning, and visualizing model predictions.

### 📂 Project Structure

#### 1. Data Pipeline
- **Dataset**: Custom dataset extracted and split into Train (80%), Validation (10%), and Test (10%).
- **Augmentations**: Applied strong augmentations (ColorJitter, RandomGrayscale, GaussianBlur) crucial for SimCLR and robust fine-tuning.

#### 2. Phase 1: Self-Supervised Learning (SimCLR)
- **Pre-training**: Trained a ResNet18 encoder to maximize agreement between differently augmented views of the same image.
- **Fine-tuning**: A linear classifier was attached to the frozen, pre-trained encoder to evaluate the quality of learned features.
- **LightningModule**: Custom `SimCLR` class inheriting from `pl.LightningModule` to handle the contrastive loss calculation.

#### 3. Phase 2: Supervised Transfer Learning (ResNet50)
- **Strategy**: Switched to a larger, pre-trained ResNet50 (ImageNet weights) to improve accuracy.
- **Optimization**: 
  - **Focal Loss**: To handle potential class imbalances.
  - **Schedulers**: `OneCycleLR` and `CosineAnnealingWarmRestarts` for better convergence.
  - **Regularization**: Dropout and Weight Decay tuning.

#### 4. Results & Visualization
- **Test Evaluation**: Final models evaluated on the unseen test set.
- **Inference**: Visualized predictions vs. ground truth labels using WandB and Matplotlib.
