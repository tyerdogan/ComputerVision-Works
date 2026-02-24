
## Transformers, Detection & Segmentation
---

## Project Overview
This project explores advanced computer vision techniques through two distinct tasks: image classification using Vision Transformers and object detection/segmentation using YOLO and GroundedSAM.

## Table of Contents
1. [Task 1: CIFAR-10 Classification with Transformers](#task-1-cifar-10-classification-with-transformers)
2. [Task 2: Raccoon Detection & Segmentation](#task-2-raccoon-detection--segmentation)
3. [Installation](#installation)

## Task 1: CIFAR-10 Classification with Transformers
**Goal:** Optimize a Transformer-based network with Convolutional Patch Embeddings to achieve >80% validation accuracy on the CIFAR-10 dataset.

**Key Components:**
-   **Architecture:** Custom Transformer blocks with Multi-head Attention and FeedForward layers.
-   **Embeddings:** Convolutional Patch Embedding and Sinusoidal Position Embeddings.
-   **Optimization:** Uses AdamW optimizer, Dropouts, and Learning Rate Schedulers (ReduceLROnPlateau).

## Task 2: Raccoon Detection & Segmentation
**Goal:** Build a robust pipeline to detect and segment raccoons in images using a combination of zero-shot labeling and supervised training.

**Pipeline:**
1.  **Auto-Labeling:** Use **GroundedSAM** (Grounding DINO + SAM) to auto-label a raw dataset of raccoon images based on text prompts ("raccoon").
2.  **Data Filtering:** Interactive UI to manually verify and clean the auto-generated labels.
3.  **Model Training:** Fine-tune **YOLO11n** (detection) and **YOLO11n-seg** (segmentation) models on the labeled dataset.
4.  **Inference & Export:** Run inference on test images and export trained models to **ONNX** format for deployment.
