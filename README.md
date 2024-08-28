# Binary Face Classification Project

This project aims to develop a model for binary classification using images of human faces. The primary objective is to train a robust model that can accurately classify images into one of two categories. Once trained, the best-performing model will be used to make predictions on a validation dataset.

## Project Structure Overview

- **`project_exploration.ipynb`**: A Jupyter notebook designed for data exploration, high-level data processing, and visualization. This notebook also documents key decisions made during the project's development phases, such as data preprocessing, training configurations, and model evaluation. It's a great starting point for understanding the overall approach and methodology behind the project.

- **`split_dataset.py`**: This module handles the splitting of the original dataset into training and testing sets. Its primary goal is to ensure that both sets have balanced representation across all classes, facilitating unbiased performance evaluation. It ensures that each label class is equally distributed in both the training and testing datasets.

- **`train_model.py`**: This module initializes a Vision Transformer (ViT) model and sets up the training pipeline. Users can adjust various parameters such as seed values (for reproducibility), batch size, and machine resources (e.g., GPU/CPU). The model is trained on the training split, and its performance is evaluated on the testing split. This module also logs essential metrics for analysis.

- **`infer.py`**: This module loads the best-performing trained model and performs inference on new, unseen images, predicting their class labels based on the binary classification task.

- **`output_logs.txt`**: This file contains logs and metrics generated during training and evaluation, such as accuracy, loss curves, and other performance indicators on the training and testing sets. These logs are helpful for diagnosing the model's performance and further fine-tuning.

## Key Features

- **Data Exploration**: Visualize and explore the dataset through interactive Jupyter notebooks.
- **Balanced Dataset Splitting**: Ensure even class distribution between training and testing datasets for fair evaluation, taking into account the imbalanced nature of the original dataset.
- **Configurable Training Pipeline**: Flexibly adjust training parameters such as batch size, seed, and resource allocation.
- **State-of-the-Art Model**: Utilize the Vision Transformer (ViT) for high-performance image classification.
- **Inference Capabilities**: Leverage the trained model to infer labels on new data.

This structure allows for a streamlined approach to binary face classification, from data preparation to model training and inference.
