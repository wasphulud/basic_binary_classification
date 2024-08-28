# Binary Face Classification Project

## Overview

This project focuses on developing a robust model for binary classification of human face images. The primary goal is to accurately classify images into two categories: visible faces and occluded faces. The best-performing model will then be used to make predictions on a validation dataset.

## Project Structure

- **`project_exploration.ipynb`**: A Jupyter notebook for initial data exploration, preprocessing, and visualization. It documents key decisions and provides insight into the methodology used throughout the project.

- **`split_dataset.py`**: This script handles the splitting of the dataset into training and testing sets, ensuring balanced representation across classes to prevent biased evaluations.

- **`train_model.py`**: This script sets up and trains a Vision Transformer (ViT) model. It allows for flexible configuration of parameters such as batch size, seed values, and computing resources (GPU/CPU). The model is evaluated on the testing set, with training metrics logged for analysis.

- **`infer.py`**: This script loads the best-performing model and performs inference on new images, predicting class labels based on the binary classification task.

- **`output_logs.txt`**: Contains logs and metrics from the training and evaluation phases, including accuracy, loss curves, and other performance indicators. These logs are critical for diagnosing model performance and informing further fine-tuning efforts.

- **`best_model.py`**: This script includes the code for identifying the best-performing model based on metrics such as accuracy and F1 score. It is designed to parse the logs and extract the best model checkpoint for further inference or deployment.

- **`predictions.txt`**: This file contains the class predictions made by the best-performing model on the validation dataset.

## Key Observations and Decisions

- **Data Exploration**: Initial exploration suggested that the dataset clusters images based on whether the person's face is fully visible or occluded (e.g., by glasses, hats, or hands). This observation is preliminary and requires further validation.

- **Class Imbalance**: The dataset is significantly imbalanced, with non-occluded images appearing over seven times more frequently than occluded ones. To address this, we applied weighted sampling with replacement to oversample the minority class (occluded faces). Data augmentation was limited to this oversampling technique and mirroring to avoid over-complication.

- **Validation Dataset**: We created a balanced validation dataset of 1,000 images (500 per class) by randomly sampling from the original training set.

- **Model Choice**: We employed a transformer-based architecture, specifically the `google/vit-base-patch16-224-in21k` model. Images were upscaled from 64x64 to 224x224 pixels to match the model's input requirements.

- **Training Pipeline**: The training process was initially developed on a MacBook and later migrated to a cloud-based A100 GPU for faster execution. We used a batch size of 512 and increased the learning rate by a factor of 8 compared to the original rate due to the larger batch size.

- **Training Strategy**: We logged the training loss at each step, evaluated the model on the test dataset every 5 steps, and saved model checkpoints every 25 steps. Training was stopped at 800 steps after observing a performance plateau around step 220. While further fine-tuning (e.g., adjusting the learning rate) could improve accuracy, this was deferred for potential future work.

- **Model Evaluation**: A custom function was developed to parse the logs and identify the best model checkpoint based on accuracy and F1 score. This model was then evaluated on the validation dataset, with predictions saved in the `predictions.txt` file.

## Conclusion and Future work

The project developed a binary classification model for face images using a transformer-based approach achieving a 90+% accuracy and f1-score. Despite challenges such as class imbalance, the model achieved reasonable performance, with further tuning possible for enhanced accuracy (carefully tweak the learning rate, further data augmentation, better choice of the model, ensembling methods, etc...). The results and logs could provide a solid foundation for future improvements and applications.

This project developed a binary classification model for face images using a transformer-based architecture, achieving over 90% accuracy and F1-score. Despite the challenges posed by class imbalance, the model demonstrated strong performance. **However, there is still room for improvement. Future efforts could focus on fine-tuning the learning rate, exploring additional data augmentation techniques, experimenting with different model architectures, or implementing ensembling methods to further enhance accuracy and robustness.**

The results and detailed logs generated during this project provide a valuable foundation for future enhancements and potential applications.
