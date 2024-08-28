import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    ViTFeatureExtractor,
    ViTForImageClassification,
)


# Helper function to determine the device for computation (GPU, MPS, or CPU)
def get_device():
    if torch.cuda.is_available():
        return "cuda"  # Use CUDA (GPU) if available
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Use MPS (Apple Silicon) if available
    else:
        return "cpu"  # Default to CPU if no GPU or MPS is available


# Set a random seed for reproducibility and configure precision
RANDOM_SEED = 1337
device = get_device()
torch.manual_seed(RANDOM_SEED)
torch.set_float32_matmul_precision("high")
print(f"Using device: {device}")


# Custom callback class for logging training steps
class LogStepCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        logs = kwargs.get("logs", {})
        logs["step"] = state.global_step

        # Format logs for learning rate and gradient norm
        if "learning_rate" in logs:
            logs["learning_rate"] = f"{logs['learning_rate']:.6f}"
        if "grad_norm" in logs:
            logs["grad_norm"] = f"{logs['grad_norm']:.4f}"

        # Write logs to a file
        log_str = ", ".join([f"{k}: {v}" for k, v in logs.items()])
        log_path = os.path.join(args.logging_dir, "training_log.txt")
        with open(log_path, "a") as f:
            f.write(log_str + "\n")


# Function to prepare datasets for training and testing
def prepare_datasets(dataset_path, feature_extractor):
    # Define transformations for the dataset
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.RandomHorizontalFlip(p=0.3),  # Apply random horizontal flip
            transforms.Normalize(
                mean=feature_extractor.image_mean, std=feature_extractor.image_std
            ),  # Normalize images based on feature extractor's mean and std
        ]
    )

    # Load training and testing datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, "train"), transform=transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, "test"), transform=transform
    )

    return train_dataset, test_dataset


# Function to prepare a weighted sampler for balanced sampling in training
def prepare_sampler(train_dataset):
    class_counts = np.bincount(train_dataset.targets)  # Count instances of each class
    class_weights = (
        1.0 / class_counts
    )  # Calculate class weights inversely proportional to class frequency
    sample_weights = [
        class_weights[label] for label in train_dataset.targets
    ]  # Assign weights to each sample

    # Create a WeightedRandomSampler to handle imbalanced datasets
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    return sampler


# Function to collate data into batches for the data loader
def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])  # Stack image tensors
    labels = torch.tensor([item[1] for item in batch])  # Stack labels
    return {"pixel_values": pixel_values, "labels": labels}


# Function to compute evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # Get predicted class labels
    accuracy = accuracy_score(labels, preds)  # Calculate accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )  # Calculate precision, recall, and F1 score for binary classification

    return {
        "eval_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Main function to set up and run the training process
def main():
    dataset_path = "./ml_exercise_therapanacea"

    # Load the feature extractor from a pre-trained Vision Transformer (ViT) model
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )

    # Prepare datasets and the sampler
    train_dataset, test_dataset = prepare_datasets(dataset_path, feature_extractor)
    sampler = prepare_sampler(train_dataset)

    BATCH_SIZE = 32  # Define batch size for training and evaluation

    # Load the pre-trained ViT model for image classification
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=len(train_dataset.classes)
    ).to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=10,
        evaluation_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=25,
        eval_on_start=True,
        logging_dir="./logs",
        log_level="info",
        logging_steps=1,
        learning_rate=5e-5 * 8,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        report_to="",
        # fp16=True, # Enable if possible
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[LogStepCallback()],
    )

    # Start training and evaluation
    trainer.train()
    trainer.evaluate()


# Entry point for the script
if __name__ == "__main__":
    main()
