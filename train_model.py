import os

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from transformers import (
    Trainer,
    TrainingArguments,
    ViTFeatureExtractor,
    ViTForImageClassification,
)

torch.manual_seed(1337)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# Define the dataset path
dataset_path = (
    # "/Users/aimans/Projects/basic_binary_classification/overfitting_dataset"
    "/Users/aimans/Projects/basic_binary_classification/ml_exercise_therapanacea"
)

# Define the feature extractor (preprocessing) for ViT
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

# Define the image transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=feature_extractor.image_mean, std=feature_extractor.image_std
        ),
    ]
)

# Load the dataset
train_dataset = datasets.ImageFolder(
    os.path.join(dataset_path, "train"), transform=transform
)
test_dataset = datasets.ImageFolder(
    os.path.join(dataset_path, "test"), transform=transform
)

# Calculate the number of samples for each class
class_counts = np.bincount(train_dataset.targets)
class_weights = 1.0 / class_counts

# Assign a weight to each sample
sample_weights = [class_weights[label] for label in train_dataset.targets]

# Create a WeightedRandomSampler
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define the model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(train_dataset.classes),  # Adjust the number of classes
)
model.to(device)  # Move the model to the device


from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {
        "eval_accuracy": acc,  # Make sure this key is 'eval_accuracy'
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=3,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",  # Change to "wandb" or "tensorboard" for logging
)


# Load the datasets into Hugging Face's Dataset format
def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch]).to(
        device
    )  # Move tensors to device
    labels = torch.tensor([item[1] for item in batch]).to(
        device
    )  # Move labels to device
    return {"pixel_values": pixel_values, "labels": labels}


# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()
