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


# Set the device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


RANDOM_SEED = 1337


device = get_device()
torch.manual_seed(RANDOM_SEED)
torch.set_float32_matmul_precision("high")
print(f"Using device: {device}")


# Define logging callback
class LogStepCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        logs = kwargs.get("logs", {})
        logs["step"] = state.global_step
        if "learning_rate" in logs:
            logs["learning_rate"] = f"{logs['learning_rate']:.6f}"
        if "grad_norm" in logs:
            logs["grad_norm"] = f"{logs['grad_norm']:.4f}"
        log_str = ", ".join([f"{k}: {v}" for k, v in logs.items()])
        log_path = os.path.join(args.logging_dir, "training_log.txt")
        with open(log_path, "a") as f:
            f.write(log_str + "\n")


# Dataset preparation
def prepare_datasets(dataset_path, feature_extractor):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.Normalize(
                mean=feature_extractor.image_mean, std=feature_extractor.image_std
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, "train"), transform=transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, "test"), transform=transform
    )

    return train_dataset, test_dataset


# Sampler preparation
def prepare_sampler(train_dataset):
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_dataset.targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    return sampler


# Data collator function
def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


# Metric computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    return {
        "eval_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Main function
def main():
    dataset_path = "./ml_exercise_therapanacea"

    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )

    train_dataset, test_dataset = prepare_datasets(dataset_path, feature_extractor)
    sampler = prepare_sampler(train_dataset)

    BATCH_SIZE = 32  # 512

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=len(train_dataset.classes)
    ).to(device)

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

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
