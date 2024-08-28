import os

import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification

BATCH_SIZE = 512  # to adjust
image_folder = "./ml_exercise_therapanacea/val_img"  # Set this to your folder containing the images
output_file = "predictions.txt"
checkpoint_dir = "./checkpoint-175"  # Set this to your actual checkpoint directory


# Set the device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


device = get_device()
print(f"Using device: {device}")

# Load the model and feature extractor from a checkpoint

model = ViTForImageClassification.from_pretrained(checkpoint_dir)
model.to(device)
model.eval()

feature_extractor = ViTFeatureExtractor.from_pretrained(checkpoint_dir)

# Define the transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=feature_extractor.image_mean, std=feature_extractor.image_std
        ),
    ]
)


# Define the inference function with batch processing
def predict_images_batch(image_paths, batch_size=8):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = transform(image)
        images.append(image)

    images_tensor = torch.stack(images).to(device)

    with torch.no_grad():
        outputs = model(images_tensor)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_classes = torch.argmax(probabilities, dim=-1).cpu().tolist()

    return predicted_classes


# Inference on a folder of images with batch processing
def run_inference_on_folder(image_folder, output_file, batch_size=8):
    image_names = sorted(
        [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    )
    image_paths = [os.path.join(image_folder, img_name) for img_name in image_names]

    with open(output_file, "w") as f:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            predicted_classes = predict_images_batch(batch_paths, batch_size)

            for img_name, pred_class in zip(
                image_names[i : i + batch_size], predicted_classes
            ):
                f.write(f"{pred_class}\n")
                print(f"Processed {img_name} -> Class: {pred_class}")


# Run the inference

run_inference_on_folder(image_folder, output_file, batch_size=BATCH_SIZE)
