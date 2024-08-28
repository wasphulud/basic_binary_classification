import os

import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Constants
BATCH_SIZE = 512  # Batch size for processing images, adjust as needed
image_folder = (
    "./ml_exercise_therapanacea/val_img"  # Folder containing the images for inference
)
output_file = "predictions.txt"  # File to save the predictions
checkpoint_dir = "./checkpoint-175"  # Directory containing the model checkpoint


# Function to determine the device for computation (GPU, MPS, or CPU)
def get_device():
    if torch.cuda.is_available():
        return "cuda"  # Use CUDA (GPU) if available
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Use MPS (Apple Silicon) if available
    else:
        return "cpu"  # Default to CPU if no GPU or MPS is available


# Set the device and display it
device = get_device()
print(f"Using device: {device}")

# Load the pre-trained model and feature extractor from the checkpoint directory
model = ViTForImageClassification.from_pretrained(checkpoint_dir)
model.to(device)
model.eval()  # Set the model to evaluation mode

feature_extractor = ViTFeatureExtractor.from_pretrained(checkpoint_dir)

# Define the image transformation pipeline
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(
            mean=feature_extractor.image_mean, std=feature_extractor.image_std
        ),  # Normalize images using the mean and std from the feature extractor
    ]
)


# Function to predict classes for a batch of images
def predict_images_batch(image_paths, batch_size=8):
    images = []

    # Process each image in the batch
    for image_path in image_paths:
        image = Image.open(image_path)  # Open the image
        image = transform(image)  # Apply transformations
        images.append(image)

    # Stack images into a single tensor and move to the appropriate device
    images_tensor = torch.stack(images).to(device)

    # Perform inference without calculating gradients
    with torch.no_grad():
        outputs = model(images_tensor)

    # Apply softmax to obtain probabilities and get the predicted class for each image
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_classes = torch.argmax(probabilities, dim=-1).cpu().tolist()

    return predicted_classes


# Function to run inference on all images in the specified folder
def run_inference_on_folder(image_folder, output_file, batch_size=8):
    # Get sorted list of image filenames (only .jpg files)
    image_names = sorted(
        [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    )
    image_paths = [
        os.path.join(image_folder, img_name) for img_name in image_names
    ]  # Get full paths to images

    # Open the output file to write predictions
    with open(output_file, "w") as f:
        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[
                i : i + batch_size
            ]  # Get the current batch of image paths
            predicted_classes = predict_images_batch(
                batch_paths, batch_size
            )  # Predict classes for the batch

            # Write the predicted classes to the output file and print results
            for img_name, pred_class in zip(
                image_names[i : i + batch_size], predicted_classes
            ):
                f.write(f"{pred_class}\n")  # Write the predicted class to the file
                print(
                    f"Processed {img_name} -> Class: {pred_class}"
                )  # Print the result to the console


# Run the inference on the folder of images
run_inference_on_folder(image_folder, output_file, batch_size=BATCH_SIZE)
