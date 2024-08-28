import os
import random
import shutil

import tqdm

# Constants
NUM_IMAGES_PER_CLASS = 500  # Number of images per class to be allocated to the test set
RANDOM_SEED = 1337  # Seed for random operations to ensure reproducibility

# Paths
parent_folder = "./ml_exercise_therapanacea"  # Base directory for the project

# Original image folder and data labels file
original_folder = os.path.join(
    parent_folder, "train_img"
)  # Path to the folder containing the original images
labels_file = os.path.join(
    parent_folder, "label_train.txt"
)  # Path to the file containing image labels

# Read labels into a list
with open(labels_file, "r") as f:
    labels = [
        int(line.strip()) for line in f
    ]  # Read and convert each label to an integer

# Create a list of corresponding image filenames
image_files = [
    f"{i+1:06d}.jpg" for i in range(len(labels))
]  # Generate filenames like '000001.jpg', '000002.jpg', etc.
image_label_pairs = list(
    zip(image_files, labels)
)  # Pair each filename with its corresponding label

# Shuffle the image-label pairs to randomize their order
random.seed(RANDOM_SEED)  # Set the random seed for reproducibility
random.shuffle(image_label_pairs)  # Shuffle the list of (image, label) pairs

# Create folders for training and testing datasets, with sub-folders for each class
for case in ["train", "test"]:
    folder_name = os.path.join(
        parent_folder, case
    )  # Create a folder for 'train' and 'test'
    os.makedirs(folder_name, exist_ok=True)  # Create the folder if it doesn't exist
    for label in ["0", "1"]:  # Create sub-folders for the two classes (0 and 1)
        os.makedirs(os.path.join(folder_name, label), exist_ok=True)

# Initialize counters for the number of images moved to the test folders for each class
positif_label_counter = 0  # Counter for class '1'
negatif_label_counter = 0  # Counter for class '0'

# Distribute the images into train and test folders
for image, label in tqdm.tqdm(
    image_label_pairs
):  # Iterate over the shuffled image-label pairs
    if (
        negatif_label_counter >= NUM_IMAGES_PER_CLASS
        and positif_label_counter >= NUM_IMAGES_PER_CLASS
    ):
        # If both classes have reached the desired number of test images, move remaining images to the train folder
        img_destination_folder = os.path.join(parent_folder, "train", f"{label}")
    elif negatif_label_counter < NUM_IMAGES_PER_CLASS and label == 0:
        # If class '0' needs more test images, move the image to the test folder
        img_destination_folder = os.path.join(parent_folder, "test", f"{label}")
        negatif_label_counter += 1  # Increment the counter for class '0'
    elif positif_label_counter < NUM_IMAGES_PER_CLASS and label == 1:
        # If class '1' needs more test images, move the image to the test folder
        img_destination_folder = os.path.join(parent_folder, "test", f"{label}")
        positif_label_counter += 1  # Increment the counter for class '1'
    else:
        # If the test set is complete for a class, move the image to the train folder
        img_destination_folder = os.path.join(parent_folder, "train", f"{label}")

    # Move the image file to the determined destination folder
    shutil.move(
        os.path.join(original_folder, image),
        os.path.join(img_destination_folder, image),
    )
