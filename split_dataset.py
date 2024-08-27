import os
import random
import shutil

import tqdm

NUM_IMAGES_PER_CLASS = 5000
RANDOM_SEED = 1337


parent_folder = "./ml_exercise_therapanacea"

# original image folder and data labels documents
original_folder = os.path.join(parent_folder, "train_img")
labels_file = os.path.join(parent_folder, "label_train.txt")

# Read labels into a list
with open(labels_file, "r") as f:
    labels = [int(line.strip()) for line in f]

# Create a list of corresponding image filenames
image_files = [f"{i+1:06d}.jpg" for i in range(len(labels))]
image_label_pairs = list(zip(image_files, labels))

# shuffle
random.seed(RANDOM_SEED)
random.shuffle(image_label_pairs)

# create folders for training and testing, each folder will contain a couple of sub-folders, one per class
for case in ["train", "test"]:
    folder_name = os.path.join(
        parent_folder, case
    )  # first create a folder for train and test
    os.makedirs(folder_name, exist_ok=True)
    for label in ["0", "1"]:  # then create a sub-folder for the different classes
        os.makedirs(os.path.join(folder_name, label), exist_ok=True)


positif_label_counter = 0
negatif_label_counter = 0

# move the first NUM_IMAGES_PER_CLASS to the test folders then move the rest to the train folders for each class
for image, label in tqdm.tqdm(image_label_pairs):
    if (
        negatif_label_counter >= NUM_IMAGES_PER_CLASS - 1
        and positif_label_counter >= NUM_IMAGES_PER_CLASS - 1
    ):
        img_destination_folder = os.path.join(parent_folder, "train", f"{label}")
    elif negatif_label_counter < NUM_IMAGES_PER_CLASS - 1 and label == 0:
        img_destination_folder = os.path.join(parent_folder, "test", f"{label}")
        negatif_label_counter += 1
    elif positif_label_counter < NUM_IMAGES_PER_CLASS - 1 and label == 1:
        img_destination_folder = os.path.join(parent_folder, "test", f"{label}")
        positif_label_counter += 1
    else:
        img_destination_folder = os.path.join(parent_folder, "train", f"{label}")

    shutil.move(
        os.path.join(original_folder, image),
        os.path.join(img_destination_folder, image),
    )
