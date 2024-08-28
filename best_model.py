import re


# Function to parse the log file and find the best model based on accuracy and F1-score
def find_best_model(file_path):
    # Initialize variables to track the best model for all steps
    best_accuracy = 0
    best_f1 = 0
    best_step_acc = None
    best_step_f1 = None

    # Initialize variables to track the best model for saved steps (every 25 steps)
    best_saved_accuracy = 0
    best_saved_f1 = 0
    best_saved_step_acc = None
    best_saved_step_f1 = None

    # Regex pattern to match evaluation metrics
    pattern = r"eval_accuracy: ([\d\.]+), eval_loss: [\d\.]+, eval_precision: [\d\.]+, eval_recall: [\d\.]+, eval_f1: ([\d\.]+), eval_runtime: [\d\.]+, eval_samples_per_second: [\d\.]+, eval_steps_per_second: [\d\.]+, epoch: [\d\.]+, step: (\d+)"

    # Open and read the log file line by line
    with open(file_path, "r") as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                eval_accuracy = float(match.group(1))
                eval_f1 = float(match.group(2))
                step = int(match.group(3))

                # Track the best model for all steps
                if eval_accuracy > best_accuracy:
                    best_accuracy = eval_accuracy
                    best_step_acc = step

                if eval_f1 > best_f1:
                    best_f1 = eval_f1
                    best_step_f1 = step

                # Track the best model for saved steps (multiples of 25)
                if step % 25 == 0:
                    if eval_accuracy > best_saved_accuracy:
                        best_saved_accuracy = eval_accuracy
                        best_saved_step_acc = step

                    if eval_f1 > best_saved_f1:
                        best_saved_f1 = eval_f1
                        best_saved_step_f1 = step

    # Output the best models based on accuracy and F1-score
    print(
        f"Best model overall by accuracy: Step {best_step_acc} with accuracy {best_accuracy}"
    )
    print(
        f"Best model overall by F1-score: Step {best_step_f1} with F1-score {best_f1}"
    )
    print(
        f"Best saved model by accuracy (every 25 steps): Step {best_saved_step_acc} with accuracy {best_saved_accuracy}"
    )
    print(
        f"Best saved model by F1-score (every 25 steps): Step {best_saved_step_f1} with F1-score {best_saved_f1}"
    )


# File path to the log file
log_file_path = "./output_logs.txt"

# Call the function with the file path
find_best_model(log_file_path)
