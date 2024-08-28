import re


# Function to parse the log file and find the best model based on accuracy and F1-score
def find_best_model(file_path):
    # Initialize variables to track the best model across all steps
    best_accuracy = 0  # Best accuracy observed
    best_f1 = 0  # Best F1-score observed
    best_step_acc = None  # Step number corresponding to the best accuracy
    best_step_f1 = None  # Step number corresponding to the best F1-score

    # Initialize variables to track the best model for saved steps (multiples of 25)
    best_saved_accuracy = 0  # Best accuracy observed at saved steps
    best_saved_f1 = 0  # Best F1-score observed at saved steps
    best_saved_step_acc = None  # Step number corresponding to the best saved accuracy
    best_saved_step_f1 = None  # Step number corresponding to the best saved F1-score

    # Regex pattern to match lines with evaluation metrics in the log file
    pattern = r"eval_accuracy: ([\d\.]+), eval_loss: [\d\.]+, eval_precision: [\d\.]+, eval_recall: [\d\.]+, eval_f1: ([\d\.]+), eval_runtime: [\d\.]+, eval_samples_per_second: [\d\.]+, eval_steps_per_second: [\d\.]+, epoch: [\d\.]+, step: (\d+)"

    # Open and read the log file line by line
    with open(file_path, "r") as file:
        for line in file:
            # Use regex to find matches for evaluation metrics
            match = re.search(pattern, line)
            if match:
                eval_accuracy = float(match.group(1))  # Extract accuracy
                eval_f1 = float(match.group(2))  # Extract F1-score
                step = int(match.group(3))  # Extract step number

                # Update the best model for all steps based on accuracy
                if eval_accuracy > best_accuracy:
                    best_accuracy = eval_accuracy
                    best_step_acc = step

                # Update the best model for all steps based on F1-score
                if eval_f1 > best_f1:
                    best_f1 = eval_f1
                    best_step_f1 = step

                # Check if the current step is a saved step (multiple of 25)
                if step % 25 == 0:
                    # Update the best saved model based on accuracy
                    if eval_accuracy > best_saved_accuracy:
                        best_saved_accuracy = eval_accuracy
                        best_saved_step_acc = step

                    # Update the best saved model based on F1-score
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

# Call the function with the file path to find and print the best models
find_best_model(log_file_path)
