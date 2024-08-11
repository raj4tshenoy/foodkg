import json
import random
from pathlib import Path

def split_dataset(input_file, output_dir, train_ratio=0.8, valid_ratio=0.1):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        input_file (str): Path to the input JSON file.
        output_dir (str): Directory where the split files will be saved.
        train_ratio (float): Proportion of the dataset to use for training.
        valid_ratio (float): Proportion of the dataset to use for validation.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Shuffle the data
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save the splits
    with open(f"{output_dir}/train.jsonl", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(f"{output_dir}/valid.jsonl", 'w') as f:
        for item in valid_data:
            f.write(json.dumps(item) + '\n')

    with open(f"{output_dir}/test.jsonl", 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    split_dataset('Datasets/Recipe1M/recipe1m.json', 'data')