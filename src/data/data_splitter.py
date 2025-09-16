# functions to split ECG data into train/val/test
import os
import glob
import random
import shutil

def split_dataset(image_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split ECG dataset into train/val/test with YOLO structure.

    Args:
        image_dir (str): Path to folder containing images.
        output_dir (str): Path where train/val/test folders will be created.
        train_ratio (float): Proportion of training data.
        val_ratio (float): Proportion of validation data.
        test_ratio (float): Proportion of test data.
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)

    # Get all images
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    random.shuffle(image_paths)

    total = len(image_paths)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_files = image_paths[:n_train]
    val_files = image_paths[n_train:n_train+n_val]
    test_files = image_paths[n_train+n_val:]

    print(f"Total: {total}, Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # Create YOLO folders
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    # Helper to copy image + corresponding label
    def copy_files(file_list, split):
        for img_path in file_list:
            base = os.path.basename(img_path)
            label_path = os.path.splitext(img_path)[0] + ".txt"

            # Copy image
            shutil.copy(img_path, os.path.join(output_dir, split, "images", base))

            # Copy label if exists
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(output_dir, split, "labels", os.path.basename(label_path)))

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print(f"✅ Dataset split complete. Saved at: {output_dir}")


if __name__ == "__main__":
    # Example usage
    image_dir = "../../raw_ecg/images"       # where all ECG images are stored
    output_dir = "../../custom_data/data_ecg"
    split_dataset(image_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
