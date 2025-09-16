# functions to load ECG images
import os
import glob
import cv2

def load_images_from_folder(folder, extensions=("*.jpg", "*.png")):
    """
    Load image file paths from a folder.
    
    Args:
        folder (str): Path to the folder containing images.
        extensions (tuple): File extensions to search for.
    
    Returns:
        list: List of image file paths.
    """
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))
    return image_paths


def load_yolo_labels(label_path):
    """
    Load YOLO format labels from a .txt file.
    
    Args:
        label_path (str): Path to the YOLO label file (.txt).
    
    Returns:
        list: List of annotations [class_id, x_center, y_center, width, height].
    """
    if not os.path.exists(label_path):
        return []
    
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:  # YOLO format
            class_id, x, y, w, h = map(float, parts)
            labels.append([int(class_id), x, y, w, h])
    return labels


def load_image_and_labels(image_path):
    """
    Load an image and its corresponding YOLO labels.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        image (numpy.ndarray): Loaded image (BGR).
        labels (list): YOLO labels for the image.
    """
    # Load image
    image = cv2.imread(image_path)
    
    # Find label file (same name but .txt)
    label_path = os.path.splitext(image_path)[0] + ".txt"
    labels = load_yolo_labels(label_path)
    
    return image, labels


if __name__ == "__main__":
    # Example usage
    train_folder = "../../custom_data/data_ecg/train/images"
    images = load_images_from_folder(train_folder)
    print(f"Found {len(images)} training images")

    if images:
        img, lbls = load_image_and_labels(images[0])
        print(f"Image shape: {img.shape}, Labels: {lbls}")
