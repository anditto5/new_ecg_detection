# unit tests for data module
import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from src.data.data_loader import load_images_from_folder, load_yolo_labels, load_image_and_labels
from src.data.data_splitter import split_dataset

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        """Create a temporary dataset for testing"""
        self.test_dir = tempfile.mkdtemp()

        # Create dummy image
        img_path = os.path.join(self.test_dir, "test1.jpg")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)

        # Create corresponding YOLO label
        with open(os.path.splitext(img_path)[0] + ".txt", "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

    def tearDown(self):
        """Clean up temporary dataset"""
        shutil.rmtree(self.test_dir)

    def test_load_images_from_folder(self):
        images = load_images_from_folder(self.test_dir)
        self.assertEqual(len(images), 1)

    def test_load_yolo_labels(self):
        label_path = os.path.join(self.test_dir, "test1.txt")
        labels = load_yolo_labels(label_path)
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0][0], 0)  # class_id = 0

    def test_load_image_and_labels(self):
        img_path = os.path.join(self.test_dir, "test1.jpg")
        img, labels = load_image_and_labels(img_path)
        self.assertIsNotNone(img)
        self.assertEqual(len(labels), 1)


class TestDataSplitter(unittest.TestCase):

    def setUp(self):
        """Create a temporary dataset with multiple images"""
        self.image_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()

        # Create 5 dummy images with labels
        for i in range(5):
            img_path = os.path.join(self.image_dir, f"test{i}.jpg")
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2.imwrite(img_path, img)
            with open(os.path.splitext(img_path)[0] + ".txt", "w") as f:
                f.write(f"0 0.5 0.5 0.3 0.3\n")

    def tearDown(self):
        shutil.rmtree(self.image_dir)
        shutil.rmtree(self.output_dir)

    def test_split_dataset(self):
        split_dataset(self.image_dir, self.output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        # Check directories created
        for split in ["train", "val", "test"]:
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, split, "images")))
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, split, "labels")))


if __name__ == "__main__":
    unittest.main()
