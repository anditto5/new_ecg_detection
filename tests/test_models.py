# unit tests for models module
import os
import cv2
import unittest
import numpy as np

from ultralytics import YOLO
from src.models.yolo_predict import predict_ecg

class TestYOLOModel(unittest.TestCase):

    def setUp(self):
        """Setup a dummy black image for testing inference"""
        self.test_img_path = "test_dummy.jpg"
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.imwrite(self.test_img_path, dummy_img)

        # Use a tiny pretrained model for testing (lightweight)
        self.model_path = "yolo11n.pt"   # nano version of YOLO11
        self.model = YOLO(self.model_path)

    def tearDown(self):
        """Clean up test image"""
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)

    def test_model_loads(self):
        """Ensure YOLO model loads correctly"""
        self.assertIsNotNone(self.model)

    def test_inference_runs(self):
        """Run inference on dummy image"""
        results = self.model(self.test_img_path, conf=0.25)
        self.assertIsNotNone(results)

    def test_predict_ecg_function(self):
        """Test the predict_ecg wrapper function"""
        status, label, conf = predict_ecg(self.model_path, self.test_img_path, conf=0.25)
        self.assertIn(status, ["normal", "Abnormal"])


if __name__ == "__main__":
    unittest.main()
