# YOLO evaluation metrics (mAP, confusion matrix)
import os
import cv2
import unittest
import numpy as np

from ultralytics import YOLO
from src.models.yolo_predict import predict_ecg

def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    Each box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


class TestYOLOModel(unittest.TestCase):

    def setUp(self):
        """Setup dummy image for testing"""
        self.test_img_path = "test_dummy.jpg"
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.imwrite(self.test_img_path, dummy_img)

        # Load small YOLO model for quick tests
        self.model_path = "yolo11n.pt"
        self.model = YOLO(self.model_path)

    def tearDown(self):
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)

    def test_model_loads(self):
        """Ensure YOLO model loads"""
        self.assertIsNotNone(self.model)

    def test_inference_runs(self):
        """Run inference on dummy image"""
        results = self.model(self.test_img_path, conf=0.25)
        self.assertIsNotNone(results)

    def test_predict_ecg_function(self):
        """Check predict_ecg wrapper"""
        status, label, conf = predict_ecg(self.model_path, self.test_img_path, conf=0.25)
        self.assertIn(status, ["Healthy", "Abnormal"])

    def test_iou_function(self):
        """Test IoU calculation"""
        box_a = [0, 0, 100, 100]
        box_b = [50, 50, 150, 150]
        iou = compute_iou(box_a, box_b)
        self.assertTrue(0 <= iou <= 1)

    def test_model_validation_map(self):
        """
        Run YOLO validation on a dummy dataset.
        Requires that you have at least a tiny dataset configured in data.yaml.
        """
        # If a dataset config exists, run validation
        if os.path.exists("config/data.yaml"):
            metrics = self.model.val(data="config/data.yaml", imgsz=640)
            # mAP@0.5 should be reported
            self.assertIn("metrics/mAP50(B)", metrics.keys())


if __name__ == "__main__":
    unittest.main()
