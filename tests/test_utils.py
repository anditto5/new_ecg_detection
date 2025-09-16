# unit tests for utils module
import os
import unittest
import tempfile
import yaml
import matplotlib.pyplot as plt

from src.utils.logger import logging
from src.utils.config import load_config
from src.utils.plot import plot_dummy_curve


class TestLogger(unittest.TestCase):
    def test_logging_output(self):
        """Ensure logger can print messages"""
        logging.info("This is an info log")
        logging.warning("This is a warning log")
        logging.error("This is an error log")
        # If no exceptions, test passes
        self.assertTrue(True)


class TestConfig(unittest.TestCase):
    def setUp(self):
        """Create a temporary yaml config file"""
        self.tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        self.config_data = {"param1": 10, "param2": "test"}
        with open(self.tmp_file.name, "w") as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        if os.path.exists(self.tmp_file.name):
            os.remove(self.tmp_file.name)

    def test_load_config(self):
        """Check if yaml config loads correctly"""
        config = load_config(self.tmp_file.name)
        self.assertEqual(config["param1"], 10)
        self.assertEqual(config["param2"], "test")


class TestPlot(unittest.TestCase):
    def test_plot_curve(self):
        """Ensure plotting utility works without errors"""
        fig = plot_dummy_curve()
        self.assertIsInstance(fig, plt.Figure)


if __name__ == "__main__":
    unittest.main()
