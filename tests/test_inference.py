"""
Unit tests for model inference engine and transparent missing weights handling.
Ensures the system refuses to fabricate predictions or mock scores.
"""
import unittest
from pathlib import Path
import sys

# Ensure root dir in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.inference import ModelInferencer
from config import MODELS_DIR


class TestInference(unittest.TestCase):
    def setUp(self):
        self.inferencer = ModelInferencer(MODELS_DIR)

    def test_missing_weights_detection(self):
        for model in ["mDeBERTa-v3", "XLM-R", "mBERT"]:
            has_weights, path, msg = self.inferencer.check_weights_available(model)
            self.assertFalse(has_weights, f"Weights should be missing by default for {model}.")
            self.assertIn("does not exist", msg.lower())

    def test_get_status_reports_unavailable(self):
        status = self.inferencer.get_status("mDeBERTa-v3")
        self.assertFalse(status["weights_available"])
        self.assertFalse(status["ready_for_inference"])
        self.assertIn("how_to_enable", status)

    def test_predict_refuses_to_fabricate(self):
        """Strict verification: system MUST NOT fabricate predictions or probabilities."""
        result = self.inferencer.predict(
            text="This is an amazing product, I loved it!",
            model_key="mDeBERTa-v3",
        )
        self.assertFalse(result["success"])
        self.assertIsNone(result["prediction"], "Prediction must be None when weights are absent.")
        self.assertIsNone(result["confidence"], "Confidence must be None when weights are absent.")
        self.assertIsNone(result["probabilities"], "Probabilities must be None when weights are absent.")
        self.assertIn("notice", result)
        self.assertIn("never fabricates", result["notice"].lower())


if __name__ == "__main__":
    unittest.main()
