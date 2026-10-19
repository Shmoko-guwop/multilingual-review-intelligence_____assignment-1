"""
Unit tests for data loader and prediction filtering.
"""
import unittest
from pathlib import Path
import pandas as pd
import sys

# Ensure root dir in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_loader import DataLoader, filter_predictions, resolve_data_path
from config import DATA_DIR


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader(DATA_DIR)

    def test_expected_files_exist(self):
        status = self.loader.get_load_status()
        for key, info in status.items():
            self.assertTrue(info["available"], f"Dataset '{key}' ({info['filename']}) failed to load: {info['error']}")
            self.assertGreater(info["rows"], 0, f"Dataset '{key}' has 0 rows.")

    def test_load_predictions_schema(self):
        df = self.loader.load_csv("predictions")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 4500)
        expected_cols = ["experiment", "model", "language", "text", "true_label", "predicted_label", "confidence", "correct"]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_load_nonexistent_file(self):
        bad_loader = DataLoader(Path("/nonexistent/path"))
        df = bad_loader.load_csv("predictions")
        self.assertIsNone(df)
        status = bad_loader.get_load_status()
        self.assertFalse(status["predictions"]["available"])

    def test_filter_predictions_by_model(self):
        df = self.loader.load_csv("predictions")
        filtered = filter_predictions(df, model="mDeBERTa-v3")
        self.assertEqual(len(filtered), 1500)
        self.assertTrue((filtered["model"] == "mDeBERTa-v3").all())

    def test_filter_predictions_by_language_and_experiment(self):
        df = self.loader.load_csv("predictions")
        filtered = filter_predictions(df, experiment="bilingual", language="es")
        # 3 models * 500 spanish bilingual reviews = 1500
        self.assertEqual(len(filtered), 1500)
        self.assertTrue((filtered["experiment"] == "bilingual").all())
        self.assertTrue((filtered["language"] == "es").all())

    def test_filter_predictions_by_correctness(self):
        df = self.loader.load_csv("predictions")
        correct_only = filter_predictions(df, correctness="Correct Only")
        misclassified = filter_predictions(df, correctness="Misclassified Only")
        self.assertEqual(len(correct_only) + len(misclassified), len(df))
        self.assertTrue(correct_only["correct"].all())
        self.assertFalse(misclassified["correct"].any())

    def test_filter_predictions_by_confidence_and_query(self):
        df = self.loader.load_csv("predictions")
        filtered = filter_predictions(df, min_confidence=0.9, search_query="book")
        self.assertTrue((filtered["confidence"] >= 0.9).all())
        self.assertTrue(filtered["text"].str.lower().str.contains("book").all())


if __name__ == "__main__":
    unittest.main()
