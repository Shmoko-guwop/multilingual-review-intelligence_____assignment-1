"""
Unit tests for metric computations.
Verifies dynamic metric calculations match the ground truth numbers recorded in overall_metrics.csv.
"""
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Ensure root dir in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.metrics import compute_classification_metrics, format_metric
from utils.data_loader import DataLoader
from config import DATA_DIR


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader(DATA_DIR)
        self.df_preds = self.loader.load_csv("predictions")
        self.df_overall = self.loader.load_csv("overall_metrics")

    def test_dynamic_metrics_match_overall_csv(self):
        """Verifies that dynamic metric calculation matches overall_metrics.csv exactly."""
        for _, row in self.df_overall.iterrows():
            exp = row["experiment"]
            model = row["model"]
            expected_acc = row["accuracy"]
            expected_macro_f1 = row["macro_f1"]
            expected_weighted_f1 = row["weighted_f1"]

            subset = self.df_preds[(self.df_preds["experiment"] == exp) & (self.df_preds["model"] == model)]
            metrics = compute_classification_metrics(subset["true_label"].values, subset["predicted_label"].values)

            self.assertAlmostEqual(metrics["accuracy"], expected_acc, places=3,
                                   msg=f"Accuracy mismatch for {exp} - {model}")
            self.assertAlmostEqual(metrics["macro_f1"], expected_macro_f1, places=3,
                                   msg=f"Macro-F1 mismatch for {exp} - {model}")
            self.assertAlmostEqual(metrics["weighted_f1"], expected_weighted_f1, places=3,
                                   msg=f"Weighted-F1 mismatch for {exp} - {model}")

    def test_confusion_matrix_dimensions(self):
        subset = self.df_preds[(self.df_preds["experiment"] == "bilingual") & (self.df_preds["model"] == "mDeBERTa-v3")]
        metrics = compute_classification_metrics(subset["true_label"].values, subset["predicted_label"].values)
        cm = metrics["confusion_matrix"]
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual(cm.values.sum(), 1000)

    def test_empty_predictions(self):
        metrics = compute_classification_metrics([], [])
        self.assertEqual(metrics["test_examples"], 0)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["macro_f1"], 0.0)

    def test_format_metric(self):
        self.assertEqual(format_metric(0.7825), "78.25%")
        self.assertEqual(format_metric(0.7825, as_pct=False, decimals=3), "0.782")
        self.assertEqual(format_metric(None), "N/A")


if __name__ == "__main__":
    unittest.main()
