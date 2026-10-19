"""
Unit tests for CSV upload validator and schema checking.
"""
import unittest
from pathlib import Path
import pandas as pd
import sys

# Ensure root dir in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_loader import validate_uploaded_csv


class TestUploadValidator(unittest.TestCase):
    def test_valid_prediction_csv(self):
        df = pd.DataFrame({
            "text": ["Great product", "Awful quality", "Just okay"],
            "true_label": ["positive", "negative", "neutral"],
            "predicted_label": ["positive", "negative", "positive"],
            "confidence": [0.95, 0.88, 0.62],
        })
        is_valid, mode, issues, clean_df = validate_uploaded_csv(df)
        self.assertTrue(is_valid)
        self.assertEqual(mode, "predictions")
        self.assertEqual(len(issues), 0)
        self.assertIn("correct", clean_df.columns)
        self.assertEqual(list(clean_df["correct"]), [True, True, False])

    def test_case_insensitive_and_whitespace_columns(self):
        df = pd.DataFrame({
            "  Text  ": ["Buen producto", "Malo"],
            "Target ": ["positive", "negative"],
            "Prediction": ["positive", "neutral"],
        })
        is_valid, mode, issues, clean_df = validate_uploaded_csv(df)
        self.assertTrue(is_valid)
        self.assertEqual(mode, "predictions")
        self.assertIn("text", clean_df.columns)
        self.assertIn("true_label", clean_df.columns)
        self.assertIn("predicted_label", clean_df.columns)

    def test_raw_review_csv(self):
        df = pd.DataFrame({
            "review_text": ["This is a review without labels", "Another review"],
            "language": ["en", "en"],
        })
        is_valid, mode, issues, clean_df = validate_uploaded_csv(df)
        self.assertTrue(is_valid)
        self.assertEqual(mode, "raw_reviews")
        self.assertIn("text", clean_df.columns)

    def test_missing_text_column(self):
        df = pd.DataFrame({
            "id": [1, 2],
            "score": [5, 1],
        })
        is_valid, mode, issues, clean_df = validate_uploaded_csv(df)
        self.assertFalse(is_valid)
        self.assertEqual(mode, "invalid")
        self.assertTrue(any("text" in i.lower() for i in issues))

    def test_invalid_sentiment_labels(self):
        df = pd.DataFrame({
            "text": ["Review 1", "Review 2"],
            "true_label": ["happy", "sad"],  # Invalid
            "predicted_label": ["positive", "negative"],
        })
        is_valid, mode, issues, clean_df = validate_uploaded_csv(df)
        self.assertFalse(is_valid)
        self.assertTrue(any("unrecognized" in i.lower() for i in issues))

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        is_valid, mode, issues, clean_df = validate_uploaded_csv(df)
        self.assertFalse(is_valid)
        self.assertEqual(mode, "invalid")


if __name__ == "__main__":
    unittest.main()
