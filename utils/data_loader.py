"""
Data loading, validation, caching, and filtering utilities for Multilingual Review Intelligence.
Loads experimental results from data/ directory using relative paths with graceful fallbacks.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd

from config import DATA_DIR, SENTIMENT_CLASSES


def resolve_data_path(filename: str, preferred_dir: Optional[Path] = None) -> Path:
    """Resolves data file path looking in preferred_dir, config DATA_DIR, ./data, and current directory."""
    candidates = []
    if preferred_dir:
        candidates.append(Path(preferred_dir) / filename)

    candidates.extend([
        DATA_DIR / filename,
        Path("data") / filename,
        Path(filename),
        Path(__file__).resolve().parent.parent / "data" / filename if "__file__" in globals() else Path("data") / filename,
    ])

    for p in candidates:
        if p.exists():
            return p
    # If preferred_dir was given, return that path for error reporting
    return Path(preferred_dir) / filename if preferred_dir else DATA_DIR / filename


class DataLoader:
    """Robust data loader for all project CSVs."""

    EXPECTED_FILES = {
        "overall_metrics": "overall_metrics.csv",
        "bilingual_comparison": "bilingual_comparison.csv",
        "cross_lingual_comparison": "cross_lingual_comparison.csv",
        "language_wise_metrics": "language_wise_metrics.csv",
        "model_characteristics": "model_characteristics.csv",
        "error_summary": "error_summary.csv",
        "predictions": "predictions.csv",
        "error_analysis": "error_analysis.csv",
    }

    REQUIRED_COLUMNS = {
        "overall_metrics": ["experiment", "model", "accuracy", "macro_f1", "weighted_f1"],
        "bilingual_comparison": ["experiment", "model", "accuracy", "macro_f1", "weighted_f1"],
        "cross_lingual_comparison": ["experiment", "model", "accuracy", "macro_f1", "weighted_f1"],
        "language_wise_metrics": ["experiment", "model", "language", "accuracy", "macro_f1"],
        "model_characteristics": ["model", "architecture", "pretraining_data"],
        "error_summary": ["experiment", "model", "language", "accuracy", "error_rate"],
        "predictions": ["experiment", "model", "language", "text", "true_label", "predicted_label", "confidence", "correct"],
        "error_analysis": ["experiment", "model", "language", "text", "true_label", "predicted_label", "confidence", "correct"],
    }

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._cache: Dict[str, pd.DataFrame] = {}
        self._load_errors: Dict[str, str] = {}

    def load_csv(self, key: str) -> Optional[pd.DataFrame]:
        """Loads a specific CSV by key with caching and column verification."""
        if key in self._cache:
            return self._cache[key]

        filename = self.EXPECTED_FILES.get(key)
        if not filename:
            self._load_errors[key] = f"Unknown dataset key: {key}"
            return None

        # If an explicit data_dir is passed, check it strictly
        file_path = self.data_dir / filename
        if not file_path.exists():
            # Try general fallback resolver only if data_dir is default DATA_DIR
            if self.data_dir == DATA_DIR:
                file_path = resolve_data_path(filename, self.data_dir)
            if not file_path.exists():
                self._load_errors[key] = f"File not found: {file_path}"
                return None

        try:
            df = pd.read_csv(file_path)
            # Verify required columns
            expected_cols = self.REQUIRED_COLUMNS.get(key, [])
            missing_cols = [c for c in expected_cols if c not in df.columns]
            if missing_cols:
                self._load_errors[key] = f"File {filename} is missing required columns: {missing_cols}"
                return None

            # Normalization / cleanup
            if "correct" in df.columns:
                df["correct"] = df["correct"].astype(bool)
            if "confidence" in df.columns:
                df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

            self._cache[key] = df
            return df
        except Exception as e:
            self._load_errors[key] = f"Error reading {file_path}: {str(e)}"
            return None

    def get_load_status(self) -> Dict[str, Any]:
        """Returns inventory of loaded files and any missing or invalid files."""
        status = {}
        for key, fname in self.EXPECTED_FILES.items():
            df = self.load_csv(key)
            if df is not None:
                status[key] = {
                    "available": True,
                    "filename": fname,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "error": None,
                }
            else:
                status[key] = {
                    "available": False,
                    "filename": fname,
                    "rows": 0,
                    "columns": [],
                    "error": self._load_errors.get(key, "Unknown error"),
                }
        return status

    def load_all(self) -> Dict[str, Optional[pd.DataFrame]]:
        """Loads all expected CSV files into a dictionary."""
        return {key: self.load_csv(key) for key in self.EXPECTED_FILES}


def filter_predictions(
    df: pd.DataFrame,
    experiment: Optional[str] = None,
    model: Optional[str] = None,
    language: Optional[str] = None,
    true_label: Optional[str] = None,
    predicted_label: Optional[str] = None,
    correctness: Optional[str] = None,  # "All", "Correct Only", "Misclassified Only"
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
    search_query: Optional[str] = None,
) -> pd.DataFrame:
    """Applies multi-criteria filtering to the predictions dataframe."""
    filtered = df.copy()

    if experiment and experiment != "All":
        filtered = filtered[filtered["experiment"] == experiment]

    if model and model != "All":
        filtered = filtered[filtered["model"] == model]

    if language and language != "All":
        filtered = filtered[filtered["language"] == language]

    if true_label and true_label != "All":
        filtered = filtered[filtered["true_label"] == true_label]

    if predicted_label and predicted_label != "All":
        filtered = filtered[filtered["predicted_label"] == predicted_label]

    if correctness == "Correct Only":
        filtered = filtered[filtered["correct"] == True]
    elif correctness == "Misclassified Only":
        filtered = filtered[filtered["correct"] == False]

    if "confidence" in filtered.columns:
        filtered = filtered[
            (filtered["confidence"] >= min_confidence) & (filtered["confidence"] <= max_confidence)
        ]

    if search_query and search_query.strip():
        q = search_query.strip().lower()
        filtered = filtered[filtered["text"].str.lower().str.contains(q, na=False)]

    return filtered


def validate_uploaded_csv(df: pd.DataFrame) -> Tuple[bool, str, List[str], pd.DataFrame]:
    """
    Validates user-uploaded CSV.
    Determines whether it is a Prediction CSV, a Raw Review CSV, or Invalid.
    
    Returns:
        (is_valid, mode, issues, clean_df)
        mode is one of: "predictions", "raw_reviews", "invalid"
    """
    issues = []
    clean_df = df.copy()

    if clean_df.empty:
        return False, "invalid", ["The uploaded file is empty."], clean_df

    # Normalize column names (strip whitespace and lower case for matching)
    col_map = {c: c.strip().lower() for c in clean_df.columns}
    inv_map = {v: k for k, v in col_map.items()}

    # Check for text column
    text_candidates = ["text", "review", "review_text", "content", "body"]
    found_text_col = None
    for cand in text_candidates:
        if cand in inv_map:
            found_text_col = inv_map[cand]
            break

    if not found_text_col:
        issues.append(f"Missing required text column. Expected one of: {text_candidates}")
        return False, "invalid", issues, clean_df

    # Rename text column to canonical 'text'
    if found_text_col != "text":
        clean_df = clean_df.rename(columns={found_text_col: "text"})

    # Check for prediction evaluation columns
    true_candidates = ["true_label", "ground_truth", "label", "actual", "target", "true_sentiment"]
    pred_candidates = ["predicted_label", "prediction", "pred", "pred_label", "predicted_sentiment"]

    found_true_col = next((inv_map[c] for c in true_candidates if c in inv_map), None)
    found_pred_col = next((inv_map[c] for c in pred_candidates if c in inv_map), None)

    if found_true_col and found_pred_col:
        # Prediction mode
        if found_true_col != "true_label":
            clean_df = clean_df.rename(columns={found_true_col: "true_label"})
        if found_pred_col != "predicted_label":
            clean_df = clean_df.rename(columns={found_pred_col: "predicted_label"})

        # Clean strings
        clean_df["true_label"] = clean_df["true_label"].astype(str).str.strip().str.lower()
        clean_df["predicted_label"] = clean_df["predicted_label"].astype(str).str.strip().str.lower()

        # Check label domain
        invalid_true = set(clean_df["true_label"].unique()) - set(SENTIMENT_CLASSES)
        invalid_pred = set(clean_df["predicted_label"].unique()) - set(SENTIMENT_CLASSES)

        if invalid_true:
            issues.append(f"Unrecognized true labels found: {invalid_true}. Expected {SENTIMENT_CLASSES}")
        if invalid_pred:
            issues.append(f"Unrecognized predicted labels found: {invalid_pred}. Expected {SENTIMENT_CLASSES}")

        # Compute 'correct' column if missing
        if "correct" not in clean_df.columns:
            clean_df["correct"] = clean_df["true_label"] == clean_df["predicted_label"]
        else:
            clean_df["correct"] = clean_df["correct"].astype(bool)

        # Parse confidence if present
        conf_candidates = ["confidence", "prob", "probability", "score"]
        found_conf = next((inv_map[c] for c in conf_candidates if c in inv_map), None)
        if found_conf and found_conf != "confidence":
            clean_df = clean_df.rename(columns={found_conf: "confidence"})

        if "confidence" in clean_df.columns:
            clean_df["confidence"] = pd.to_numeric(clean_df["confidence"], errors="coerce").fillna(1.0)
        else:
            clean_df["confidence"] = 1.0

        is_valid = len(issues) == 0
        return is_valid, "predictions", issues, clean_df

    else:
        # Raw reviews mode (unlabeled or text only)
        mode = "raw_reviews"
        # Language detection / column check
        lang_candidates = ["language", "lang", "locale"]
        found_lang = next((inv_map[c] for c in lang_candidates if c in inv_map), None)
        if found_lang and found_lang != "language":
            clean_df = clean_df.rename(columns={found_lang: "language"})

        return True, mode, issues, clean_df
