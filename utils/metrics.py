"""
Metric computation utilities for Multilingual Review Intelligence.
Computes standard multi-class classification metrics.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

DEFAULT_LABELS = ["negative", "neutral", "positive"]

def compute_classification_metrics(
    y_true: np.ndarray | List[str],
    y_pred: np.ndarray | List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if labels is None:
        labels = DEFAULT_LABELS

    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    total = len(y_t)
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "test_examples": 0,
            "per_class": {l: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for l in labels},
            "confusion_matrix": pd.DataFrame(0, index=labels, columns=labels),
        }

    accuracy = float(np.mean(y_t == y_p))

    precisions = []
    recalls = []
    f1s = []
    supports = []
    per_class = {}

    for l in labels:
        tp = int(np.sum((y_t == l) & (y_p == l)))
        fp = int(np.sum((y_t != l) & (y_p == l)))
        fn = int(np.sum((y_t == l) & (y_p != l)))
        support = int(np.sum(y_t == l))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(support)

        per_class[l] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support": support,
        }

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    total_support = sum(supports)
    if total_support > 0:
        weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / total_support)
    else:
        weighted_f1 = 0.0

    cm_data = np.zeros((len(labels), len(labels)), dtype=int)
    for i, true_l in enumerate(labels):
        for j, pred_l in enumerate(labels):
            cm_data[i, j] = int(np.sum((y_t == true_l) & (y_p == pred_l)))

    cm_df = pd.DataFrame(cm_data, index=labels, columns=labels)
    cm_df.index.name = "True Label"
    cm_df.columns.name = "Predicted Label"

    return {
        "accuracy": accuracy,
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "test_examples": total,
        "per_class": per_class,
        "confusion_matrix": cm_df,
    }

def format_metric(val: float | int | None, as_pct: bool = True, decimals: int = 2) -> str:
    if val is None or pd.isna(val):
        return "N/A"
    if as_pct:
        return f"{val * 100:.{decimals}f}%"
    return f"{val:.{decimals}f}"
