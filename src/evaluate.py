"""
evaluate.py
===========
Standard metric computation utilities shared across both models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score,
                              confusion_matrix, r2_score,
                              classification_report)


def compute_classification_metrics(y_true, y_pred, model_name: str = "") -> dict:
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, zero_division=0)

    metrics = {
        "model":     model_name,
        "accuracy":  round(acc, 4),
        "macro_f1":  round(f1, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "confusion_matrix": cm,
        "report":    report,
    }
    return metrics


def exact_match(y_true: list, y_pred: list) -> float:
    """Strict character-level exact match."""
    correct = sum(1 for t, p in zip(y_true, y_pred)
                  if str(t).strip() == str(p).strip())
    return correct / (len(y_true) + 1e-9)


def compute_regression_metrics(y_true, y_scores) -> dict:
    r2 = r2_score(y_true, y_scores)
    mse = np.mean((np.array(y_true) - np.array(y_scores)) ** 2)
    mae = np.mean(np.abs(np.array(y_true) - np.array(y_scores)))
    rmse = mse ** 0.5
    return {
        "r2":   round(r2, 4),
        "mse":  round(mse, 4),
        "mae":  round(mae, 4),
        "rmse": round(rmse, 4),
    }


def distractor_diversity(distractors: list) -> float:
    """
    Average pairwise cosine distance between distractor word-sets.
    Higher = more diverse.
    """
    if len(distractors) < 2:
        return 0.0
    from itertools import combinations
    scores = []
    for a, b in combinations(distractors, 2):
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        union = len(set_a | set_b)
        inter = len(set_a & set_b)
        scores.append(1.0 - (inter / (union + 1e-9)))
    return float(np.mean(scores))
