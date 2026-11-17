"""
Evaluation Module for Fake News Detection.

Calculates accuracy, precision, recall, F1-score, confusion matrix,
and generates performance visualizations.
"""

import os
import sys
from pathlib import Path

# Add project root to path for robust imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import logging
from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes standard classification evaluation metrics according to FR-05 & Section 8.5.

    Args:
        y_true: True binary labels (0 = FAKE, 1 = REAL).
        y_pred: Predicted binary labels.
        y_proba: Optional predicted probabilities.

    Returns:
        Dictionary of performance metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    report_dict = classification_report(
        y_true, y_pred, target_names=["FAKE", "REAL"], output_dict=True, zero_division=0
    )

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "labels": ["FAKE", "REAL"],
        },
        "classification_report": report_dict,
        "sample_count": len(y_true),
    }

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = ["FAKE", "REAL"],
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix - Fake News Detection",
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plots a high-quality annotated confusion matrix with counts and percentages.

    Args:
        cm: 2x2 confusion matrix array.
        labels: Class label names.
        output_path: Optional file path to save the generated image.
        title: Plot title.
        cmap: Color map name.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cm_array = np.array(cm)

    # Compute percentage annotations
    cm_sum = np.sum(cm_array)
    annot = np.empty_like(cm_array).astype(str)
    nrows, ncols = cm_array.shape
    for i in range(nrows):
        for j in range(ncols):
            count = cm_array[i, j]
            pct = (count / cm_sum) * 100
            annot[i, j] = f"{count:,}\n({pct:.1f}%)"

    sns.heatmap(
        cm_array,
        annot=annot,
        fmt="",
        cmap=cmap,
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=1.5,
        linecolor="#ffffff",
        annot_kws={"size": 12, "weight": "bold"},
    )

    ax.set_title(title, fontsize=13, pad=15, weight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10, weight="bold")
    ax.set_ylabel("True Label", fontsize=11, labelpad=10, weight="bold")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        logger.info(f"Saved confusion matrix plot to {output_path}")

    return fig


def evaluate_pipeline(
    pipeline: Any,
    X_test: Any,
    y_test: np.ndarray,
    save_dir: str = "models",
) -> Dict[str, Any]:
    """
    Evaluates a trained scikit-learn pipeline on test data and persists metrics and plots.

    Args:
        pipeline: Trained scikit-learn Pipeline.
        X_test: Test features (text strings or series).
        y_test: Test labels.
        save_dir: Directory to save metrics JSON and plots.

    Returns:
        Dictionary of computed metrics.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception as e:
            logger.warning(f"Could not compute probabilities: {e}")

    metrics = compute_classification_metrics(np.array(y_test), np.array(y_pred), y_proba)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        metrics_file = os.path.join(save_dir, "evaluation_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved evaluation metrics to: {metrics_file}")

        plot_file = os.path.join(save_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]["matrix"]),
            labels=metrics["confusion_matrix"]["labels"],
            output_path=plot_file,
        )

    return metrics
