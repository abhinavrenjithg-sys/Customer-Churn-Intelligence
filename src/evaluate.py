"""Model evaluation, metrics computation, threshold analysis, and diagnostic plotting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COLOR_CHURN = "#E11D48"     # Rose / Vibrant Crimson
COLOR_RETAINED = "#2563EB"  # Deep Royal Blue
COLOR_ACCENT = "#8B5CF6"    # Purple accent
FIG_DPI = 300


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes comprehensive binary classification metrics for the positive (churn=1) class.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground-truth binary labels (0 or 1).
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class (1).
    threshold : float
        Decision boundary threshold. Default is 0.5.

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed performance metrics.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_bin = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics
    roc_auc = float(roc_auc_score(y_true_arr, y_pred_proba)) if len(np.unique(y_true_arr)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true_arr, y_pred_proba)) if len(np.unique(y_true_arr)) > 1 else 0.0
    rec = float(recall_score(y_true_arr, y_pred_bin, pos_label=1, zero_division=0))
    prec = float(precision_score(y_true_arr, y_pred_bin, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_bin, pos_label=1, zero_division=0))
    brier = float(brier_score_loss(y_true_arr, y_pred_proba))

    cm = confusion_matrix(y_true_arr, y_pred_bin)
    tn, fp, fn, tp = cm.ravel()

    report_dict = classification_report(
        y_true_arr,
        y_pred_bin,
        target_names=["Retained (0)", "Churn (1)"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "threshold": threshold,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "classification_report": report_dict,
    }


def analyze_thresholds(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Evaluates Precision, Recall, and F1 across an array of decision thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 33)

    records = []
    y_true_arr = np.asarray(y_true).astype(int)

    for th in thresholds:
        y_pred_bin = (y_pred_proba >= th).astype(int)
        rec = recall_score(y_true_arr, y_pred_bin, pos_label=1, zero_division=0)
        prec = precision_score(y_true_arr, y_pred_bin, pos_label=1, zero_division=0)
        f1 = f1_score(y_true_arr, y_pred_bin, pos_label=1, zero_division=0)
        records.append({
            "threshold": round(float(th), 3),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
        })

    return pd.DataFrame(records)


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    output_path: Path | str,
    title: str = "Test Set Confusion Matrix",
) -> Path:
    """Plots and saves an annotated confusion matrix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=FIG_DPI)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"},
        xticklabels=["Retained (No)", "Churn (Yes)"],
        yticklabels=["Retained (No)", "Churn (Yes)"],
        ax=ax,
    )
    ax.set_title(title, pad=15, fontweight="bold", fontsize=12)
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Ground-Truth", fontweight="bold", labelpad=10)

    # Annotate business quadrants
    labels = [["True Negatives", "False Positives\n(Unnecessary Retention Outreach)"],
              ["False Negatives\n(Missed Churners!)", "True Positives\n(Successfully Flagged Churn)"]]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.75, labels[i][j], ha="center", va="center", color="#475569", fontsize=7.5)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roc_curve(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray,
    output_path: Path | str,
    model_name: str = "Primary Model",
) -> Path:
    """Plots the ROC curve against the random guessing baseline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=FIG_DPI)
    ax.plot(fpr, tpr, color=COLOR_RETAINED, lw=2.5, label=f"{model_name} (ROC-AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", pad=15, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_precision_recall_curve(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray,
    output_path: Path | str,
    model_name: str = "Primary Model",
) -> Path:
    """Plots the Precision-Recall curve against the no-skill baseline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prec, rec, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    baseline = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=FIG_DPI)
    ax.plot(rec, prec, color=COLOR_CHURN, lw=2.5, label=f"{model_name} (PR-AUC / AP = {pr_auc:.3f})")
    ax.axhline(y=baseline, color="#94A3B8", lw=1.5, linestyle="--", label=f"No-Skill Baseline ({baseline:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Coverage of Churners)", fontweight="bold")
    ax.set_ylabel("Precision (Accuracy of Churn Alerts)", fontweight="bold")
    ax.set_title("Precision-Recall (PR) Curve", pad=15, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_threshold_analysis(
    threshold_df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Plots Precision, Recall, and F1 across classification thresholds."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=FIG_DPI)
    ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", color="#059669", lw=2.2)
    ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", color=COLOR_CHURN, lw=2.2)
    ax.plot(threshold_df["threshold"], threshold_df["f1_score"], label="F1-Score", color=COLOR_RETAINED, lw=2.2, linestyle="--")

    # Find optimal F1 threshold
    best_f1_row = threshold_df.loc[threshold_df["f1_score"].idxmax()]
    ax.axvline(x=best_f1_row["threshold"], color="#D97706", linestyle=":", lw=1.8,
               label=f"Max F1 ({best_f1_row['f1_score']:.3f} @ th={best_f1_row['threshold']})")

    ax.set_title("Trade-off Analysis: Threshold vs. Precision, Recall & F1", pad=15, fontweight="bold")
    ax.set_xlabel("Classification Decision Threshold", fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0)", fontweight="bold")
    ax.set_xlim(threshold_df["threshold"].min(), threshold_df["threshold"].max())
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="center right", frameon=True, facecolor="white")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_model_comparison(
    cv_results_df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Generates comparison bar chart across models evaluated in Cross-Validation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ["test_roc_auc", "test_pr_auc", "test_f1", "test_recall"]
    metric_labels = ["ROC-AUC", "PR-AUC", "F1-Score", "Recall"]

    # Filter to available metrics
    available_metrics = [m for m in metrics_to_plot if f"{m}_mean" in cv_results_df.columns]
    labels = [metric_labels[i] for i, m in enumerate(metrics_to_plot) if f"{m}_mean" in cv_results_df.columns]

    models = cv_results_df["model"].tolist()
    n_models = len(models)
    n_metrics = len(available_metrics)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=FIG_DPI)
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    palette = [COLOR_RETAINED, COLOR_CHURN, "#10B981", "#8B5CF6"]

    for i, metric in enumerate(available_metrics):
        means = cv_results_df[f"{metric}_mean"]
        stds = cv_results_df[f"{metric}_std"]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=labels[i], color=palette[i % len(palette)], alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylabel("Cross-Validation Score (5-Fold Mean ± Std)")
    ax.set_title("5-Fold Stratified Cross-Validation Benchmark Comparison", pad=15, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", frameon=True, facecolor="white")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
