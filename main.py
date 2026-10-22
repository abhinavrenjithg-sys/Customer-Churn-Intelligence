"""Customer Churn Intelligence - End-to-End Execution Pipeline.

Usage:
    python main.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from src.data_loader import (
    DEFAULT_DATA_FILE,
    load_raw_data,
    save_data_quality_report,
)
from src.eda import generate_all_eda_figures
from src.train import train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def generate_markdown_report(summary: dict, output_path: Path) -> None:
    """Generates an executive markdown evaluation report."""
    test_metrics = summary["test_metrics"]
    cm = test_metrics["confusion_matrix"]
    cv_list = summary["cross_validation_results"]

    cv_rows = ""
    for row in cv_list:
        cv_rows += (
            f"| **{row['model']}** "
            f"| {row.get('test_roc_auc_mean', 0.0):.4f} ± {row.get('test_roc_auc_std', 0.0):.4f} "
            f"| {row.get('test_pr_auc_mean', 0.0):.4f} ± {row.get('test_pr_auc_std', 0.0):.4f} "
            f"| {row.get('test_recall_mean', 0.0):.4f} ± {row.get('test_recall_std', 0.0):.4f} "
            f"| {row.get('test_f1_mean', 0.0):.4f} ± {row.get('test_f1_std', 0.0):.4f} |\n"
        )

    md_content = f"""# Customer Churn Intelligence — Model Evaluation Report

**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Primary Selected Architecture:** `{summary['selected_model']}`  
**Hyperparameter Tuning:** `{summary['tuning_info'].get('best_params', {})}`  

---

## 1. Executive Summary & Business Rationale

In subscription businesses, identifying customers at risk of churn before they cancel is critical for revenue retention. 
Because the cost of proactive customer retention outreach (a discount, customer success call, or perk) is drastically smaller than the customer lifetime value (LTV) lost from customer departure, **minimizing False Negatives (missed churners) while maintaining strong ROC-AUC is our primary technical objective**.

The primary model was chosen based on 5-fold Stratified Cross-Validation on the training partition and subsequently evaluated once on the untouched held-out test set (20% sample, {summary['test_samples']} customers).

---

## 2. 5-Fold Stratified Cross-Validation Benchmark

| Model | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) | Recall (Mean ± Std) | F1-Score (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- |
{cv_rows}

---

## 3. Final Evaluation on Untouched Test Set

- **Held-Out Test Samples:** {summary['test_samples']}
- **True Churn Rate in Test Set:** {summary['test_churn_rate'] * 100:.2f}%
- **Decision Threshold Applied:** {test_metrics['threshold']}

### Performance Metrics Table

| Metric | Measured Value | Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **{test_metrics['roc_auc']:.4f}** | Overall ranking capability across all thresholds |
| **PR-AUC (Avg Precision)** | **{test_metrics['pr_auc']:.4f}** | Performance specifically on the minority churn class |
| **Recall (Sensitivity)** | **{test_metrics['recall']:.4f}** | Percentage of actual churners caught |
| **Precision** | **{test_metrics['precision']:.4f}** | Percentage of flagged accounts who actually churn |
| **F1-Score** | **{test_metrics['f1']:.4f}** | Harmonic mean of precision and recall |
| **Brier Score** | **{test_metrics['brier_score']:.4f}** | Probability calibration accuracy (lower is better) |

### Test Confusion Matrix

| | Predicted Retained (0) | Predicted Churn (1) |
| :--- | :--- | :--- |
| **Actual Retained (0)** | **{cm['true_negatives']}** (True Negatives) | **{cm['false_positives']}** (False Positives) |
| **Actual Churn (1)** | **{cm['false_negatives']}** (False Negatives) | **{cm['true_positives']}** (True Positives) |

---

## 4. Key Predictive Associations (Global Explainability)

Based on model analysis, the top factors statistically associated with increased churn risk include:
1. **Month-to-month contracts:** Customers without long-term commitments churn at significantly higher rates.
2. **Fiber optic internet service:** Associated with elevated churn, frequently correlating with higher monthly bills ($70-$100+).
3. **Electronic check payment method:** Correlated with higher attrition compared to automated bank or credit card transfers.
4. **Lack of tech support / online security add-ons:** Subscribers without core protective services exhibit greater turnover.

Conversely, **longer tenure (months with provider)** and **multi-year contracts** are the strongest protective factors against churn.

> **Causality Disclaimer:** These drivers represent statistical patterns identified across historical data. They do not prove that altering an isolated factor (e.g. forcing a customer into auto-pay) will guarantee retention without addressing underlying customer sentiment or price sensitivity.

---

## 5. Artifacts and Generated Visualizations

- ROC Curve: `reports/figures/roc_curve.png`
- Precision-Recall Curve: `reports/figures/precision_recall_curve.png`
- Confusion Matrix: `reports/figures/confusion_matrix.png`
- Model Comparison: `reports/figures/model_comparison.png`
- Threshold Trade-off Curve: `reports/figures/threshold_analysis.png`
- Feature Importance: `reports/figures/feature_importance.png`
- Trained Pipeline Artifact: `artifacts/best_model_pipeline.joblib`
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved markdown evaluation report to %s", output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Customer Churn Intelligence Training Pipeline")
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH), help="Path to raw CSV dataset")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA figure generation")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("STARTING CUSTOMER CHURN INTELLIGENCE PIPELINE")
    logger.info("=" * 70)

    # 1. Load and Validate Raw Data
    logger.info("Step 1/5: Loading and validating dataset ...")
    csv_file = Path(args.data_path)
    df, quality_report = load_raw_data(csv_file)
    save_data_quality_report(quality_report, REPORTS_DIR / "data_quality_report.json")
    logger.info("Dataset validated: %d rows, %d columns. Blank TotalCharges handled: %d.",
                quality_report["rows"], quality_report["columns"], quality_report["blank_total_charges_count"])

    # 2. Generate EDA Visualizations
    if not args.skip_eda:
        logger.info("Step 2/5: Generating high-resolution Exploratory Data Analysis figures ...")
        eda_figs = generate_all_eda_figures(df, FIGURES_DIR)
        logger.info("Saved %d EDA figures to %s", len(eda_figs), FIGURES_DIR)
    else:
        logger.info("Step 2/5: Skipping EDA figure generation (--skip-eda set)")

    # 3. Model Training, Cross-Validation, Tuning & Final Evaluation
    logger.info("Step 3/5: Running Stratified CV Benchmark, Tuning & Evaluation ...")
    summary = train_and_evaluate(
        df=df,
        test_size=0.20,
        random_state=42,
        artifacts_dir=ARTIFACTS_DIR,
        figures_dir=FIGURES_DIR,
    )

    # 4. Generate Comprehensive Evaluation Report
    logger.info("Step 4/5: Generating Markdown evaluation report ...")
    report_file = REPORTS_DIR / "model_evaluation_report.md"
    generate_markdown_report(summary, report_file)

    # 5. Pipeline Summary
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("CUSTOMER CHURN PIPELINE COMPLETED IN %.2f SECONDS", elapsed)
    logger.info("Selected Primary Model: %s", summary["selected_model"])
    logger.info("Test ROC-AUC: %.4f | PR-AUC: %.4f | Recall: %.4f | F1: %.4f",
                summary["test_metrics"]["roc_auc"],
                summary["test_metrics"]["pr_auc"],
                summary["test_metrics"]["recall"],
                summary["test_metrics"]["f1"])
    logger.info("Artifacts saved to: %s", ARTIFACTS_DIR)
    logger.info("Reports saved to:   %s", REPORTS_DIR)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
