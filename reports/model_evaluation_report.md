# Customer Churn Intelligence — Model Evaluation Report

**Generated on:** 2026-10-22 14:54:13  
**Primary Selected Architecture:** `RandomForest`  
**Hyperparameter Tuning:** `{'classifier__max_depth': 8, 'classifier__min_samples_split': 10, 'classifier__n_estimators': 200}`  

---

## 1. Executive Summary & Business Rationale

In subscription businesses, identifying customers at risk of churn before they cancel is critical for revenue retention. 
Because the cost of proactive customer retention outreach (a discount, customer success call, or perk) is drastically smaller than the customer lifetime value (LTV) lost from customer departure, **minimizing False Negatives (missed churners) while maintaining strong ROC-AUC is our primary technical objective**.

The primary model was chosen based on 5-fold Stratified Cross-Validation on the training partition and subsequently evaluated once on the untouched held-out test set (20% sample, 1409 customers).

---

## 2. 5-Fold Stratified Cross-Validation Benchmark

| Model | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) | Recall (Mean ± Std) | F1-Score (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- |
| **RandomForest** | 0.8468 ± 0.0101 | 0.6593 ± 0.0181 | 0.7933 ± 0.0235 | 0.6338 ± 0.0147 |
| **LogisticRegression** | 0.8460 ± 0.0124 | 0.6601 ± 0.0196 | 0.8013 ± 0.0379 | 0.6282 ± 0.0231 |
| **SVC** | 0.8267 ± 0.0096 | 0.6065 ± 0.0257 | 0.7799 ± 0.0373 | 0.6236 ± 0.0194 |
| **DummyBaseline** | 0.5000 ± 0.0000 | 0.2654 ± 0.0001 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |


---

## 3. Final Evaluation on Untouched Test Set

- **Held-Out Test Samples:** 1409
- **True Churn Rate in Test Set:** 26.54%
- **Decision Threshold Applied:** 0.5

### Performance Metrics Table

| Metric | Measured Value | Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8432** | Overall ranking capability across all thresholds |
| **PR-AUC (Avg Precision)** | **0.6533** | Performance specifically on the minority churn class |
| **Recall (Sensitivity)** | **0.7941** | Percentage of actual churners caught |
| **Precision** | **0.5238** | Percentage of flagged accounts who actually churn |
| **F1-Score** | **0.6312** | Harmonic mean of precision and recall |
| **Brier Score** | **0.1625** | Probability calibration accuracy (lower is better) |

### Test Confusion Matrix

| | Predicted Retained (0) | Predicted Churn (1) |
| :--- | :--- | :--- |
| **Actual Retained (0)** | **765** (True Negatives) | **270** (False Positives) |
| **Actual Churn (1)** | **77** (False Negatives) | **297** (True Positives) |

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
