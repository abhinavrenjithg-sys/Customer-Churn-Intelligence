"""Explainability module for global and local customer churn risk drivers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.preprocessing import get_feature_names_from_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COLOR_CHURN = "#E11D48"     # Rose / Crimson (Increases churn risk)
COLOR_RETAINED = "#2563EB"  # Royal Blue (Protective against churn)
FIG_DPI = 300

CAUSALITY_DISCLAIMER = (
    "Disclaimer: These feature associations represent statistical correlations identified "
    "by the machine learning model, not confirmed causal mechanisms. A high feature weight indicates "
    "strong predictive association in historical subscriber data, but does not guarantee that altering "
    "this attribute will independently prevent churn for an individual subscriber."
)


def extract_global_feature_importance(
    pipeline: Pipeline,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extracts model-native global feature importance or coefficients from a fitted pipeline.

    For Logistic Regression: Returns coefficients, odds ratios (exp(beta)), and direction.
    For Tree/Forest/Boosting: Returns feature importances (normalized sum to 1.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['feature', 'importance', 'direction', 'odds_ratio' (if linear)]
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    if feature_names is None:
        feature_names = get_feature_names_from_preprocessor(preprocessor)

    # 1. Linear Model (e.g. LogisticRegression)
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        odds_ratios = np.exp(coefs)
        directions = ["Increases Churn Risk" if c > 0 else "Protective (Reduces Churn)" for c in coefs]
        df_imp = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "importance": np.abs(coefs),
            "odds_ratio": odds_ratios,
            "direction": directions,
        }).sort_values(by="importance", ascending=False)
        return df_imp

    # 2. Tree-based Model (e.g. RandomForest, XGBoost)
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
            "direction": ["Non-linear association" for _ in importances],
        }).sort_values(by="importance", ascending=False)
        return df_imp

    else:
        logger.warning("Model does not expose native coef_ or feature_importances_.")
        return pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names))})


def compute_permutation_importance(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    scoring: str = "roc_auc",
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Computes model-agnostic permutation importance on validation or test features."""
    logger.info("Computing permutation importance with scoring=%s ...", scoring)
    result = permutation_importance(
        pipeline,
        X_val,
        y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    df_perm = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values(by="importance_mean", ascending=False)

    return df_perm


def plot_global_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path | str,
    top_n: int = 15,
    title: str = "Top Predictive Features (Global Association)",
) -> Path:
    """Renders and saves a horizontal bar chart of top predictive features."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_df = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=FIG_DPI)

    # Clean feature names for display
    clean_names = [f.replace("cat__", "").replace("num__", "") for f in top_df["feature"]]

    if "coefficient" in top_df.columns:
        colors = [COLOR_CHURN if c > 0 else COLOR_RETAINED for c in top_df["coefficient"]]
        bars = ax.barh(clean_names, top_df["coefficient"], color=colors, height=0.6, alpha=0.88)
        ax.set_xlabel("Logistic Regression Coefficient (Positive = Higher Churn Odds)", fontweight="bold")
        ax.axvline(x=0, color="#64748B", linestyle="--", lw=1.2)
    else:
        bars = ax.barh(clean_names, top_df["importance"], color="#6366F1", height=0.6, alpha=0.88)
        ax.set_xlabel("Relative Importance Score", fontweight="bold")

    ax.set_title(title, pad=15, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def explain_local_customer_prediction(
    pipeline: Pipeline,
    customer_df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Generates a localized feature contribution breakdown for an individual customer profile.

    Parameters
    ----------
    pipeline : Pipeline
        Trained end-to-end pipeline.
    customer_df : pd.DataFrame
        Single-row DataFrame representing the customer.
    top_k : int
        Number of top risk and protective factors to extract.

    Returns
    -------
    Dict[str, Any]
        Structured explanation payload with contribution scores and plain-English narrative.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    if feature_names is None:
        feature_names = get_feature_names_from_preprocessor(preprocessor)

    # Transform single customer
    transformed_x = preprocessor.transform(customer_df)
    if hasattr(transformed_x, "toarray"):
        transformed_x = transformed_x.toarray()
    transformed_x = transformed_x.flatten()

    prob_churn = float(pipeline.predict_proba(customer_df)[0, 1])

    risk_factors: List[Dict[str, Any]] = []
    protective_factors: List[Dict[str, Any]] = []

    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        contributions = transformed_x * coefs

        contrib_df = pd.DataFrame({
            "feature": feature_names,
            "contribution": contributions,
            "feature_val": transformed_x,
            "coefficient": coefs,
        })

        # Features with non-zero impact
        active = contrib_df[np.abs(contrib_df["contribution"]) > 1e-4].copy()

        # Risk drivers: positive contribution (pushing toward churn)
        risk_df = active[active["contribution"] > 0].sort_values(by="contribution", ascending=False).head(top_k)
        for _, row in risk_df.iterrows():
            clean_feat = row["feature"].replace("cat__", "").replace("num__", "")
            risk_factors.append({
                "feature": clean_feat,
                "contribution": round(float(row["contribution"]), 3),
                "odds_impact": f"+{round(float(row['contribution']), 2)} log-odds",
            })

        # Protective drivers: negative contribution (pushing toward retention)
        prot_df = active[active["contribution"] < 0].sort_values(by="contribution", ascending=True).head(top_k)
        for _, row in prot_df.iterrows():
            clean_feat = row["feature"].replace("cat__", "").replace("num__", "")
            protective_factors.append({
                "feature": clean_feat,
                "contribution": round(float(row["contribution"]), 3),
                "odds_impact": f"{round(float(row['contribution']), 2)} log-odds",
            })

    elif hasattr(model, "feature_importances_"):
        # Use TreeSHAP for accurate local contribution
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            # transformed_x is 1D array of shape (n_features,)
            X_2d = transformed_x.reshape(1, -1)
            shap_raw = explainer.shap_values(X_2d)

            # Handle shap 3D array or list
            if isinstance(shap_raw, list):
                shap_vals = np.asarray(shap_raw[1])[0]
            elif isinstance(shap_raw, np.ndarray) and len(shap_raw.shape) == 3:
                shap_vals = shap_raw[0, :, 1]
            elif isinstance(shap_raw, np.ndarray) and len(shap_raw.shape) == 2:
                shap_vals = shap_raw[0]
            else:
                shap_vals = np.asarray(shap_raw).flatten()

            contrib_df = pd.DataFrame({
                "feature": feature_names,
                "contribution": shap_vals,
            })

            # Risk drivers: pushing probability towards churn (positive SHAP)
            risk_df = contrib_df[contrib_df["contribution"] > 0].sort_values(by="contribution", ascending=False).head(top_k)
            for _, row in risk_df.iterrows():
                clean_feat = row["feature"].replace("cat__", "").replace("num__", "")
                risk_factors.append({
                    "feature": clean_feat,
                    "contribution": round(float(row["contribution"]), 3),
                    "odds_impact": f"+{round(float(row['contribution']) * 100, 1)}% probability shift",
                })

            # Protective drivers: pushing probability towards retention (negative SHAP)
            prot_df = contrib_df[contrib_df["contribution"] < 0].sort_values(by="contribution", ascending=True).head(top_k)
            for _, row in prot_df.iterrows():
                clean_feat = row["feature"].replace("cat__", "").replace("num__", "")
                protective_factors.append({
                    "feature": clean_feat,
                    "contribution": round(float(row["contribution"]), 3),
                    "odds_impact": f"{round(float(row['contribution']) * 100, 1)}% probability shift",
                })
        except Exception as exc:
            logger.warning("TreeSHAP calculation failed, falling back to top global importance: %s", exc)
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:top_k]
            for idx in top_indices:
                clean_feat = feature_names[idx].replace("cat__", "").replace("num__", "")
                risk_factors.append({
                    "feature": clean_feat,
                    "contribution": round(float(importances[idx]), 3),
                    "odds_impact": "High global impact",
                })

    # Generate plain-English synthesis
    narrative_lines: List[str] = [
        f"The predicted churn probability is {prob_churn * 100:.1f}%.",
    ]

    if risk_factors:
        top_risks = ", ".join([rf["feature"] for rf in risk_factors[:3]])
        narrative_lines.append(f"Primary factors associated with churn risk: {top_risks}.")
    if protective_factors:
        top_prots = ", ".join([pf["feature"] for pf in protective_factors[:3]])
        narrative_lines.append(f"Strongest retention indicators: {top_prots}.")

    return {
        "churn_probability": round(prob_churn, 4),
        "risk_level": "High Risk" if prob_churn >= 0.60 else ("Medium Risk" if prob_churn >= 0.35 else "Low Risk"),
        "top_risk_factors": risk_factors,
        "top_protective_factors": protective_factors,
        "plain_english_summary": " ".join(narrative_lines),
        "disclaimer": CAUSALITY_DISCLAIMER,
    }
