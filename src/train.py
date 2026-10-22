"""Model training, 5-fold cross-validation benchmarking, tuning, and evaluation module."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.evaluate import (
    analyze_thresholds,
    calculate_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_analysis,
)
from src.explain import (
    extract_global_feature_importance,
    plot_global_feature_importance,
)
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    create_preprocessor,
    get_feature_names_from_preprocessor,
    save_feature_config,
    split_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Instantiates candidate estimators for fair benchmarking."""
    return {
        "DummyBaseline": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "SVC": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def run_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """Executes 5-fold Stratified Cross-Validation on all candidate models.

    Ensures that preprocessing is fitted ONLY within each training fold,
    preventing data leakage.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Pipeline]]
        Summary DataFrame with mean and std for each metric, plus full candidate pipelines.
    """
    logger.info("Executing %d-Fold Stratified Cross-Validation benchmark ...", n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    candidates = get_candidate_models(random_state)
    records = []
    pipelines: Dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        logger.info("Evaluating candidate pipeline: %s", name)
        preprocessor = create_preprocessor()
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])
        pipelines[name] = pipe

        cv_res = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        row: Dict[str, Any] = {"model": name}
        for metric in scoring.keys():
            test_scores = cv_res[f"test_{metric}"]
            row[f"test_{metric}_mean"] = round(float(np.mean(test_scores)), 4)
            row[f"test_{metric}_std"] = round(float(np.std(test_scores)), 4)
        records.append(row)

    results_df = pd.DataFrame(records).sort_values(by="test_roc_auc_mean", ascending=False).reset_index(drop=True)
    return results_df, pipelines


def tune_best_model(
    top_model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Conducts hyperparameter optimization using GridSearchCV."""
    logger.info("Tuning top candidate architecture: %s", top_model_name)
    preprocessor = create_preprocessor()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    if top_model_name == "LogisticRegression":
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000)),
        ])
        param_grid = {
            "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            "classifier__solver": ["lbfgs", "liblinear"],
        }
    elif top_model_name == "RandomForest":
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1)),
        ])
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [6, 8, 12],
            "classifier__min_samples_split": [2, 5, 10],
        }
    else:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000)),
        ])
        param_grid = {
            "classifier__C": [0.05, 0.1, 1.0, 5.0],
        }

    grid_search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = float(grid_search.best_score_)
    logger.info("Tuning complete. Best ROC-AUC: %.4f, Best params: %s", best_score, best_params)

    return best_pipeline, {"best_params": best_params, "best_cv_roc_auc": round(best_score, 4)}


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    artifacts_dir: Optional[Path | str] = None,
    figures_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Runs the complete training, benchmarking, tuning, and test-set evaluation workflow.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset from data_loader.
    test_size : float
        Holdout proportion for final test evaluation (default 0.20).
    random_state : int
        Seed for reproducibility.
    artifacts_dir : Optional[Path | str]
        Directory to save models and config. Defaults to `artifacts/`.
    figures_dir : Optional[Path | str]
        Directory to save evaluation plots. Defaults to `reports/figures/`.

    Returns
    -------
    Dict[str, Any]
        Summary dictionary containing genuine metrics and file paths.
    """
    art_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    fig_dir = Path(figures_dir) if figures_dir else FIGURES_DIR
    art_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stratified Partitioning
    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size, random_state=random_state)

    # 2. Save feature configuration for inference
    feature_config = save_feature_config(art_dir / "feature_config.json", df=df)

    # 3. 5-Fold Stratified Cross-Validation on all candidates
    cv_results_df, _ = run_cross_validation(X_train, y_train, n_splits=5, random_state=random_state)
    cv_results_df.to_csv(art_dir / "cv_results.csv", index=False)
    logger.info("Cross-validation summary:\n%s", cv_results_df.to_string())

    # Generate Model Comparison Chart
    plot_model_comparison(cv_results_df, fig_dir / "model_comparison.png")

    # 4. Select Primary Model based on documented business objective
    # Objective: In churn retention, false negatives are far more costly than false positives.
    # We choose the model with top ROC-AUC and high Recall/PR-AUC that permits calibrated probabilities.
    # Exclude DummyBaseline
    trained_candidates = cv_results_df[cv_results_df["model"] != "DummyBaseline"]
    top_model_name = str(trained_candidates.iloc[0]["model"])
    logger.info(
        "Selecting '%s' as primary architecture based on top cross-validated ROC-AUC (%.4f) & PR-AUC (%.4f).",
        top_model_name,
        trained_candidates.iloc[0]["test_roc_auc_mean"],
        trained_candidates.iloc[0]["test_pr_auc_mean"],
    )

    # 5. Hyperparameter Tuning
    final_pipeline, tuning_info = tune_best_model(top_model_name, X_train, y_train, random_state=random_state)

    # 6. Final Evaluation on Untouched Test Set
    logger.info("Evaluating finalized pipeline ONCE on untouched held-out test set ...")
    y_test_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred_bin = final_pipeline.predict(X_test)

    # Calculate real measured test metrics
    test_metrics = calculate_metrics(y_test, y_test_pred_proba, threshold=0.5)

    # 7. Generate Evaluation Figures
    plot_confusion_matrix(y_test, y_test_pred_bin, fig_dir / "confusion_matrix.png", title=f"Test Confusion Matrix ({top_model_name})")
    plot_roc_curve(y_test, y_test_pred_proba, fig_dir / "roc_curve.png", model_name=top_model_name)
    plot_precision_recall_curve(y_test, y_test_pred_proba, fig_dir / "precision_recall_curve.png", model_name=top_model_name)

    # Threshold analysis
    th_df = analyze_thresholds(y_test, y_test_pred_proba)
    th_df.to_csv(art_dir / "threshold_analysis.csv", index=False)
    plot_threshold_analysis(th_df, fig_dir / "threshold_analysis.png")

    # 8. Global Feature Explainability
    importance_df = extract_global_feature_importance(final_pipeline)
    importance_df.to_csv(art_dir / "feature_importance.csv", index=False)
    plot_global_feature_importance(importance_df, fig_dir / "feature_importance.png", top_n=15)

    # 9. Model Persistence
    # Scikit-learn security note: joblib serialized pipelines should only be loaded from trusted sources.
    model_save_path = art_dir / "best_model_pipeline.joblib"
    joblib.dump(final_pipeline, model_save_path)
    logger.info("Persisted best model pipeline to %s", model_save_path)

    # Compile run summary
    run_summary: Dict[str, Any] = {
        "selected_model": top_model_name,
        "tuning_info": tuning_info,
        "cross_validation_results": cv_results_df.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_churn_rate": round(float(y_test.mean()), 4),
        "artifacts": {
            "model_pipeline": str(model_save_path.resolve()),
            "feature_config": str((art_dir / "feature_config.json").resolve()),
            "cv_results": str((art_dir / "cv_results.csv").resolve()),
            "threshold_analysis": str((art_dir / "threshold_analysis.csv").resolve()),
            "feature_importance": str((art_dir / "feature_importance.csv").resolve()),
        },
        "figures": {
            "model_comparison": str((fig_dir / "model_comparison.png").resolve()),
            "confusion_matrix": str((fig_dir / "confusion_matrix.png").resolve()),
            "roc_curve": str((fig_dir / "roc_curve.png").resolve()),
            "precision_recall_curve": str((fig_dir / "precision_recall_curve.png").resolve()),
            "threshold_analysis": str((fig_dir / "threshold_analysis.png").resolve()),
            "feature_importance": str((fig_dir / "feature_importance.png").resolve()),
        },
    }

    # Save metrics JSON
    with open(art_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    logger.info("Training and evaluation completed successfully!")
    logger.info("Test ROC-AUC: %.4f | Test PR-AUC: %.4f | Test Recall: %.4f | Test F1: %.4f",
                test_metrics["roc_auc"], test_metrics["pr_auc"], test_metrics["recall"], test_metrics["f1"])

    return run_summary
