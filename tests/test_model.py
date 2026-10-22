"""Unit and integration tests for model training, metrics calculation, and serialization."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from src.data_loader import load_raw_data
from src.evaluate import calculate_metrics, analyze_thresholds
from src.preprocessing import create_preprocessor, split_data
from src.train import get_candidate_models, ARTIFACTS_DIR


@pytest.fixture(scope="module")
def train_test_data():
    df, _ = load_raw_data()
    return split_data(df, test_size=0.20, random_state=42)


def test_candidate_models_instantiation():
    """Verify all 4 required candidate models are defined."""
    candidates = get_candidate_models()
    expected_models = ["DummyBaseline", "LogisticRegression", "RandomForest", "SVC"]
    for model_name in expected_models:
        assert model_name in candidates, f"Missing required model: {model_name}"


def test_calculate_metrics_correctness():
    """Verify calculate_metrics outputs with known synthetic inputs."""
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.4, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_pred_proba, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1"] == 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0

    cm = metrics["confusion_matrix"]
    assert cm["true_negatives"] == 2
    assert cm["true_positives"] == 2
    assert cm["false_negatives"] == 0
    assert cm["false_positives"] == 0


def test_saved_model_artifact_loading_and_prediction(train_test_data):
    """Verify saved best_model_pipeline loads and produces valid predictions."""
    model_path = ARTIFACTS_DIR / "best_model_pipeline.joblib"
    assert model_path.exists(), "Trained model pipeline artifact missing."

    pipeline = joblib.load(model_path)
    assert isinstance(pipeline, Pipeline)

    X_train, X_test, y_train, y_test = train_test_data

    # Predict on test slice
    test_slice = X_test.head(10)
    probas = pipeline.predict_proba(test_slice)

    # Probabilities must be shape (N, 2)
    assert probas.shape == (10, 2)

    # Probabilities must sum to 1.0 across classes
    np.testing.assert_allclose(probas.sum(axis=1), np.ones(10), atol=1e-5)

    # All probabilities must be bounded strictly between 0 and 1
    assert (probas >= 0.0).all() and (probas <= 1.0).all()

    # Binary prediction
    preds = pipeline.predict(test_slice)
    assert len(preds) == 10
    assert set(preds).issubset({0, 1})


def test_threshold_analysis_monotonicity_or_bounds(train_test_data):
    """Verify threshold analysis returns valid metrics within [0, 1]."""
    model_path = ARTIFACTS_DIR / "best_model_pipeline.joblib"
    pipeline = joblib.load(model_path)
    X_train, X_test, y_train, y_test = train_test_data

    probas = pipeline.predict_proba(X_test)[:, 1]
    th_df = analyze_thresholds(y_test, probas, thresholds=np.array([0.2, 0.5, 0.8]))

    assert len(th_df) == 3
    for col in ["precision", "recall", "f1_score"]:
        assert (th_df[col] >= 0.0).all() and (th_df[col] <= 1.0).all()

    # Lower threshold should yield higher or equal recall
    assert th_df.loc[th_df["threshold"] == 0.2, "recall"].values[0] >= th_df.loc[th_df["threshold"] == 0.8, "recall"].values[0]
