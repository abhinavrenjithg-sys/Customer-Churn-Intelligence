"""Unit and integration tests for inference engine, unseen category handling, and explanations."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.predict import ChurnPredictor, ModelArtifactNotFoundError


@pytest.fixture(scope="module")
def predictor():
    return ChurnPredictor()


@pytest.fixture
def sample_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


def test_single_prediction(predictor, sample_customer):
    """Verify single customer dictionary inference."""
    result = predictor.predict(sample_customer)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "churn_probability" in result.columns
    assert "predicted_churn" in result.columns
    assert "predicted_label" in result.columns
    assert "risk_category" in result.columns

    prob = result["churn_probability"].iloc[0]
    assert 0.0 <= prob <= 1.0
    assert result["predicted_churn"].iloc[0] in [0, 1]
    assert result["predicted_label"].iloc[0] in ["No", "Yes"]
    assert result["risk_category"].iloc[0] in ["Low Risk", "Medium Risk", "High Risk"]


def test_batch_prediction(predictor, sample_customer):
    """Verify batch inference on a DataFrame of multiple customers."""
    cust2 = sample_customer.copy()
    cust2["tenure"] = 65
    cust2["Contract"] = "Two year"
    cust2["MonthlyCharges"] = 19.50
    cust2["TotalCharges"] = 1200.0

    batch_df = pd.DataFrame([sample_customer, cust2])
    results = predictor.predict(batch_df)

    assert len(results) == 2
    # Loyal customer on two-year contract should have lower churn probability than 1-month customer
    prob_month_to_month = results["churn_probability"].iloc[0]
    prob_two_year = results["churn_probability"].iloc[1]
    assert prob_two_year < prob_month_to_month


def test_unseen_categories_handling(predictor, sample_customer):
    """Ensure out-of-vocabulary categorical levels do not trigger exceptions (OneHotEncoder handle_unknown='ignore')."""
    oov_customer = sample_customer.copy()
    oov_customer["PaymentMethod"] = "Cryptocurrency / Web3"
    oov_customer["InternetService"] = "Quantum Laser Satellite"
    oov_customer["Contract"] = "Decade Lock-in"

    result = predictor.predict(oov_customer)
    assert len(result) == 1
    assert 0.0 <= result["churn_probability"].iloc[0] <= 1.0


def test_missing_features_imputation(predictor, sample_customer):
    """Ensure missing columns in input are imputed cleanly rather than throwing unhandled errors."""
    incomplete_customer = {
        "tenure": 12,
        "Contract": "One year",
        "MonthlyCharges": 45.0,
        # Missing all other fields
    }

    result = predictor.predict(incomplete_customer)
    assert len(result) == 1
    assert 0.0 <= result["churn_probability"].iloc[0] <= 1.0


def test_single_prediction_with_explanation(predictor, sample_customer):
    """Verify detailed prediction payload including local explanation and disclaimer."""
    res = predictor.predict_single_with_explanation(sample_customer)

    assert "prediction" in res
    assert "explanation" in res
    assert "threshold_applied" in res

    expl = res["explanation"]
    assert "churn_probability" in expl
    assert "risk_level" in expl
    assert "top_risk_factors" in expl
    assert "top_protective_factors" in expl
    assert "plain_english_summary" in expl
    assert "disclaimer" in expl
    assert "not confirmed causal mechanisms" in expl["disclaimer"]


def test_missing_model_raises_custom_exception(tmp_path):
    """Verify that specifying a non-existent model path raises ModelArtifactNotFoundError."""
    with pytest.raises(ModelArtifactNotFoundError) as exc_info:
        ChurnPredictor(model_path=tmp_path / "absent_model.joblib")
    assert "python main.py" in str(exc_info.value)


def test_app_helpers_and_loaders():
    """Verify Streamlit app utility loaders and data availability."""
    import app
    metrics_data = app.load_metrics_and_summary()
    assert metrics_data is not None
    assert "test_metrics" in metrics_data
    assert "selected_model" in metrics_data

    cv_df = app.load_cv_results()
    assert cv_df is not None
    assert "model" in cv_df.columns
    assert len(cv_df) >= 4
