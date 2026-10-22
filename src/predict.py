"""Reusable inference engine for Customer Churn Prediction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.explain import explain_local_customer_prediction
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_MAP_INVERSE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACTS_DIR / "best_model_pipeline.joblib"
DEFAULT_CONFIG_PATH = DEFAULT_ARTIFACTS_DIR / "feature_config.json"


class ModelArtifactNotFoundError(FileNotFoundError):
    """Raised when trained model pipeline artifacts are missing."""
    pass


class ChurnPredictor:
    """Production-grade reusable inference engine for churn probability scoring."""

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        config_path: Optional[Path | str] = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.pipeline = None
        self.feature_config: Dict[str, Any] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads serialized model pipeline and schema definitions."""
        if not self.model_path.exists():
            raise ModelArtifactNotFoundError(
                f"Model artifact not found at: {self.model_path.resolve()}\n"
                "Please train the model first by executing: python main.py"
            )

        logger.info("Loading model pipeline from %s", self.model_path)
        # Note: Serialized joblib pipelines execute arbitrary code on load; load only from trusted sources.
        self.pipeline = joblib.load(self.model_path)

        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.feature_config = json.load(f)
        else:
            self.feature_config = {
                "numerical_features": NUMERICAL_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
            }

    def validate_and_prepare(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """Validates incoming feature columns and cleans types for inference.

        Parameters
        ----------
        data : Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
            Input record or batch of records.

        Returns
        -------
        pd.DataFrame
            Sanitized DataFrame matching the training feature schema.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}. Expected dict, list of dicts, or DataFrame.")

        # Ensure all expected numerical features exist
        for col in NUMERICAL_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
            else:
                # Handle blank strings safely
                df[col] = pd.to_numeric(df[col].astype(str).str.strip().replace("", np.nan), errors="coerce")
                df[col] = df[col].fillna(0.0)

        # Ensure all expected categorical features exist
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = "Missing"
            else:
                df[col] = df[col].astype(str).fillna("Missing")

        # Keep strictly feature columns in order
        feature_order = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        return df[feature_order]

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Performs batch or single-record churn prediction.

        Parameters
        ----------
        data : Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
            Customer feature attributes.
        threshold : float
            Decision boundary probability threshold (0.0 - 1.0). Default is 0.5.

        Returns
        -------
        pd.DataFrame
            DataFrame with predictions: churn_probability, predicted_churn, predicted_label, risk_category.
        """
        df_clean = self.validate_and_prepare(data)
        probabilities = self.pipeline.predict_proba(df_clean)[:, 1]
        binary_preds = (probabilities >= threshold).astype(int)

        def assign_risk_category(prob: float) -> str:
            if prob >= 0.60:
                return "High Risk"
            elif prob >= 0.35:
                return "Medium Risk"
            else:
                return "Low Risk"

        results = pd.DataFrame({
            "churn_probability": np.round(probabilities, 4),
            "predicted_churn": binary_preds,
            "predicted_label": [TARGET_MAP_INVERSE.get(p, "No") for p in binary_preds],
            "risk_category": [assign_risk_category(p) for p in probabilities],
        })

        return results

    def predict_single_with_explanation(
        self,
        customer_dict: Dict[str, Any],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Provides a detailed single-subscriber prediction with local risk factors."""
        df_clean = self.validate_and_prepare(customer_dict)
        pred_df = self.predict(df_clean, threshold=threshold)
        row_res = pred_df.iloc[0].to_dict()

        local_expl = explain_local_customer_prediction(self.pipeline, df_clean)

        return {
            "prediction": row_res,
            "threshold_applied": threshold,
            "explanation": local_expl,
        }
