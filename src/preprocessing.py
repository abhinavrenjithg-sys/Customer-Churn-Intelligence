"""Leakage-safe preprocessing and dataset partitioning module."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

NUMERICAL_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURE_COLUMNS: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TARGET_MAP = {"No": 0, "Yes": 1}
TARGET_MAP_INVERSE = {0: "No", 1: "Yes"}


def create_preprocessor(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for numerical and categorical features.

    Numerical pipeline:
      - Median imputation (safe against outliers and zero-tenure artifacts)
      - Standard scaling (zero mean, unit variance)

    Categorical pipeline:
      - Most frequent imputer
      - OneHotEncoder with handle_unknown='ignore' (leakage and OOV safe)

    Returns
    -------
    ColumnTransformer
        Unfitted transformer ready to embed inside an Estimator Pipeline.
    """
    num_cols = numerical_cols if numerical_cols is not None else NUMERICAL_FEATURES
    cat_cols = categorical_cols if categorical_cols is not None else CATEGORICAL_FEATURES

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits dataset into stratified train and test partitions, isolating features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Validated raw or cleaned dataframe.
    test_size : float
        Fraction reserved for final held-out test evaluation. Default is 0.20 (20%).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not present in dataframe.")

    # Drop ID column from feature space
    feature_cols = [c for c in df.columns if c not in [ID_COLUMN, TARGET_COLUMN]]
    X = df[feature_cols].copy()

    # Encode target strictly to binary 0/1
    if df[TARGET_COLUMN].dtype == object or pd.api.types.is_string_dtype(df[TARGET_COLUMN]):
        y = df[TARGET_COLUMN].map(TARGET_MAP)
        if y.isna().any():
            raise ValueError(f"Target contains unmapped values: {df[TARGET_COLUMN].unique()}")
        y = y.astype(int)
    else:
        y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Data split complete: Train shape=%s (Churn rate=%.2f%%), Test shape=%s (Churn rate=%.2f%%)",
        X_train.shape,
        y_train.mean() * 100,
        X_test.shape,
        y_test.mean() * 100,
    )

    return X_train, X_test, y_train, y_test


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    """Extracts output feature names from a fitted ColumnTransformer."""
    feature_names: List[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "named_steps"):
            # Check pipeline last step
            last_step = trans.steps[-1][1]
            if hasattr(last_step, "get_feature_names_out"):
                names = last_step.get_feature_names_out(cols)
                feature_names.extend(names)
            else:
                feature_names.extend(cols)
        elif hasattr(trans, "get_feature_names_out"):
            feature_names.extend(trans.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)
    return [str(fn) for fn in feature_names]


def save_feature_config(output_path: Path | str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Saves metadata about the feature set, types, and categorical options to JSON."""
    config: Dict[str, Any] = {
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "target_mapping": TARGET_MAP,
        "inverse_target_mapping": TARGET_MAP_INVERSE,
    }

    if df is not None:
        categories_dict: Dict[str, List[Any]] = {}
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                unique_vals = sorted([str(v) for v in df[col].dropna().unique().tolist()])
                categories_dict[col] = unique_vals
        config["categorical_options"] = categories_dict

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("Saved feature configuration to %s", output_path)
    return config
