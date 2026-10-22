"""Data loading, validation, and quality assessment module for Customer Churn Prediction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DEFAULT_DATA_FILE = DEFAULT_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

EXPECTED_NUMERICAL_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

EXPECTED_CATEGORICAL_COLUMNS = [
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

EXPECTED_ALL_COLUMNS = [ID_COLUMN] + EXPECTED_CATEGORICAL_COLUMNS + EXPECTED_NUMERICAL_COLUMNS + [TARGET_COLUMN]


class DataValidationError(Exception):
    """Raised when the dataset fails schema or integrity validation."""
    pass


def load_raw_data(
    filepath: Optional[Path | str] = None,
    handle_blank_total_charges: bool = True,
    fill_value_blank_total_charges: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads and validates the IBM Telco Customer Churn dataset.

    Parameters
    ----------
    filepath : Optional[Path | str]
        Path to the CSV file. If None, uses the standard location `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
    handle_blank_total_charges : bool
        If True, converts blank string entries in 'TotalCharges' (which correspond to tenure=0 customers)
        to the specified numerical `fill_value_blank_total_charges`.
    fill_value_blank_total_charges : float
        Replacement value for zero-tenure blank 'TotalCharges', default is 0.0.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        The loaded and validated DataFrame, alongside a comprehensive data quality report.

    Raises
    ------
    FileNotFoundError
        If the file cannot be located.
    DataValidationError
        If mandatory columns or the target variable are missing.
    """
    if filepath is None:
        target_path = DEFAULT_DATA_FILE
    else:
        target_path = Path(filepath)

    if not target_path.exists():
        download_hint = (
            f"Dataset not found at expected path: {target_path.resolve()}\n"
            "Please obtain 'WA_Fn-UseC_-Telco-Customer-Churn.csv' from Kaggle:\n"
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            f"and place it inside: {DEFAULT_DATA_DIR.resolve()}"
        )
        logger.error(download_hint)
        raise FileNotFoundError(download_hint)

    logger.info("Loading dataset from %s", target_path)
    try:
        df = pd.read_csv(target_path)
    except Exception as exc:
        raise DataValidationError(f"Failed to read CSV at {target_path}: {exc}") from exc

    # Validate presence of expected columns
    missing_cols = [c for c in [ID_COLUMN, TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Critical columns missing from dataset: {missing_cols}. Expected columns: {EXPECTED_ALL_COLUMNS}"
        )

    # Initial shape & duplicate checks
    initial_shape = df.shape
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df.columns else 0

    if duplicate_ids > 0:
        logger.warning("Found %d duplicate customerIDs in dataset!", duplicate_ids)

    # Target class distribution check
    target_counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
    target_percentages = (
        (df[TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100).round(2).to_dict()
    )

    # Inspect TotalCharges blank spaces and tenure relation
    blank_total_charges_mask = pd.Series(False, index=df.index)
    if "TotalCharges" in df.columns:
        if pd.api.types.is_string_dtype(df["TotalCharges"]) or pd.api.types.is_object_dtype(df["TotalCharges"]):
            blank_total_charges_mask = df["TotalCharges"].astype(str).str.strip().isin(["", "nan", "null"])
        else:
            blank_total_charges_mask = df["TotalCharges"].isna()

    blank_count = int(blank_total_charges_mask.sum())
    tenure_for_blanks = []
    if blank_count > 0:
        tenure_for_blanks = df.loc[blank_total_charges_mask, "tenure"].tolist()
        logger.info(
            "Identified %d records with blank/empty 'TotalCharges'. All associated tenures: %s",
            blank_count,
            set(tenure_for_blanks),
        )

    # Convert TotalCharges safely
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", np.nan), errors="coerce")
        if handle_blank_total_charges and blank_count > 0:
            logger.info(
                "Converting %d blank 'TotalCharges' values to %s because tenure == 0 for these customers.",
                blank_count,
                fill_value_blank_total_charges,
            )
            df.loc[blank_total_charges_mask, "TotalCharges"] = fill_value_blank_total_charges

    # Construct comprehensive data quality report
    missing_values = {k: int(v) for k, v in df.isna().sum().items() if v > 0}
    data_quality_report = {
        "source_path": str(target_path.resolve()),
        "rows": initial_shape[0],
        "columns": initial_shape[1],
        "duplicate_rows": duplicate_rows,
        "duplicate_customer_ids": duplicate_ids,
        "blank_total_charges_count": blank_count,
        "blank_total_charges_tenures": list(set(tenure_for_blanks)),
        "all_blank_total_charges_have_zero_tenure": all(t == 0 for t in tenure_for_blanks) if tenure_for_blanks else True,
        "missing_values_after_conversion": missing_values,
        "target_class_counts": target_counts,
        "target_class_percentages": target_percentages,
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    return df, data_quality_report


def save_data_quality_report(report: Dict[str, Any], output_path: Path | str) -> None:
    """Serializes the data quality assessment report to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved data quality report to %s", output_path)
