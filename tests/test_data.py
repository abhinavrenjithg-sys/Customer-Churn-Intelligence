"""Unit and integration tests for data loading, schema integrity, and validation."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    DEFAULT_DATA_FILE,
    EXPECTED_ALL_COLUMNS,
    EXPECTED_CATEGORICAL_COLUMNS,
    EXPECTED_NUMERICAL_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    DataValidationError,
    load_raw_data,
)


def test_raw_data_file_exists():
    """Verify that the authentic raw CSV file is present in data/raw/."""
    assert DEFAULT_DATA_FILE.exists(), f"Expected dataset missing at {DEFAULT_DATA_FILE}"


def test_load_raw_data_schema():
    """Verify loaded dataset shape, columns, and target column."""
    df, report = load_raw_data(DEFAULT_DATA_FILE)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 7043
    assert df.shape[1] == 21

    # Check all expected columns exist
    for col in EXPECTED_ALL_COLUMNS:
        assert col in df.columns, f"Missing expected column: {col}"

    assert ID_COLUMN in df.columns
    assert TARGET_COLUMN in df.columns


def test_customer_id_uniqueness():
    """Verify that all customerIDs are unique."""
    df, report = load_raw_data(DEFAULT_DATA_FILE)
    assert report["duplicate_customer_ids"] == 0
    assert df[ID_COLUMN].nunique() == len(df)


def test_total_charges_conversion_and_zero_tenure():
    """Test safe conversion of TotalCharges and validation of zero-tenure blank entries."""
    df, report = load_raw_data(DEFAULT_DATA_FILE, handle_blank_total_charges=True, fill_value_blank_total_charges=0.0)

    # TotalCharges should now be float
    assert pd.api.types.is_float_dtype(df["TotalCharges"])

    # Exactly 11 records had blank string TotalCharges
    assert report["blank_total_charges_count"] == 11

    # All 11 blank records must have tenure == 0
    assert report["all_blank_total_charges_have_zero_tenure"] is True

    # No missing values remaining after fill
    assert df["TotalCharges"].isna().sum() == 0

    # Specifically check zero tenure rows
    zero_tenure_rows = df[df["tenure"] == 0]
    assert len(zero_tenure_rows) == 11
    assert (zero_tenure_rows["TotalCharges"] == 0.0).all()


def test_target_distribution_and_values():
    """Verify the target variable contains only 'Yes' and 'No'."""
    df, report = load_raw_data(DEFAULT_DATA_FILE)
    unique_targets = set(df[TARGET_COLUMN].unique())
    assert unique_targets == {"Yes", "No"}

    # Validate class proportions (approx 73.5% No, 26.5% Yes)
    yes_pct = report["target_class_percentages"]["Yes"]
    assert 25.0 <= yes_pct <= 28.0


def test_missing_file_raises_filenotfound():
    """Verify that attempting to load a non-existent file raises FileNotFoundError with helpful context."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_raw_data(Path("data/raw/non_existent_file.csv"))
    assert "https://www.kaggle.com/datasets/blastchar/telco-customer-churn" in str(exc_info.value)


def test_missing_critical_columns_raises_validation_error(tmp_path):
    """Verify that loading a malformed CSV missing critical columns raises DataValidationError."""
    bad_df = pd.DataFrame({"colA": [1, 2], "colB": [3, 4]})
    bad_path = tmp_path / "bad_dataset.csv"
    bad_df.to_csv(bad_path, index=False)

    with pytest.raises(DataValidationError) as exc_info:
        load_raw_data(bad_path)
    assert "Critical columns missing" in str(exc_info.value)
