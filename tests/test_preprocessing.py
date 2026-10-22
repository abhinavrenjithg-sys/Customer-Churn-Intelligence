"""Unit and integration tests for preprocessing pipelines, feature engineering, and data partitioning."""

import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from src.data_loader import load_raw_data
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    TARGET_MAP,
    create_preprocessor,
    get_feature_names_from_preprocessor,
    split_data,
)


@pytest.fixture(scope="module")
def loaded_data():
    df, _ = load_raw_data()
    return df


def test_feature_target_separation(loaded_data):
    """Ensure customerID is excluded and target is separated."""
    X_train, X_test, y_train, y_test = split_data(loaded_data, test_size=0.20, random_state=42)

    assert ID_COLUMN not in X_train.columns
    assert ID_COLUMN not in X_test.columns
    assert TARGET_COLUMN not in X_train.columns
    assert TARGET_COLUMN not in X_test.columns

    # Verify target encoding
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
    assert y_train.dtype in [np.int32, np.int64, int]


def test_stratified_split_proportions(loaded_data):
    """Verify test split size and stratified churn rate alignment."""
    test_ratio = 0.20
    X_train, X_test, y_train, y_test = split_data(loaded_data, test_size=test_ratio, random_state=42)

    total_samples = len(loaded_data)
    expected_test_samples = int(np.round(total_samples * test_ratio))

    assert len(X_test) == expected_test_samples
    assert len(X_train) == total_samples - expected_test_samples

    # Churn proportion should match across splits (within 0.5%)
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    overall_churn_rate = (loaded_data[TARGET_COLUMN] == "Yes").mean()

    assert abs(train_churn_rate - overall_churn_rate) < 0.005
    assert abs(test_churn_rate - overall_churn_rate) < 0.005


def test_split_reproducibility(loaded_data):
    """Verify that using the same random seed produces identical splits."""
    X_tr1, X_te1, y_tr1, y_te1 = split_data(loaded_data, test_size=0.20, random_state=42)
    X_tr2, X_te2, y_tr2, y_te2 = split_data(loaded_data, test_size=0.20, random_state=42)

    pd.testing.assert_frame_equal(X_tr1, X_tr2)
    pd.testing.assert_frame_equal(X_te1, X_te2)
    pd.testing.assert_series_equal(y_tr1, y_tr2)
    pd.testing.assert_series_equal(y_te1, y_te2)


def test_preprocessor_transformation(loaded_data):
    """Verify ColumnTransformer transforms features without data leakage."""
    X_train, X_test, y_train, y_test = split_data(loaded_data, test_size=0.20, random_state=42)

    preprocessor = create_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)

    # Fit ONLY on train
    X_train_trans = preprocessor.fit_transform(X_train)
    assert isinstance(X_train_trans, np.ndarray)
    assert not np.isnan(X_train_trans).any()

    # Transform test without re-fitting
    X_test_trans = preprocessor.transform(X_test)
    assert isinstance(X_test_trans, np.ndarray)
    assert not np.isnan(X_test_trans).any()

    assert X_train_trans.shape[0] == len(X_train)
    assert X_test_trans.shape[0] == len(X_test)
    assert X_train_trans.shape[1] == X_test_trans.shape[1]

    # Verify feature names extraction
    feature_names = get_feature_names_from_preprocessor(preprocessor)
    assert len(feature_names) == X_train_trans.shape[1]
    assert "tenure" in feature_names[0]
