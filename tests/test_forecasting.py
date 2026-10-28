"""
Tests for Machine Learning Demand Forecasting, Feature Engineering, and Inference.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.features import build_feature_pipeline
from src.forecasting import forecast_product_demand, load_forecast_artifacts
from src.train import train_and_evaluate


def test_build_feature_pipeline():
    """Verify feature engineering produces expected columns without NaN values."""
    df = load_dataset("data/dataset.csv")
    df_feats, feature_cols, target_col = build_feature_pipeline(df)

    assert target_col == "daily_sales"
    assert len(feature_cols) >= 20
    assert "lag_sales_1" in feature_cols
    assert "rolling_mean_7" in feature_cols
    assert "sin_month" in feature_cols
    # No NaNs in feature matrix
    assert df_feats[feature_cols].isnull().sum().sum() == 0


def test_model_artifact_loading():
    """Verify that serialized model artifacts exist and contain required keys."""
    artifacts = load_forecast_artifacts("models/demand_forecast_model.joblib")
    assert artifacts is not None
    assert "model" in artifacts
    assert "model_name" in artifacts
    assert "feature_cols" in artifacts
    assert "metrics" in artifacts
    assert artifacts["best_rmse"] > 0


def test_forecast_product_demand():
    """Verify multi-day forecast generation, positive outputs, and prediction bounds."""
    df = load_dataset("data/dataset.csv")
    res = forecast_product_demand("SKU0487", horizon_days=14, df_history=df)

    assert res["product_id"] == "SKU0487"
    assert res["horizon_days"] == 14
    assert len(res["forecast_df"]) == 14
    assert res["total_demand"] >= 0

    forecast_df = res["forecast_df"]
    assert "forecast_sales" in forecast_df.columns
    assert "lower_bound" in forecast_df.columns
    assert "upper_bound" in forecast_df.columns

    # Lower bound should never exceed upper bound
    assert (forecast_df["lower_bound"] <= forecast_df["upper_bound"]).all()
    # Non-negative sales
    assert (forecast_df["forecast_sales"] >= 0).all()


def test_forecast_fallback_sparse_data():
    """Verify fallback mechanism when historical data is minimal or unknown SKU."""
    sparse_df = pd.DataFrame({
        "date": ["2024-01-01"],
        "product_id": ["NEW_SKU_999"],
        "category": ["Electronics"],
        "season": ["Winter"],
        "stock_qty": [50],
        "reorder_level": [20],
        "daily_sales": [12],
        "unit_cost": [100.0],
        "supplier_id": ["SUP01"],
        "lead_days": [5]
    })
    res = forecast_product_demand("NEW_SKU_999", horizon_days=7, df_history=sparse_df)
    assert len(res["forecast_df"]) == 7
    assert res["total_demand"] >= 0
    assert "LOW" in res["data_sufficiency"]


def test_chronological_split_and_validation_separation():
    """Verify that model artifacts contain distinct validation and holdout test metrics."""
    artifacts = load_forecast_artifacts("models/demand_forecast_model.joblib")
    assert "val_metrics" in artifacts
    assert "test_metrics" in artifacts
    assert "val_cutoff_date" in artifacts
    assert "test_cutoff_date" in artifacts
    assert artifacts["val_cutoff_date"] < artifacts["test_cutoff_date"]
    assert "RandomForest" in artifacts["val_metrics"]
    assert "RandomForest" in artifacts["test_metrics"]


def test_artifact_sample_keys_and_schema_compatibility():
    """
    Regression test for KeyError 'validation_samples':
    Verifies that artifacts contain consistent sample count keys and handles
    older or alternative split schemas without crashing.
    """
    artifacts = load_forecast_artifacts("models/demand_forecast_model.joblib")
    assert artifacts is not None
    # Verify sample counts exist and are positive integers
    assert "val_samples" in artifacts
    assert "test_samples" in artifacts
    assert "validation_samples" in artifacts
    assert "holdout_samples" in artifacts
    assert artifacts["val_samples"] == 676
    assert artifacts["validation_samples"] == 676
    assert artifacts["test_samples"] == 697
    assert artifacts["holdout_samples"] == 697


def test_legacy_artifact_schema_compatibility(tmp_path):
    """
    Regression test verifying that load_forecast_artifacts gracefully normalizes
    an older artifact that was saved without the 'validation_samples' key.
    """
    import joblib
    legacy_artifact = {
        "model": "mock_model",
        "model_name": "RandomForest",
        "val_samples": 676,
        "test_samples": 697,
        "best_rmse": 15.553,
    }
    legacy_file = str(tmp_path / "legacy_model.joblib")
    joblib.dump(legacy_artifact, legacy_file)

    loaded = load_forecast_artifacts(legacy_file)
    assert loaded is not None
    # Must have normalized validation_samples without raising KeyError
    assert "validation_samples" in loaded
    assert loaded["validation_samples"] == 676
    assert "holdout_samples" in loaded
    assert loaded["holdout_samples"] == 697




