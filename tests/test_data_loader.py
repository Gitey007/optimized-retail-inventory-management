"""
Tests for Data Loader and Schema Validation.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    load_dataset,
    validate_schema,
    detect_data_issues,
    EXPECTED_COLUMNS
)


def test_load_dataset_success():
    """Verify that data/dataset.csv loads with exact schema and expected dimensions."""
    df = load_dataset("data/dataset.csv")
    assert df is not None
    assert len(df) == 5000
    assert len(df.columns) == 10
    assert list(df.columns) == EXPECTED_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["product_id"].nunique() == 500


def test_load_dataset_missing_file():
    """Verify that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset("data/non_existent_file.csv")


def test_validate_schema_missing_column():
    """Verify that missing required columns fail schema validation."""
    df_broken = pd.DataFrame({
        "date": ["2024-01-01"],
        "product_id": ["SKU0001"]
        # Missing other 8 columns
    })
    is_valid, errors = validate_schema(df_broken)
    assert not is_valid
    assert any("Missing required columns" in e for e in errors)


def test_validate_schema_empty_df():
    """Verify that an empty dataframe fails validation gracefully."""
    is_valid, errors = validate_schema(pd.DataFrame())
    assert not is_valid


def test_detect_data_issues():
    """Verify that data issues, nulls, negative values, and anomalies are correctly identified."""
    sample_data = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "product_id": ["SKU001", "SKU002", "SKU001"],
        "category": ["Grocery", "Electronics", "UnknownCategory"],
        "season": ["Winter", "Summer", "UnknownSeason"],
        "stock_qty": [100, 0, -5],
        "reorder_level": [50, 40, 30],
        "daily_sales": [20, -10, 500],
        "unit_cost": [10.5, 25.0, -2.0],
        "supplier_id": ["SUP01", "SUP02", "SUP01"],
        "lead_days": [7, 0, 14]
    })
    issues = detect_data_issues(sample_data)

    assert issues["negative_stock_count"] == 1
    assert issues["stockout_zero_stock_count"] == 1
    assert issues["negative_sales_count"] == 1
    assert issues["negative_cost_count"] == 1
    assert issues["invalid_lead_days_count"] == 1
    assert "UnknownCategory" in issues["unexpected_categories"]
    assert "UnknownSeason" in issues["unexpected_seasons"]

