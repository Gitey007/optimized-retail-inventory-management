"""
Data loader and schema validation module for Optimized Retail Inventory Management System.
Handles data ingestion, type conversions, schema validation, anomaly detection, and data cleaning.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

EXPECTED_COLUMNS: List[str] = [
    "date",
    "product_id",
    "category",
    "season",
    "stock_qty",
    "reorder_level",
    "daily_sales",
    "unit_cost",
    "supplier_id",
    "lead_days"
]

NUMERIC_COLUMNS: List[str] = [
    "stock_qty",
    "reorder_level",
    "daily_sales",
    "unit_cost",
    "lead_days"
]

VALID_CATEGORIES: List[str] = ["Grocery", "Electronics", "Apparel", "HomeDecor"]
VALID_SEASONS: List[str] = ["Winter", "Summer", "Monsoon", "Festive"]


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates whether the dataframe adheres to the required schema, columns, and data types.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: List[str] = []

    if df is None or df.empty:
        return False, ["Dataframe is empty or None."]

    # Check for missing expected columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors

    # Check exact column order
    if list(df.columns[:len(EXPECTED_COLUMNS)]) != EXPECTED_COLUMNS:
        errors.append(f"Column order mismatch. Expected: {EXPECTED_COLUMNS}")

    # Check numeric columns are castable
    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                pd.to_numeric(df[col])
            except Exception as e:
                errors.append(f"Column '{col}' cannot be parsed as numeric: {e}")

    # Check date column is parseable
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        try:
            pd.to_datetime(df["date"])
        except Exception as e:
            errors.append(f"Column 'date' cannot be parsed as datetime: {e}")

    is_valid = len(errors) == 0
    return is_valid, errors


def detect_data_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Scans the dataframe for missing values, exact duplicates, invalid negative values,
    and suspicious retail inventory anomalies.
    """
    issues: Dict[str, Any] = {
        "total_records": len(df),
        "total_products": df["product_id"].nunique() if "product_id" in df.columns else 0,
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "exact_duplicates": int(df.duplicated().sum()),
        "negative_stock_count": 0,
        "negative_sales_count": 0,
        "negative_cost_count": 0,
        "invalid_lead_days_count": 0,
        "stockout_zero_stock_count": 0,
        "unusually_high_sales_count": 0,
        "unexpected_categories": [],
        "unexpected_seasons": []
    }

    if "stock_qty" in df.columns:
        issues["negative_stock_count"] = int((df["stock_qty"] < 0).sum())
        issues["stockout_zero_stock_count"] = int((df["stock_qty"] == 0).sum())

    if "daily_sales" in df.columns:
        issues["negative_sales_count"] = int((df["daily_sales"] < 0).sum())
        # Statistical outlier: sales > mean + 3*std
        sales_thresh = df["daily_sales"].mean() + 3 * df["daily_sales"].std()
        issues["unusually_high_sales_count"] = int((df["daily_sales"] > sales_thresh).sum())

    if "unit_cost" in df.columns:
        issues["negative_cost_count"] = int((df["unit_cost"] <= 0).sum())

    if "lead_days" in df.columns:
        issues["invalid_lead_days_count"] = int((df["lead_days"] <= 0).sum())

    if "category" in df.columns:
        unexpected = set(df["category"].dropna().unique()) - set(VALID_CATEGORIES)
        issues["unexpected_categories"] = list(unexpected)

    if "season" in df.columns:
        unexpected = set(df["season"].dropna().unique()) - set(VALID_SEASONS)
        issues["unexpected_seasons"] = list(unexpected)

    return issues


def load_dataset(filepath: str = "data/dataset.csv") -> pd.DataFrame:
    """
    Loads and standardizes the retail inventory dataset.
    Converts 'date' to datetime, validates columns, ensures proper numeric types,
    and sorts chronologically.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If schema validation fails fatally.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at '{filepath}'.")

    df = pd.read_csv(filepath)

    is_valid, errors = validate_schema(df)
    if not is_valid:
        raise ValueError(f"Dataset schema validation failed: {errors}")

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Ensure correct numeric types
    df["stock_qty"] = pd.to_numeric(df["stock_qty"], errors="coerce").fillna(0).astype(int)
    df["reorder_level"] = pd.to_numeric(df["reorder_level"], errors="coerce").fillna(0).astype(int)
    df["daily_sales"] = pd.to_numeric(df["daily_sales"], errors="coerce").fillna(0).astype(int)
    df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce").fillna(0.0).astype(float)
    df["lead_days"] = pd.to_numeric(df["lead_days"], errors="coerce").fillna(1).astype(int)

    # Clean string categories
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["season"] = df["season"].astype(str).str.strip()
    df["supplier_id"] = df["supplier_id"].astype(str).str.strip()

    # Sort chronologically and by product
    df = df.sort_values(by=["date", "product_id"]).reset_index(drop=True)

    return df

