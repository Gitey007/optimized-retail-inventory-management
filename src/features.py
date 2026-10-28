"""
Feature engineering module for Optimized Retail Inventory Management System.
Creates temporal, cyclical, categorical, and leakage-free historical lag/rolling demand features.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts calendar and cyclical temporal features from the 'date' column.
    """
    df_out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_out["date"]):
        df_out["date"] = pd.to_datetime(df_out["date"])

    dt = df_out["date"].dt
    df_out["year"] = dt.year
    df_out["month"] = dt.month
    df_out["day"] = dt.day
    df_out["day_of_week"] = dt.day_of_week
    df_out["is_weekend"] = dt.day_of_week.isin([5, 6]).astype(int)
    df_out["quarter"] = dt.quarter
    df_out["day_of_year"] = dt.day_of_year

    # Cyclical encodings for periodicity (months 1-12, days 0-6)
    df_out["sin_month"] = np.sin(2 * np.pi * df_out["month"] / 12.0)
    df_out["cos_month"] = np.cos(2 * np.pi * df_out["month"] / 12.0)
    df_out["sin_dow"] = np.sin(2 * np.pi * df_out["day_of_week"] / 7.0)
    df_out["cos_dow"] = np.cos(2 * np.pi * df_out["day_of_week"] / 7.0)

    return df_out


def add_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes categorical columns ('category', 'season') with standard fixed categories
    to guarantee identical columns between training and inference.
    """
    df_out = df.copy()

    categories = ["Grocery", "Electronics", "Apparel", "HomeDecor"]
    for cat in categories:
        df_out[f"cat_{cat}"] = (df_out["category"] == cat).astype(int)

    seasons = ["Winter", "Summer", "Monsoon", "Festive"]
    for s in seasons:
        df_out[f"season_{s}"] = (df_out["season"] == s).astype(int)

    return df_out


def add_historical_demand_features(
    df: pd.DataFrame,
    split_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Creates historical lag and rolling demand features strictly preventing data leakage.
    Sorts chronologically, shifts by 1 before computing rolling windows,
    and uses category medians as fallbacks for initial records.
    If split_date is provided, fallback category medians are computed strictly
    on training data (date < split_date) to prevent any future information leakage.
    """
    df_out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_out["date"]):
        df_out["date"] = pd.to_datetime(df_out["date"])

    # Ensure chronological sort
    df_out = df_out.sort_values(by=["date", "product_id"]).reset_index(drop=True)

    # Compute lags strictly shifted by 1
    # Group by product_id
    grouped_sales = df_out.groupby("product_id")["daily_sales"]
    df_out["lag_sales_1"] = grouped_sales.shift(1)
    df_out["lag_sales_2"] = grouped_sales.shift(2)

    # Rolling demand based on past observations (shifted)
    df_out["rolling_mean_3"] = (
        grouped_sales.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df_out["rolling_mean_7"] = (
        grouped_sales.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df_out["rolling_std_7"] = (
        grouped_sales.shift(1).rolling(window=7, min_periods=1).std().fillna(0.0)
    )

    # Fallback imputation for products with no prior observations:
    # If split_date is provided, compute medians strictly on pre-split training records
    if split_date is not None:
        train_ref = df_out[df_out["date"] < split_date]
        if not train_ref.empty:
            cat_medians_map = train_ref.groupby("category")["daily_sales"].median().to_dict()
            global_median = float(train_ref["daily_sales"].median())
        else:
            cat_medians_map = df_out.groupby("category")["daily_sales"].median().to_dict()
            global_median = float(df_out["daily_sales"].median())
    else:
        cat_medians_map = df_out.groupby("category")["daily_sales"].median().to_dict()
        global_median = float(df_out["daily_sales"].median())

    cat_fallback = df_out["category"].map(cat_medians_map).fillna(global_median)

    df_out["lag_sales_1"] = df_out["lag_sales_1"].fillna(cat_fallback).fillna(global_median)
    df_out["lag_sales_2"] = df_out["lag_sales_2"].fillna(cat_fallback).fillna(global_median)
    df_out["rolling_mean_3"] = df_out["rolling_mean_3"].fillna(cat_fallback).fillna(global_median)
    df_out["rolling_mean_7"] = df_out["rolling_mean_7"].fillna(cat_fallback).fillna(global_median)
    df_out["rolling_std_7"] = df_out["rolling_std_7"].fillna(0.0)

    return df_out


def build_feature_pipeline(
    df: pd.DataFrame,
    split_date: Optional[pd.Timestamp] = None
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Executes full feature engineering pipeline.

    Returns:
        (transformed_df, feature_column_names, target_column_name)
    """
    df_feats = add_calendar_features(df)
    df_feats = add_categorical_features(df_feats)
    df_feats = add_historical_demand_features(df_feats, split_date=split_date)

    feature_cols = [
        # Calendar & cyclical
        "month",
        "day",
        "day_of_week",
        "is_weekend",
        "quarter",
        "sin_month",
        "cos_month",
        "sin_dow",
        "cos_dow",
        # Categories & Seasons
        "cat_Grocery",
        "cat_Electronics",
        "cat_Apparel",
        "cat_HomeDecor",
        "season_Winter",
        "season_Summer",
        "season_Monsoon",
        "season_Festive",
        # Inventory contextual features
        "reorder_level",
        "lead_days",
        "unit_cost",
        # Historical demand
        "lag_sales_1",
        "lag_sales_2",
        "rolling_mean_3",
        "rolling_mean_7",
        "rolling_std_7",
    ]

    target_col = "daily_sales"
    return df_feats, feature_cols, target_col
