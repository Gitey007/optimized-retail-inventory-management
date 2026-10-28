"""
Demand Forecasting Inference Module for Optimized Retail Inventory Management System.
Generates multi-day future demand projections, prediction intervals, and fallback estimators.
"""

import os
from typing import Dict, Any, Optional, List
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join("models", "demand_forecast_model.joblib")


def load_forecast_artifacts(model_path: str = MODEL_PATH) -> Optional[Dict[str, Any]]:
    """
    Safely loads serialized machine learning model and feature metadata.
    Normalizes schema keys for backward and forward compatibility across 2-way and 3-way split models.
    Returns None if file is missing or invalid.
    """
    if not os.path.exists(model_path):
        return None
    try:
        artifacts = joblib.load(model_path)
        if isinstance(artifacts, dict):
            # Backward and forward compatibility normalization across split schemas
            if "validation_samples" not in artifacts and "val_samples" in artifacts:
                artifacts["validation_samples"] = artifacts["val_samples"]
            elif "val_samples" not in artifacts and "validation_samples" in artifacts:
                artifacts["val_samples"] = artifacts["validation_samples"]

            if "holdout_samples" not in artifacts and "test_samples" in artifacts:
                artifacts["holdout_samples"] = artifacts["test_samples"]
            elif "test_samples" not in artifacts and "holdout_samples" in artifacts:
                artifacts["test_samples"] = artifacts["holdout_samples"]
        return artifacts
    except Exception as e:
        print(f"Warning: Failed to load model artifacts from {model_path}: {e}")
        return None


def get_season_for_date(date_val: pd.Timestamp) -> str:
    """
    Determines season consistently with the empirical dataset mapping:
    - Winter: January, February, December
    - Summer: March, April, May, June
    - Monsoon: July, August, September
    - Festive: October, November
    """
    m = date_val.month
    if m in [1, 2, 12]:
        return "Winter"
    elif m in [3, 4, 5, 6]:
        return "Summer"
    elif m in [7, 8, 9]:
        return "Monsoon"
    else:
        return "Festive"


def forecast_product_demand(
    product_id: str,
    horizon_days: int = 14,
    df_history: Optional[pd.DataFrame] = None,
    artifacts: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates demand forecast for a single product across a configurable future horizon (e.g. 7, 14, 30 days).

    Returns:
        Dict containing forecast DataFrame, cumulative demand, mean demand,
        model information, data sufficiency indicators, and confidence intervals.
    """
    if df_history is None or df_history.empty:
        raise ValueError("Historical dataframe is required for forecasting.")

    if artifacts is None:
        artifacts = load_forecast_artifacts()

    # Filter product history
    p_hist = df_history[df_history["product_id"] == product_id].copy()
    p_hist = p_hist.sort_values("date").reset_index(drop=True)
    obs_count = len(p_hist)

    # Determine product characteristics
    if obs_count > 0:
        category = p_hist["category"].iloc[-1]
        last_date = pd.to_datetime(p_hist["date"].iloc[-1])
        lead_days = float(p_hist["lead_days"].iloc[-1])
        unit_cost = float(p_hist["unit_cost"].iloc[-1])
        reorder_level = float(p_hist["reorder_level"].iloc[-1])
        hist_mean = float(p_hist["daily_sales"].mean())
        hist_std = float(p_hist["daily_sales"].std()) if obs_count > 1 else 0.0
        recent_sales = list(p_hist["daily_sales"].tail(7).values)
    else:
        # Fallback to global statistics if SKU is completely unknown
        category = df_history["category"].mode().iloc[0]
        last_date = pd.to_datetime(df_history["date"].max())
        lead_days = float(df_history["lead_days"].median())
        unit_cost = float(df_history["unit_cost"].median())
        reorder_level = float(df_history["reorder_level"].median())
        hist_mean = float(df_history["daily_sales"].mean())
        hist_std = float(df_history["daily_sales"].std())
        recent_sales = [hist_mean]

    # Data sufficiency assessment
    if obs_count >= 8:
        sufficiency = "HIGH"
    elif obs_count >= 3:
        sufficiency = "MODERATE"
    else:
        sufficiency = "LOW (Sparse historical records)"

    # Future dates
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)]

    # Check if ML model is usable
    can_use_ml = artifacts is not None and "model" in artifacts and "feature_cols" in artifacts

    preds: List[float] = []
    rmse_err = artifacts["best_rmse"] if (can_use_ml and "best_rmse" in artifacts) else max(5.0, hist_std)

    if can_use_ml:
        model = artifacts["model"]
        feature_cols = artifacts["feature_cols"]

        # Recursive multi-step simulation
        rolling_buffer = list(recent_sales)
        for f_date in future_dates:
            dt = f_date
            season = get_season_for_date(dt)

            # Build feature vector
            row_dict = {
                "month": dt.month,
                "day": dt.day,
                "day_of_week": dt.day_of_week,
                "is_weekend": 1 if dt.day_of_week in [5, 6] else 0,
                "quarter": dt.quarter,
                "sin_month": np.sin(2 * np.pi * dt.month / 12.0),
                "cos_month": np.cos(2 * np.pi * dt.month / 12.0),
                "sin_dow": np.sin(2 * np.pi * dt.day_of_week / 7.0),
                "cos_dow": np.cos(2 * np.pi * dt.day_of_week / 7.0),
                "cat_Grocery": 1 if category == "Grocery" else 0,
                "cat_Electronics": 1 if category == "Electronics" else 0,
                "cat_Apparel": 1 if category == "Apparel" else 0,
                "cat_HomeDecor": 1 if category == "HomeDecor" else 0,
                "season_Winter": 1 if season == "Winter" else 0,
                "season_Summer": 1 if season == "Summer" else 0,
                "season_Monsoon": 1 if season == "Monsoon" else 0,
                "season_Festive": 1 if season == "Festive" else 0,
                "reorder_level": reorder_level,
                "lead_days": lead_days,
                "unit_cost": unit_cost,
                "lag_sales_1": rolling_buffer[-1] if len(rolling_buffer) >= 1 else hist_mean,
                "lag_sales_2": rolling_buffer[-2] if len(rolling_buffer) >= 2 else hist_mean,
                "rolling_mean_3": np.mean(rolling_buffer[-3:]) if len(rolling_buffer) >= 1 else hist_mean,
                "rolling_mean_7": np.mean(rolling_buffer[-7:]) if len(rolling_buffer) >= 1 else hist_mean,
                "rolling_std_7": np.std(rolling_buffer[-7:]) if len(rolling_buffer) >= 2 else 0.0,
            }

            feat_df = pd.DataFrame([row_dict])[feature_cols]
            pred_val = float(model.predict(feat_df)[0])
            pred_val = max(0.0, pred_val)
            preds.append(pred_val)
            rolling_buffer.append(pred_val)

        model_name = f"Machine Learning ({artifacts['model_name']})"
    else:
        # Transparent statistical baseline fallback
        model_name = "Historical Moving Average (Baseline Fallback)"
        preds = [max(0.0, hist_mean)] * horizon_days

    # Build forecast dataframe
    forecast_records = []
    for d, p in zip(future_dates, preds):
        p_rounded = round(p, 1)
        lower_bound = max(0.0, round(p - 1.96 * rmse_err, 1))
        upper_bound = round(p + 1.96 * rmse_err, 1)
        forecast_records.append({
            "date": d.strftime("%Y-%m-%d"),
            "forecast_sales": p_rounded,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        })

    forecast_df = pd.DataFrame(forecast_records)
    total_demand = float(forecast_df["forecast_sales"].sum())
    mean_daily = float(forecast_df["forecast_sales"].mean())

    return {
        "product_id": product_id,
        "category": category,
        "horizon_days": horizon_days,
        "forecast_df": forecast_df,
        "total_demand": round(total_demand, 1),
        "mean_daily_demand": round(mean_daily, 2),
        "model_used": model_name,
        "data_sufficiency": sufficiency,
        "historical_observations_count": obs_count,
        "historical_mean_sales": round(hist_mean, 2),
        "prediction_rmse_used": round(rmse_err, 2),
        "interval_type": "Approximate error-based heuristic interval (±1.96 × RMSE assuming normal errors)",
        "validation_metrics": artifacts.get("metrics") if artifacts else None
    }
