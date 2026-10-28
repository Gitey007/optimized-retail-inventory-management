"""
Business and Seasonal Analytics Module for Optimized Retail Inventory Management System.
Generates metrics, aggregations, trend tables, and statistical summaries for inventory dashboards.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def get_kpi_summary(df: pd.DataFrame, inventory_status_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes top-level executive KPI metrics across the entire enterprise.
    """
    total_products = inventory_status_df["product_id"].nunique()
    total_stock_units = int(inventory_status_df["current_stock"].sum())
    total_inventory_value = float(inventory_status_df["inventory_value"].sum())

    reorder_needed_count = int(
        (inventory_status_df["priority"].isin(["CRITICAL", "HIGH"])).sum()
    )
    critical_stock_count = int(
        (inventory_status_df["priority"] == "CRITICAL").sum()
    )
    stockout_risk_count = int(inventory_status_df["stockout_risk"].sum())

    total_daily_demand = float(inventory_status_df["mean_daily_demand"].sum())

    return {
        "total_products": total_products,
        "total_stock_units": total_stock_units,
        "total_inventory_value": total_inventory_value,
        "reorder_needed_count": reorder_needed_count,
        "critical_stock_count": critical_stock_count,
        "stockout_risk_count": stockout_risk_count,
        "total_daily_demand": round(total_daily_demand, 1),
    }


def get_sales_trend(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Computes aggregated sales trend over time ('D' for daily, 'W' for weekly, 'M' for monthly).
    """
    df_temp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_temp["date"]):
        df_temp["date"] = pd.to_datetime(df_temp["date"])

    trend = (
        df_temp.set_index("date")
        .groupby([pd.Grouper(freq=freq), "category"])["daily_sales"]
        .sum()
        .reset_index()
    )
    return trend


def get_demand_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes total demand, average daily sales, and record counts broken down by category.
    """
    agg = (
        df.groupby("category")["daily_sales"]
        .agg(
            total_sales="sum",
            mean_daily_sales="mean",
            record_count="count",
            std_daily_sales="std"
        )
        .reset_index()
    )
    agg["mean_daily_sales"] = agg["mean_daily_sales"].round(2)
    agg["std_daily_sales"] = agg["std_daily_sales"].round(2)
    return agg.sort_values(by="total_sales", ascending=False)


def get_inventory_value_by_category(inventory_status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates on-hand stock units and capital inventory value by category.
    Compatible with Pandas 2.x and 3.x named tuple aggregation.
    """
    agg = (
        inventory_status_df.groupby("category")
        .agg(
            total_units=("current_stock", "sum"),
            total_value=("inventory_value", "sum"),
            product_count=("product_id", "count"),
        )
        .reset_index()
    )
    crit_map = (
        inventory_status_df[inventory_status_df["priority"] == "CRITICAL"]
        .groupby("category")["product_id"]
        .count()
        .to_dict()
    )
    agg["critical_items"] = agg["category"].map(crit_map).fillna(0).astype(int)
    agg["total_value"] = agg["total_value"].round(2)
    return agg.sort_values(by="total_value", ascending=False)


def get_seasonal_demand_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates seasonal demand statistics: mean, median, min, max, total sales per season and category.
    """
    agg = (
        df.groupby(["season", "category"])["daily_sales"]
        .agg(
            total_sales="sum",
            avg_daily_sales="mean",
            median_sales="median",
            std_sales="std",
            observations="count"
        )
        .reset_index()
    )
    agg["avg_daily_sales"] = agg["avg_daily_sales"].round(2)
    agg["std_sales"] = agg["std_sales"].round(2)
    return agg


def get_top_and_slow_moving_products(
    inventory_status_df: pd.DataFrame,
    top_n: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Identifies top-moving products (highest daily demand) and slow-moving products
    (lowest daily demand with excessive days of supply).
    """
    top_moving = (
        inventory_status_df.sort_values(by="mean_daily_demand", ascending=False)
        .head(top_n)[["product_id", "category", "current_stock", "mean_daily_demand", "unit_cost", "days_of_supply"]]
    )

    slow_moving = (
        inventory_status_df.sort_values(by=["days_of_supply", "mean_daily_demand"], ascending=[False, True])
        .head(top_n)[["product_id", "category", "current_stock", "mean_daily_demand", "days_of_supply", "inventory_value"]]
    )

    return {
        "top_moving": top_moving,
        "slow_moving": slow_moving
    }


def get_supplier_performance(inventory_status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes supplier metrics: product count, average lead time, and total stockout risk items.
    Compatible with Pandas 2.x and 3.x named tuple aggregation.
    """
    supp = (
        inventory_status_df.groupby("supplier_id")
        .agg(
            skus_managed=("product_id", "count"),
            avg_lead_days=("lead_days", "mean"),
            total_stock=("current_stock", "sum"),
            stockout_risk_items=("stockout_risk", "sum")
        )
        .reset_index()
    )
    crit_by_sup = (
        inventory_status_df[inventory_status_df["priority"] == "CRITICAL"]
        .groupby("supplier_id")["product_id"]
        .count()
        .to_dict()
    )
    supp["critical_skus"] = supp["supplier_id"].map(crit_by_sup).fillna(0).astype(int)
    supp["avg_lead_days"] = supp["avg_lead_days"].round(1)
    return supp.sort_values(by="critical_skus", ascending=False)
