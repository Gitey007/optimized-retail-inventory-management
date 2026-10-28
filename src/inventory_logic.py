"""
Inventory Optimization Engine for Optimized Retail Inventory Management System.
Implements Reorder Point (ROP), Safety Stock (SS), Suggested Order Quantities,
Economic Order Quantity (EOQ), Stockout Risk Analysis, and Multi-Tier Priority Classification.
"""

from typing import Dict, Any, Optional
import math
import numpy as np
import pandas as pd


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts value to float, converting None or NaN to the default value."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


def calculate_safety_stock(
    demand_std: float,
    lead_days: float,
    service_level_z: float = 1.65,
    fallback_mean_demand: Optional[float] = None
) -> float:
    """
    Computes Safety Stock (SS) to protect against demand and lead-time variability.

    Formula:
        SS = Z * sigma_d * sqrt(L)
    where:
        Z = Service level multiplier (1.65 ~ 95% in-stock probability)
        sigma_d = Standard deviation of daily demand
        L = Lead time in days

    Fallback:
        If demand_std <= 0, missing, or NaN, uses 20% of lead-time demand buffer.
        Returns 0.0 if lead_days <= 0 or NaN.
    """
    l_eff = _safe_float(lead_days, 0.0)
    if l_eff <= 0.0:
        return 0.0

    z_eff = _safe_float(service_level_z, 1.65)
    std_eff = _safe_float(demand_std, 0.0)

    if std_eff > 0.0:
        return float(z_eff * std_eff * math.sqrt(l_eff))

    # Defensive fallback buffer for zero-variance or sparse observations
    fb_mean = _safe_float(fallback_mean_demand, 0.0)
    if fb_mean > 0.0:
        return float(0.20 * fb_mean * l_eff)

    return 0.0


def calculate_reorder_point(
    daily_demand: float,
    lead_days: float,
    safety_stock: float
) -> float:
    """
    Computes Reorder Point (ROP).
    Trigger threshold: replenishment order should be initiated when stock drops to or below ROP.

    Formula:
        ROP = (Expected Daily Demand * Lead Time) + Safety Stock
    """
    d_eff = max(0.0, _safe_float(daily_demand, 0.0))
    l_eff = max(0.0, _safe_float(lead_days, 0.0))
    ss_eff = max(0.0, _safe_float(safety_stock, 0.0))

    lead_time_demand = d_eff * l_eff
    return float(lead_time_demand + ss_eff)


def calculate_suggested_reorder_qty(
    current_stock: float,
    daily_demand: float,
    lead_days: float,
    safety_stock: float,
    planning_horizon_days: int = 14
) -> float:
    """
    Recommends replenishment quantity to cover expected demand over the planning horizon
    plus safety stock, accounting for current on-hand inventory.

    Formula:
        Target Stock = (Daily Demand * (Lead Time + Planning Horizon)) + Safety Stock
        Order Qty = max(0, Target Stock - Current Stock)
    """
    effective_stock = max(0.0, _safe_float(current_stock, 0.0))
    eff_demand = max(0.0, _safe_float(daily_demand, 0.0))
    eff_lead = max(0.0, _safe_float(lead_days, 0.0))
    eff_ss = max(0.0, _safe_float(safety_stock, 0.0))
    eff_horizon = max(0, int(_safe_float(planning_horizon_days, 14)))

    target_stock = (eff_demand * (eff_lead + eff_horizon)) + eff_ss
    order_qty = max(0.0, target_stock - effective_stock)
    return float(order_qty)


def calculate_eoq(
    annual_demand: float,
    unit_cost: float,
    ordering_cost_per_order: float = 50.0,
    holding_cost_rate: float = 0.20
) -> float:
    """
    Calculates Economic Order Quantity (EOQ) using the classic Wilson EOQ model.
    Minimizes the sum of annual ordering costs and inventory holding costs.

    Formula:
        EOQ = sqrt((2 * D * S) / H)
    where:
        D = Annual Demand in units
        S = Fixed ordering cost per order (configurable parameter, default $50.00)
        H = Annual holding cost per unit = unit_cost * holding_cost_rate (default 20%)

    Assumptions & Limitations:
        - Constant demand rate throughout the year.
        - Fixed order cost independent of batch size.
        - Instantaneous delivery / fixed lead time.
    """
    d_eff = _safe_float(annual_demand, 0.0)
    c_eff = _safe_float(unit_cost, 0.0)
    s_eff = _safe_float(ordering_cost_per_order, 50.0)
    h_rate = _safe_float(holding_cost_rate, 0.20)

    if d_eff <= 0.0 or c_eff <= 0.0 or s_eff <= 0.0 or h_rate <= 0.0:
        return 0.0

    holding_cost_per_unit = c_eff * h_rate
    eoq = math.sqrt((2.0 * d_eff * s_eff) / holding_cost_per_unit)
    return float(eoq)


def classify_inventory_priority(
    current_stock: float,
    reorder_point: float,
    lead_time_demand: float
) -> str:
    """
    Classifies inventory replenishment urgency into four discrete tiers:
        - CRITICAL: Out of stock (stock == 0) OR stock is less than lead-time demand.
        - HIGH: Stock is at or below the reorder point (ROP).
        - MEDIUM: Stock is between ROP and 1.5 * ROP.
        - LOW: Stock is healthy (> 1.5 * ROP).
    """
    stock = max(0.0, _safe_float(current_stock, 0.0))
    rop = max(0.0, _safe_float(reorder_point, 0.0))
    ltd = max(0.0, _safe_float(lead_time_demand, 0.0))

    if stock == 0.0:
        return "CRITICAL"

    if ltd > 0.0 and stock < ltd:
        return "CRITICAL"

    if stock <= rop:
        return "HIGH"

    if stock <= 1.5 * rop:
        return "MEDIUM"

    return "LOW"


def evaluate_inventory_status(
    df: pd.DataFrame,
    service_level_z: float = 1.65,
    planning_horizon_days: int = 14,
    ordering_cost: float = 50.0,
    holding_cost_rate: float = 0.20
) -> pd.DataFrame:
    """
    Evaluates complete inventory recommendations across all products in the dataset.
    Takes latest observed state per product_id, computes statistical demand,
    safety stock, ROP, suggested order quantity, EOQ, stockout risks, and priorities.
    """
    df_clean = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_clean["date"]):
        df_clean["date"] = pd.to_datetime(df_clean["date"])

    # Aggregate historical demand metrics per product
    demand_stats = (
        df_clean.groupby("product_id")["daily_sales"]
        .agg(mean_daily_sales="mean", std_daily_sales="std")
        .reset_index()
    )
    demand_stats["std_daily_sales"] = demand_stats["std_daily_sales"].fillna(0.0)

    # Get latest status for each product
    latest_df = (
        df_clean.sort_values(by=["date"])
        .groupby("product_id")
        .last()
        .reset_index()
    )

    merged = pd.merge(latest_df, demand_stats, on="product_id", how="left")

    results = []
    for _, row in merged.iterrows():
        stock = max(0.0, _safe_float(row.get("stock_qty", 0.0)))
        lead = max(0.0, _safe_float(row.get("lead_days", 1.0)))
        cost = max(0.0, _safe_float(row.get("unit_cost", 0.0)))
        existing_reorder_level = max(0.0, _safe_float(row.get("reorder_level", 0.0)))
        d_mean = max(0.0, _safe_float(row.get("mean_daily_sales", 0.0)))
        d_std = max(0.0, _safe_float(row.get("std_daily_sales", 0.0)))

        # Safety Stock & Reorder Point
        ss = calculate_safety_stock(
            demand_std=d_std,
            lead_days=lead,
            service_level_z=service_level_z,
            fallback_mean_demand=d_mean
        )
        rop = calculate_reorder_point(
            daily_demand=d_mean,
            lead_days=lead,
            safety_stock=ss
        )

        lead_time_demand = d_mean * lead
        suggested_qty = calculate_suggested_reorder_qty(
            current_stock=stock,
            daily_demand=d_mean,
            lead_days=lead,
            safety_stock=ss,
            planning_horizon_days=planning_horizon_days
        )

        annual_demand = d_mean * 365.0
        eoq = calculate_eoq(
            annual_demand=annual_demand,
            unit_cost=cost,
            ordering_cost_per_order=ordering_cost,
            holding_cost_rate=holding_cost_rate
        )

        priority = classify_inventory_priority(
            current_stock=stock,
            reorder_point=rop,
            lead_time_demand=lead_time_demand
        )

        # Days of inventory remaining
        days_of_supply = stock / max(0.1, d_mean) if d_mean > 0 else (999.0 if stock > 0 else 0.0)
        stockout_risk = days_of_supply < lead if d_mean > 0 else False

        inventory_value = stock * cost

        # Reason string
        if stock == 0:
            reason = "Stockout: Zero units in inventory."
        elif lead_time_demand > 0 and stock < lead_time_demand:
            reason = f"Critical deficit: Stock ({stock:.0f}) < Lead Time Demand ({lead_time_demand:.0f})."
        elif stock <= rop:
            reason = f"Below ROP: Stock ({stock:.0f}) <= Reorder Point ({rop:.0f})."
        elif stock <= 1.5 * rop:
            reason = "Moderate buffer: Stock nearing reorder boundary."
        else:
            reason = "Adequate inventory levels."

        results.append({
            "product_id": str(row["product_id"]),
            "category": str(row["category"]),
            "supplier_id": str(row["supplier_id"]),
            "current_stock": int(stock),
            "unit_cost": round(cost, 2),
            "lead_days": int(lead),
            "mean_daily_demand": round(d_mean, 2),
            "demand_std": round(d_std, 2),
            "existing_reorder_level": int(existing_reorder_level),
            "safety_stock": round(ss, 1),
            "calculated_rop": round(rop, 1),
            "lead_time_demand": round(lead_time_demand, 1),
            "suggested_reorder_qty": int(math.ceil(suggested_qty)),
            "economic_order_qty": int(round(eoq)),
            "days_of_supply": round(days_of_supply, 1),
            "stockout_risk": bool(stockout_risk),
            "inventory_value": round(inventory_value, 2),
            "priority": priority,
            "recommendation_reason": reason,
            "latest_record_date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
        })

    return pd.DataFrame(results)
