"""
Tests for Inventory Optimization Logic.
"""

import math
import pytest
import pandas as pd

from src.inventory_logic import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_suggested_reorder_qty,
    calculate_eoq,
    classify_inventory_priority,
    evaluate_inventory_status
)


def test_safety_stock_formula():
    """Verify SS = Z * sigma * sqrt(L)."""
    # Z=1.65, sigma=10, L=4 -> sqrt(4)=2 -> SS = 1.65 * 10 * 2 = 33.0
    ss = calculate_safety_stock(demand_std=10.0, lead_days=4.0, service_level_z=1.65)
    assert pytest.approx(ss, 0.01) == 33.0


def test_safety_stock_fallback():
    """Verify fallback buffer is applied when demand_std is zero or negative."""
    ss = calculate_safety_stock(demand_std=0.0, lead_days=10.0, fallback_mean_demand=20.0)
    # Fallback: 0.20 * 20 * 10 = 40.0
    assert pytest.approx(ss, 0.01) == 40.0


def test_reorder_point_formula():
    """Verify ROP = (daily_demand * lead_days) + safety_stock."""
    rop = calculate_reorder_point(daily_demand=15.0, lead_days=5.0, safety_stock=25.0)
    assert rop == (15.0 * 5.0) + 25.0  # 100.0


def test_suggested_reorder_qty_non_negative():
    """Verify suggested order quantity never becomes negative when current stock is high."""
    order_qty = calculate_suggested_reorder_qty(
        current_stock=1000.0,
        daily_demand=10.0,
        lead_days=5.0,
        safety_stock=20.0,
        planning_horizon_days=14
    )
    # Target = 10 * (5 + 14) + 20 = 210. Current = 1000 -> max(0, 210 - 1000) = 0.0
    assert order_qty == 0.0


def test_suggested_reorder_qty_understocked():
    """Verify suggested order quantity recommends correct replenishment when stock is low."""
    order_qty = calculate_suggested_reorder_qty(
        current_stock=50.0,
        daily_demand=20.0,
        lead_days=5.0,
        safety_stock=30.0,
        planning_horizon_days=10
    )
    # Target = 20 * (5 + 10) + 30 = 330. Order = 330 - 50 = 280.0
    assert order_qty == 280.0


def test_economic_order_quantity():
    """Verify EOQ formula and boundary handling."""
    # D = 3650 units/yr, S = $50, H = $10 * 0.20 = $2/unit/yr
    # EOQ = sqrt((2 * 3650 * 50) / 2) = sqrt(182500) ~ 427.20
    eoq = calculate_eoq(annual_demand=3650.0, unit_cost=10.0, ordering_cost_per_order=50.0, holding_cost_rate=0.20)
    assert pytest.approx(eoq, 0.1) == 427.20

    # Zero annual demand should return 0
    assert calculate_eoq(annual_demand=0.0, unit_cost=10.0) == 0.0


def test_priority_classification():
    """Verify 4-tier priority classification under various stock positions."""
    # Out of stock
    assert classify_inventory_priority(current_stock=0, reorder_point=100, lead_time_demand=50) == "CRITICAL"

    # Below lead time demand (cannot survive lead time)
    assert classify_inventory_priority(current_stock=30, reorder_point=100, lead_time_demand=50) == "CRITICAL"

    # At or below ROP
    assert classify_inventory_priority(current_stock=80, reorder_point=100, lead_time_demand=50) == "HIGH"

    # Moderate buffer (between ROP and 1.5 * ROP)
    assert classify_inventory_priority(current_stock=130, reorder_point=100, lead_time_demand=50) == "MEDIUM"

    # Healthy stock (> 1.5 * ROP)
    assert classify_inventory_priority(current_stock=200, reorder_point=100, lead_time_demand=50) == "LOW"


def test_evaluate_inventory_status_df():
    """Verify batch evaluation across dataframe produces expected output columns."""
    sample_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "product_id": ["SKU100", "SKU100"],
        "category": ["Grocery", "Grocery"],
        "season": ["Winter", "Winter"],
        "stock_qty": [40, 35],
        "reorder_level": [50, 50],
        "daily_sales": [10, 15],
        "unit_cost": [25.0, 25.0],
        "supplier_id": ["SUP01", "SUP01"],
        "lead_days": [5, 5]
    })
    res_df = evaluate_inventory_status(sample_df)
    assert len(res_df) == 1
    assert "calculated_rop" in res_df.columns
    assert "safety_stock" in res_df.columns
    assert "suggested_reorder_qty" in res_df.columns
    assert "priority" in res_df.columns
    assert res_df["current_stock"].iloc[0] == 35  # Latest stock


def test_nan_and_edge_case_inputs():
    """Verify that NaN and extreme inputs do not crash or propagate NaNs."""
    nan_val = float("nan")

    # Safety Stock
    assert calculate_safety_stock(demand_std=nan_val, lead_days=nan_val) == 0.0
    assert calculate_safety_stock(demand_std=10.0, lead_days=nan_val) == 0.0
    assert calculate_safety_stock(demand_std=-5.0, lead_days=5.0) == 0.0

    # Reorder Point
    assert calculate_reorder_point(daily_demand=nan_val, lead_days=nan_val, safety_stock=nan_val) == 0.0
    assert calculate_reorder_point(daily_demand=-10.0, lead_days=-5.0, safety_stock=-2.0) == 0.0

    # EOQ
    assert calculate_eoq(annual_demand=nan_val, unit_cost=nan_val) == 0.0
    assert calculate_eoq(annual_demand=100.0, unit_cost=nan_val) == 0.0
    assert calculate_eoq(annual_demand=-50.0, unit_cost=20.0) == 0.0

    # Suggested Reorder Qty
    assert calculate_suggested_reorder_qty(
        current_stock=nan_val, daily_demand=nan_val, lead_days=nan_val, safety_stock=nan_val
    ) == 0.0

    # Priority
    assert classify_inventory_priority(
        current_stock=nan_val, reorder_point=nan_val, lead_time_demand=nan_val
    ) == "CRITICAL"


