"""
Tests for Business and Seasonal Analytics Module.
"""

import pytest
import pandas as pd
from src.data_loader import load_dataset
from src.inventory_logic import evaluate_inventory_status
from src.analytics import (
    get_kpi_summary,
    get_sales_trend,
    get_demand_by_category,
    get_inventory_value_by_category,
    get_seasonal_demand_analysis,
    get_top_and_slow_moving_products,
    get_supplier_performance,
)


@pytest.fixture
def sample_data():
    df = load_dataset("data/dataset.csv")
    inv = evaluate_inventory_status(df)
    return df, inv


def test_kpi_summary(sample_data):
    df, inv = sample_data
    kpi = get_kpi_summary(df, inv)
    assert kpi["total_products"] == 500
    assert kpi["total_stock_units"] > 0
    assert kpi["total_inventory_value"] > 0


def test_analytics_aggregations(sample_data):
    df, inv = sample_data

    # Sales trend
    trend = get_sales_trend(df, freq="W")
    assert not trend.empty

    # Category demand
    cat_demand = get_demand_by_category(df)
    assert len(cat_demand) == 4

    # Category inventory valuation (Pandas 3.x aggregation)
    cat_val = get_inventory_value_by_category(inv)
    assert len(cat_val) == 4
    assert "critical_items" in cat_val.columns

    # Seasonal demand
    seasonal = get_seasonal_demand_analysis(df)
    assert len(seasonal) == 16

    # Top & slow moving
    rankings = get_top_and_slow_moving_products(inv, top_n=10)
    assert len(rankings["top_moving"]) == 10
    assert len(rankings["slow_moving"]) == 10

    # Supplier performance (Pandas 3.x aggregation)
    supp_perf = get_supplier_performance(inv)
    assert len(supp_perf) == 30
    assert "critical_skus" in supp_perf.columns

