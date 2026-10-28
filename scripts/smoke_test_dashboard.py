"""
Smoke test script for dashboard backend calculations and interactions.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_dataset, detect_data_issues
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
from src.forecasting import forecast_product_demand, load_forecast_artifacts


def run_smoke_tests():
    print("Running dashboard backend smoke tests...")

    # 1. Dataset loading
    df = load_dataset("data/dataset.csv")
    assert df is not None and len(df) == 5000, f"Expected 5000 rows, got {len(df)}"
    print(" [OK] Dataset loading: 5000 rows, 10 columns")

    # 2. Data quality checks
    issues = detect_data_issues(df)
    assert issues["total_missing"] == 0, "Found missing values in dataset"
    assert issues["exact_duplicates"] == 0, "Found duplicates in dataset"
    assert issues["negative_stock_count"] == 0, "Found negative stock"
    print(" [OK] Data quality & anomaly scanning")

    # 3. Inventory logic calculation across all 500 products
    inv_df = evaluate_inventory_status(df, service_level_z=1.65, planning_horizon_days=14)
    assert len(inv_df) == 500, f"Expected 500 products, got {len(inv_df)}"
    assert not inv_df.isnull().any().any(), "Found NaNs in inventory evaluation table"
    print(" [OK] Inventory ledger evaluation across 500 SKUs (0 NaNs)")

    # 4. KPIs
    kpis = get_kpi_summary(df, inv_df)
    assert kpis["total_products"] == 500
    assert kpis["total_stock_units"] > 0
    assert kpis["total_inventory_value"] > 0
    print(" [OK] Executive KPI generation")

    # 5. Visual aggregations
    trend_w = get_sales_trend(df, freq="W")
    assert not trend_w.empty
    cat_demand = get_demand_by_category(df)
    assert len(cat_demand) == 4
    cat_val = get_inventory_value_by_category(inv_df)
    assert len(cat_val) == 4
    season_stats = get_seasonal_demand_analysis(df)
    assert len(season_stats) == 16  # 4 seasons x 4 categories
    top_slow = get_top_and_slow_moving_products(inv_df, top_n=10)
    assert len(top_slow["top_moving"]) == 10
    assert len(top_slow["slow_moving"]) == 10
    supp_perf = get_supplier_performance(inv_df)
    assert len(supp_perf) == 30
    print(" [OK] Visual analytics & ranking calculations")

    # 6. Forecasting for multiple SKUs and horizons
    artifacts = load_forecast_artifacts()
    assert artifacts is not None, "Model artifacts missing"

    for sku in ["SKU0487", "SKU0001", "SKU0250"]:
        for h in [7, 14, 30]:
            fc = forecast_product_demand(sku, horizon_days=h, df_history=df, artifacts=artifacts)
            assert len(fc["forecast_df"]) == h
            assert (fc["forecast_df"]["forecast_sales"] >= 0).all()
            assert (fc["forecast_df"]["lower_bound"] <= fc["forecast_df"]["upper_bound"]).all()
    print(" [OK] Machine learning forecast generation (7, 14, 30 days) for multiple SKUs")

    # 7. CSV Exports verification
    csv_bytes = inv_df.to_csv(index=False).encode("utf-8")
    assert len(csv_bytes) > 1000
    print(" [OK] CSV export serialization")

    print("\nALL DASHBOARD SMOKE TESTS PASSED!")


if __name__ == "__main__":
    run_smoke_tests()

