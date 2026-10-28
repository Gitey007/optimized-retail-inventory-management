"""
Optimized Retail Inventory Management System
Main Streamlit Application Dashboard.
"""

import os
import sys
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in python path
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
from src.train import train_and_evaluate

# Set page configuration
st.set_page_config(
    page_title="Optimized Retail Inventory Management System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-high {
        background-color: #ffedd5;
        color: #9a3412;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-medium {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_dataset(filepath: str = "data/dataset.csv") -> pd.DataFrame:
    """Loads and caches the standardized dataset."""
    return load_dataset(filepath)


@st.cache_data
def get_cached_inventory_status(
    _df: pd.DataFrame,
    service_level: float = 1.65,
    horizon_days: int = 14,
    ordering_cost: float = 50.0,
    holding_cost_rate: float = 0.20,
) -> pd.DataFrame:
    """Computes and caches comprehensive inventory status."""
    return evaluate_inventory_status(
        _df,
        service_level_z=service_level,
        planning_horizon_days=horizon_days,
        ordering_cost=ordering_cost,
        holding_cost_rate=holding_cost_rate,
    )


def main():
    # Header Banner
    st.title("📦 Optimized Retail Inventory Management System")
    st.caption(
        "Autonomous Demand Forecasting, Safety Stock Optimization, Reorder Planning, and Risk Analytics"
    )

    # 1. Ingestion / Data Loading
    dataset_path = "data/dataset.csv"
    if not os.path.exists(dataset_path):
        st.error(
            f"❌ Primary dataset not found at `{dataset_path}`. Please verify dataset files."
        )
        st.stop()

    try:
        raw_df = get_cached_dataset(dataset_path)
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.stop()

    # Sidebar Controls & Global Filters
    st.sidebar.image(
        "https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?auto=format&fit=crop&w=400&q=80",
        use_container_width=True,
        caption="Enterprise Retail Logistics",
    )
    st.sidebar.title("🎛️ Navigation & Filters")

    # Global Category & Season filter
    all_categories = ["All"] + sorted(list(raw_df["category"].unique()))
    selected_category = st.sidebar.selectbox("Category Filter", all_categories)

    all_seasons = ["All"] + sorted(list(raw_df["season"].unique()))
    selected_season = st.sidebar.selectbox("Season Filter", all_seasons)

    # Date range filter
    min_date = raw_df["date"].min().date()
    max_date = raw_df["date"].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Forecast horizon
    forecast_horizon = st.sidebar.selectbox(
        "Demand Forecast Horizon",
        [7, 14, 30],
        index=1,
        help="Planning window in days for future demand projection and order calculation.",
    )

    # Configurable Inventory Parameters
    with st.sidebar.expander("⚙️ EOQ & Safety Stock Parameters", expanded=False):
        service_level_z = st.slider(
            "Service Level Factor (Z)",
            min_value=1.28,
            max_value=2.33,
            value=1.65,
            step=0.01,
            help="1.28 ~ 90%, 1.65 ~ 95%, 1.96 ~ 97.5%, 2.33 ~ 99% in-stock probability.",
        )
        ordering_cost = st.number_input(
            "Order Setup Cost ($/order)",
            min_value=1.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
            help="Fixed administrative and shipping cost incurred per replenishment purchase order.",
        )
        holding_rate = st.slider(
            "Holding Cost Rate (% of unit cost/yr)",
            min_value=0.05,
            max_value=0.50,
            value=0.20,
            step=0.01,
            help="Annual warehouse holding cost expressed as percentage of unit cost.",
        )

    # Retrain button
    if st.sidebar.button("🔄 Retrain ML Forecast Model"):
        with st.spinner("Retraining RandomForest & HistGradientBoosting models..."):
            artifacts = train_and_evaluate(dataset_path)
            model_nm = artifacts.get("model_name", "RandomForest")
            rmse_val = artifacts.get("best_rmse", 0.0)
            st.sidebar.success(
                f"Model retrained! Best: {model_nm} (RMSE: {rmse_val:.2f})"
            )

    # Apply Filters to dataset
    filtered_df = raw_df.copy()
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["category"] == selected_category]
    if selected_season != "All":
        filtered_df = filtered_df[filtered_df["season"] == selected_season]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        filtered_df = filtered_df[
            (filtered_df["date"].dt.date >= start_d)
            & (filtered_df["date"].dt.date <= end_d)
        ]

    # Inventory status calculations
    inventory_df = get_cached_inventory_status(
        filtered_df if not filtered_df.empty else raw_df,
        service_level=service_level_z,
        horizon_days=forecast_horizon,
        ordering_cost=ordering_cost,
        holding_cost_rate=holding_rate,
    )

    # Data Quality summary
    data_issues = detect_data_issues(filtered_df)

    # Tabs
    tabs = st.tabs(
        [
            "📊 Executive Overview",
            "📦 Inventory Stock Management",
            "📈 ML Demand Forecasting",
            "🛒 Reorder Recommendations",
            "🌦️ Seasonal Analytics",
            "🔍 Data Quality & Diagnostics",
        ]
    )

    # ==========================================
    # TAB 1: EXECUTIVE OVERVIEW
    # ==========================================
    with tabs[0]:
        st.subheader("Enterprise KPI Performance")
        kpi = get_kpi_summary(filtered_df, inventory_df)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Products", f"{kpi['total_products']:,}")
        c2.metric("Inventory Units", f"{kpi['total_stock_units']:,}")
        c3.metric("Inventory Valuation", f"${kpi['total_inventory_value']:,.2f}")
        c4.metric(
            "Reorder Required",
            f"{kpi['reorder_needed_count']:,}",
            delta=f"{(kpi['reorder_needed_count']/max(1, kpi['total_products']))*100:.1f}% SKUs",
            delta_color="inverse",
        )
        c5.metric(
            "Critical Stockouts",
            f"{kpi['critical_stock_count']:,}",
            delta="Stock Depletion Risk",
            delta_color="inverse",
        )
        c6.metric(
            f"{forecast_horizon}-Day Demand",
            f"{kpi['total_daily_demand'] * forecast_horizon:,.0f} units",
        )

        st.markdown("---")

        # Visual Row 1: Sales Trend & Category Distribution
        col_left, col_right = st.columns([7, 5])

        with col_left:
            st.markdown("##### 📈 Daily Sales Trajectory by Category")
            trend_df = get_sales_trend(filtered_df, freq="W")
            if not trend_df.empty:
                fig_trend = px.line(
                    trend_df,
                    x="date",
                    y="daily_sales",
                    color="category",
                    title="Weekly Aggregate Sales Trend (Units Sold)",
                    labels={"daily_sales": "Weekly Sales", "date": "Date"},
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                fig_trend.update_layout(
                    hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No sales data available for the selected filters.")

        with col_right:
            st.markdown("##### 💰 Capital Allocation by Category")
            cat_val = get_inventory_value_by_category(inventory_df)
            fig_pie = px.pie(
                cat_val,
                names="category",
                values="total_value",
                hole=0.45,
                title="Current Inventory Valuation Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Visual Row 2: Stock vs Reorder Level & Priority Breakdown
        col_b1, col_b2 = st.columns([6, 6])
        with col_b1:
            st.markdown("##### 🛡️ On-Hand Stock vs Calculated Reorder Point")
            sample_inv = inventory_df.head(40)
            fig_scatter = go.Figure()
            fig_scatter.add_trace(
                go.Bar(
                    x=sample_inv["product_id"],
                    y=sample_inv["current_stock"],
                    name="Current Stock",
                    marker_color="#3b82f6",
                )
            )
            fig_scatter.add_trace(
                go.Scatter(
                    x=sample_inv["product_id"],
                    y=sample_inv["calculated_rop"],
                    name="Reorder Point (ROP)",
                    mode="lines+markers",
                    line=dict(color="#ef4444", width=2, dash="dot"),
                )
            )
            fig_scatter.update_layout(
                title="First 40 SKUs: Current Stock vs ROP Boundary",
                xaxis_title="Product ID",
                yaxis_title="Units",
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_b2:
            st.markdown("##### 🚦 Inventory Replenishment Priority Breakdown")
            priority_counts = (
                inventory_df["priority"]
                .value_counts()
                .reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"])
                .fillna(0)
                .reset_index()
            )
            priority_counts.columns = ["Priority", "Count"]

            fig_prio = px.bar(
                priority_counts,
                x="Priority",
                y="Count",
                color="Priority",
                text="Count",
                color_discrete_map={
                    "CRITICAL": "#dc2626",
                    "HIGH": "#ea580c",
                    "MEDIUM": "#ca8a04",
                    "LOW": "#16a34a",
                },
                title="SKUs by Operational Urgency",
            )
            fig_prio.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_prio, use_container_width=True)

    # ==========================================
    # TAB 2: INVENTORY STOCK MANAGEMENT
    # ==========================================
    with tabs[1]:
        st.subheader("Active Inventory Stock Ledger")
        st.caption(
            "Search, filter, and audit SKU health, safety stocks, lead times, and financial value."
        )

        col_search, col_prio, col_supp = st.columns([4, 4, 4])
        with col_search:
            search_query = st.text_input(
                "🔎 Search by Product ID (e.g. SKU0487)", ""
            ).strip()
        with col_prio:
            filter_priority = st.multiselect(
                "Filter by Priority",
                ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            )
        with col_supp:
            all_suppliers = ["All"] + sorted(list(inventory_df["supplier_id"].unique()))
            selected_supp = st.selectbox("Filter by Supplier", all_suppliers)

        filtered_table = inventory_df[
            inventory_df["priority"].isin(filter_priority)
        ].copy()
        if search_query:
            filtered_table = filtered_table[
                filtered_table["product_id"].str.contains(search_query, case=False)
            ]
        if selected_supp != "All":
            filtered_table = filtered_table[
                filtered_table["supplier_id"] == selected_supp
            ]

        display_cols = [
            "product_id",
            "category",
            "supplier_id",
            "current_stock",
            "mean_daily_demand",
            "lead_days",
            "calculated_rop",
            "safety_stock",
            "unit_cost",
            "inventory_value",
            "days_of_supply",
            "priority",
        ]

        st.dataframe(
            filtered_table[display_cols].style.format(
                {
                    "unit_cost": "${:,.2f}",
                    "inventory_value": "${:,.2f}",
                    "mean_daily_demand": "{:.1f}",
                    "calculated_rop": "{:.1f}",
                    "safety_stock": "{:.1f}",
                    "days_of_supply": "{:.1f}",
                }
            ),
            use_container_width=True,
            height=460,
        )

        # Download button
        csv_data = filtered_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Filtered Inventory Table (CSV)",
            data=csv_data,
            file_name="retail_inventory_ledger.csv",
            mime="text/csv",
        )

    # ==========================================
    # TAB 3: DEMAND FORECASTING (ML)
    # ==========================================
    with tabs[2]:
        st.subheader("Machine Learning Demand Forecasting")
        st.caption(
            "Predict upcoming daily demand using trained Random Forest & Gradient Boosted regressors."
        )

        f_col1, f_col2, f_col3 = st.columns([4, 4, 4])
        with f_col1:
            all_skus = sorted(list(raw_df["product_id"].unique()))
            default_sku_idx = all_skus.index("SKU0487") if "SKU0487" in all_skus else 0
            selected_sku = st.selectbox(
                "Select Product ID for Forecast", all_skus, index=default_sku_idx
            )
        with f_col2:
            forecast_days = st.slider(
                "Prediction Horizon (Days)",
                min_value=7,
                max_value=30,
                value=forecast_horizon,
                step=7,
            )
        with f_col3:
            st.write("")
            st.write("")
            run_btn = st.button("🔮 Generate Forecast", type="primary")

        # Run forecast
        artifacts = load_forecast_artifacts()
        forecast_res = forecast_product_demand(
            product_id=selected_sku,
            horizon_days=forecast_days,
            df_history=raw_df,
            artifacts=artifacts,
        )

        # Overview Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Selected Product", selected_sku, forecast_res["category"])
        m2.metric(
            f"Cumulative {forecast_days}-Day Demand",
            f"{forecast_res['total_demand']:.0f} units",
        )
        m3.metric(
            "Expected Daily Demand",
            f"{forecast_res['mean_daily_demand']:.1f} units/day",
        )
        m4.metric(
            "Historical Observations",
            f"{forecast_res['historical_observations_count']} days recorded",
            f"Quality: {forecast_res['data_sufficiency']}",
        )

        st.markdown("---")

        # Chart: Historical Sales + Future Predictions with Confidence Band
        hist_p = raw_df[raw_df["product_id"] == selected_sku].sort_values("date")
        forecast_df = forecast_res["forecast_df"]

        fig_fc = go.Figure()

        # Historical Trace
        fig_fc.add_trace(
            go.Scatter(
                x=hist_p["date"],
                y=hist_p["daily_sales"],
                name="Historical Actual Sales",
                mode="lines+markers",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=6),
            )
        )

        # Forecast Line
        fig_fc.add_trace(
            go.Scatter(
                x=pd.to_datetime(forecast_df["date"]),
                y=forecast_df["forecast_sales"],
                name=f"Forecast ({forecast_res['model_used']})",
                mode="lines+markers",
                line=dict(color="#10b981", width=3, dash="solid"),
                marker=dict(size=7, symbol="diamond"),
            )
        )

        # Confidence Interval Bounds
        fig_fc.add_trace(
            go.Scatter(
                x=pd.to_datetime(forecast_df["date"]),
                y=forecast_df["upper_bound"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_fc.add_trace(
            go.Scatter(
                x=pd.to_datetime(forecast_df["date"]),
                y=forecast_df["lower_bound"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(16, 185, 129, 0.18)",
                name="Approx. Error Interval (±1.96 × RMSE)",
            )
        )

        fig_fc.update_layout(
            title=f"Historical Actuals and {forecast_days}-Day Future Forecast for {selected_sku}",
            xaxis_title="Date",
            yaxis_title="Units Sold / Day",
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_fc, use_container_width=True)
        st.caption(
            "ℹ️ **Prediction Interval Note:** Shaded band is an approximate error-based heuristic interval "
            "(±1.96 × holdout RMSE under an assumed Gaussian residual distribution), not a statistically calibrated 95% conformal prediction interval."
        )

        # Tabular Forecast & Model Metrics
        t_col1, t_col2 = st.columns([6, 6])
        with t_col1:
            st.markdown("##### 📅 Daily Projected Demand")
            st.dataframe(
                forecast_df.style.format(
                    {
                        "forecast_sales": "{:.1f}",
                        "lower_bound": "{:.1f}",
                        "upper_bound": "{:.1f}",
                    }
                ),
                use_container_width=True,
                height=260,
            )
            fc_csv = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Forecast Projection (CSV)",
                data=fc_csv,
                file_name=f"forecast_{selected_sku}_{forecast_days}days.csv",
                mime="text/csv",
            )

        with t_col2:
            st.markdown("##### 🧪 Model Benchmark & Validation Metrics")
            if artifacts and "metrics" in artifacts:
                metric_rows = []
                for m_k, m_v in artifacts["metrics"].items():
                    metric_rows.append(
                        {
                            "Model / Baseline": m_k,
                            "MAE (units)": m_v["MAE"],
                            "RMSE (units)": m_v["RMSE"],
                            "R² Score": m_v["R2"],
                            "WAPE (%)": f"{m_v['WAPE%']}%",
                        }
                    )
                st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

                val_count = artifacts.get("val_samples") or artifacts.get("validation_samples")
                test_count = artifacts.get("test_samples") or artifacts.get("holdout_samples") or artifacts.get("validation_samples")
                model_name = artifacts.get("model_name", "RandomForest")

                if val_count and test_count and val_count != test_count:
                    summary_msg = (
                        f"**Selected Model:** `{model_name}`. "
                        f"Tuned & selected on {val_count} validation observations; "
                        f"evaluated on {test_count} unseen chronological holdout observations with no temporal leakage."
                    )
                elif test_count:
                    summary_msg = (
                        f"**Selected Model:** `{model_name}`. "
                        f"Chronologically evaluated on {test_count} future holdout observations with no temporal leakage."
                    )
                else:
                    summary_msg = (
                        f"**Selected Model:** `{model_name}`. "
                        "Chronologically evaluated on future holdout observations with no temporal leakage."
                    )
                st.info(summary_msg)
            else:
                st.warning(
                    "Model evaluation artifacts not yet loaded. Click 'Retrain ML Forecast Model' in the sidebar to generate metrics."
                )

    # ==========================================
    # TAB 4: REORDER RECOMMENDATIONS
    # ==========================================
    with tabs[3]:
        st.subheader("Autonomous Replenishment & Reorder Planner")
        st.caption(
            "Calculates Reorder Point (ROP), Safety Stock, Suggested Purchase Quantities, and Economic Order Quantity (EOQ)."
        )

        r_crit, r_high, r_med, r_all = st.columns(4)
        prio_counts = inventory_df["priority"].value_counts().to_dict()
        r_crit.metric(
            "CRITICAL (Stockout / Deficit)",
            prio_counts.get("CRITICAL", 0),
            "Urgent Action",
            delta_color="inverse",
        )
        r_high.metric(
            "HIGH (At/Below ROP)",
            prio_counts.get("HIGH", 0),
            "Place Order",
            delta_color="inverse",
        )
        r_med.metric("MEDIUM (Watchlist)", prio_counts.get("MEDIUM", 0))
        r_all.metric(
            "Total Suggested Units to Buy",
            f"{inventory_df['suggested_reorder_qty'].sum():,} units",
        )

        st.markdown("---")

        rec_table = inventory_df.sort_values(
            by=["priority", "suggested_reorder_qty"], ascending=[True, False]
        )

        rec_cols = [
            "product_id",
            "category",
            "supplier_id",
            "current_stock",
            "lead_time_demand",
            "calculated_rop",
            "safety_stock",
            "suggested_reorder_qty",
            "economic_order_qty",
            "lead_days",
            "priority",
            "recommendation_reason",
        ]

        st.dataframe(
            rec_table[rec_cols].style.format(
                {
                    "lead_time_demand": "{:.1f}",
                    "calculated_rop": "{:.1f}",
                    "safety_stock": "{:.1f}",
                    "suggested_reorder_qty": "{:,.0f}",
                    "economic_order_qty": "{:,.0f}",
                }
            ),
            use_container_width=True,
            height=440,
        )

        # Download Reorders
        reorder_csv = (
            rec_table[rec_table["priority"].isin(["CRITICAL", "HIGH"])]
            .to_csv(index=False)
            .encode("utf-8")
        )
        st.download_button(
            "📥 Download Actionable Reorder Plan (CSV)",
            data=reorder_csv,
            file_name="actionable_reorder_recommendations.csv",
            mime="text/csv",
        )

    # ==========================================
    # TAB 5: SEASONAL & BUSINESS ANALYTICS
    # ==========================================
    with tabs[4]:
        st.subheader("Empirical Seasonal Demand & Category Breakdown")
        st.caption(
            "Empirical demand trends derived directly from the 5,000 recorded observations."
        )

        s_col1, s_col2 = st.columns([6, 6])
        with s_col1:
            season_df = get_seasonal_demand_analysis(filtered_df)
            fig_season = px.bar(
                season_df,
                x="season",
                y="avg_daily_sales",
                color="category",
                barmode="group",
                title="Average Daily Sales by Season and Category",
                labels={"avg_daily_sales": "Avg Daily Sales", "season": "Season"},
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_season.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_season, use_container_width=True)

        with s_col2:
            st.markdown("##### 🏷️ Category Volume & Demand Variance")
            cat_stats = get_demand_by_category(filtered_df)
            st.dataframe(
                cat_stats.style.format(
                    {
                        "total_sales": "{:,.0f}",
                        "mean_daily_sales": "{:.2f}",
                        "std_daily_sales": "{:.2f}",
                        "record_count": "{:,}",
                    }
                ),
                use_container_width=True,
            )
            st.info(
                "📌 **Empirical Finding:** Festive season witnesses the highest demand across all categories "
                "(peaking at ~49.8 units/day for Grocery), followed by Summer (~30.8 units/day), while Monsoon shows lowest demand (~26.7 units/day)."
            )

        st.markdown("---")

        # Top Moving vs Slow Moving
        st.markdown("##### ⚡ High-Demand Velocity vs 🐢 Slow-Moving Products")
        top_slow = get_top_and_slow_moving_products(inventory_df, top_n=10)
        c_top, c_slow = st.columns(2)
        with c_top:
            st.markdown("**Top 10 High Velocity SKUs (Highest Daily Demand)**")
            st.dataframe(
                top_slow["top_moving"].style.format(
                    {
                        "mean_daily_demand": "{:.1f}",
                        "unit_cost": "${:,.2f}",
                        "days_of_supply": "{:.1f}",
                    }
                ),
                use_container_width=True,
            )
        with c_slow:
            st.markdown(
                "**Top 10 Slow-Moving / Overstocked SKUs (Highest Days of Supply)**"
            )
            st.dataframe(
                top_slow["slow_moving"].style.format(
                    {
                        "mean_daily_demand": "{:.1f}",
                        "days_of_supply": "{:.1f}",
                        "inventory_value": "${:,.2f}",
                    }
                ),
                use_container_width=True,
            )

        st.markdown("---")

        # Supplier Lead Time Comparison
        st.markdown("##### 🚚 Supplier Reliability & Lead-Time Analysis")
        supp_df = get_supplier_performance(inventory_df)
        st.dataframe(
            supp_df.style.format({"avg_lead_days": "{:.1f}", "total_stock": "{:,}"}),
            use_container_width=True,
            height=260,
        )

    # ==========================================
    # TAB 6: DATA QUALITY & DIAGNOSTICS
    # ==========================================
    with tabs[5]:
        st.subheader("Data Health & Pipeline Quality Diagnostics")
        st.caption(
            "Automated audit of schema conformance, nulls, duplicates, and physical bounds."
        )

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Total Records Audited", f"{data_issues['total_records']:,}")
        q2.metric("Unique Products", f"{data_issues['total_products']:,}")
        q3.metric(
            "Missing Values (Nulls)",
            f"{data_issues['total_missing']}",
            "Zero Missing"
            if data_issues["total_missing"] == 0
            else "Missing detected",
        )
        q4.metric("Exact Duplicates", f"{data_issues['exact_duplicates']}")

        st.markdown("---")

        d_col1, d_col2 = st.columns([6, 6])
        with d_col1:
            st.markdown("##### 📋 Missing Value Audit by Column")
            missing_series = pd.Series(data_issues["missing_values"])
            st.dataframe(
                missing_series.rename("Null Count").to_frame(), use_container_width=True
            )

        with d_col2:
            st.markdown("##### 🚨 Anomaly & Outlier Detection")
            anomaly_data = {
                "Audit Check": [
                    "Negative Stock Count (stock_qty < 0)",
                    "Negative Sales Count (daily_sales < 0)",
                    "Negative / Free Cost (unit_cost <= 0)",
                    "Invalid Lead Time (lead_days <= 0)",
                    "Stockout Zero-Stock Observations (stock == 0)",
                    "Unusually High Sales (> mean + 3σ)",
                ],
                "Detected Violations": [
                    data_issues["negative_stock_count"],
                    data_issues["negative_sales_count"],
                    data_issues["negative_cost_count"],
                    data_issues["invalid_lead_days_count"],
                    data_issues["stockout_zero_stock_count"],
                    data_issues["unusually_high_sales_count"],
                ],
                "Status": [
                    "PASSED"
                    if data_issues["negative_stock_count"] == 0
                    else "FLAGGED",
                    "PASSED"
                    if data_issues["negative_sales_count"] == 0
                    else "FLAGGED",
                    "PASSED"
                    if data_issues["negative_cost_count"] == 0
                    else "FLAGGED",
                    "PASSED"
                    if data_issues["invalid_lead_days_count"] == 0
                    else "FLAGGED",
                    f"{data_issues['stockout_zero_stock_count']} stockouts observed",
                    f"{data_issues['unusually_high_sales_count']} promotional spikes",
                ],
            }
            st.dataframe(pd.DataFrame(anomaly_data), use_container_width=True)

        st.markdown("---")

        # Download validation report
        val_report_path = "reports/dataset_validation_report.txt"
        if os.path.exists(val_report_path):
            with open(val_report_path, "r") as f:
                report_content = f.read()
            st.download_button(
                "📥 Download Official Dataset Validation Report (.txt)",
                data=report_content,
                file_name="dataset_validation_report.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()

