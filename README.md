# Optimized Retail Inventory Management System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Dashboard](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)](https://streamlit.io/)
[![Visualizations](https://img.shields.io/badge/Plotly-5.20%2B-blueviolet.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Pytest-21%20Passed-brightgreen.svg)](https://pytest.org/)

An enterprise-grade, data-driven retail inventory management and demand forecasting platform developed for college SCL evaluation. It bridges machine learning demand forecasting with quantitative operations research formulas (Safety Stock, Reorder Point, and Economic Order Quantity) inside an interactive Streamlit executive dashboard.

---

## 📌 Project Overview & Problem Statement

Modern omnichannel retail environments face dual opposing pressures:
1. **Stockout Risks**: Underestimating demand or suffering vendor lead-time delays leads to missed sales, depleted shelves, and lost customer goodwill.
2. **Excess Inventory Capital**: Overstocking locks up working capital in holding costs, warehouse storage fees, and depreciation risk.

This project delivers an end-to-end autonomous decision support system that ingests historical retail transaction logs, monitors inventory depletion rates, forecasts future customer demand over multi-day horizons, computes mathematical reorder thresholds, and generates prioritized purchase recommendations.

---

## 🎯 Key Capabilities & Features

* **Data Ingestion & Integrity Auditing**: Strict schema validation, missing value detection, zero-duplicate enforcement, and anomaly screening (stockouts, promotional sales spikes, lead-time checks).
* **Leakage-Free Feature Pipeline**: Temporal calendar extraction, cyclical trigonometric encodings, and strictly shifted ($t-1$) historical rolling demand statistics.
* **Empirical Seasonal Analytics**: Rigorous data analysis capturing verified seasonal demand shifts (e.g., Festive demand surge +40.7% over Monsoon).
* **Machine Learning Demand Forecasting**: Compares Dummy and Rolling-7 baselines against HistGradientBoosting and Random Forest models using chronological train/validation splits (no future data leakage).
* **Inventory Optimization Engine**:
  * **Safety Stock ($SS$)**: Calculated using demand volatility and supplier lead times ($Z \cdot \sigma_d \cdot \sqrt{L}$) with defensive fallback buffers.
  * **Reorder Point ($ROP$)**: Automatic threshold calculation based on lead-time demand and safety stock buffer.
  * **Suggested Order Quantity**: Dynamic replenishment sizing targeting future planning horizons without negative orders.
  * **Economic Order Quantity ($EOQ$)**: Classical Wilson batch sizing with user-configurable setup costs and holding rates.
  * **4-Tier Priority Classification**: Discrete urgency badges (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).
* **Interactive Executive Dashboard**: Polished Streamlit UI with Plotly charts, global SKU/Category/Season filters, interactive forecasts with 95% confidence intervals, and one-click CSV export capabilities.
* **Automated Test Suite**: 17 unit and integration tests covering data loading, inventory mathematics, and ML forecasting pipelines.

---

## 🏗️ Project Architecture & Layout

```text
retail-inventory-system/
├── data/                                 # Datasets & data dictionary
│   ├── dataset.csv                       # Consolidated 5,000-record master dataset
│   ├── dataset_additional_4500.csv       # Additional 4,500 records
│   └── data_dictionary.csv               # Official LMS column descriptions
├── models/                               # Serialized ML artifacts
│   ├── demand_forecast_model.joblib      # Fitted best model, features, metrics
│   └── .gitkeep
├── reports/                              # Generated validation and audit reports
│   ├── dataset_validation_report.txt     # Formal 5,000-record audit report
│   ├── model_evaluation_report.txt       # Holdout benchmark metrics report
│   └── .gitkeep
├── src/                                  # Modular Python source code
│   ├── __init__.py                       # Package initializer
│   ├── data_loader.py                    # Ingestion, schema validation, anomaly detection
│   ├── features.py                       # Calendar, cyclical & leakage-free lag features
│   ├── train.py                          # Chronological split, model training & Joblib export
│   ├── forecasting.py                    # Multi-day recursive forecasting & confidence bounds
│   ├── inventory_logic.py                # Operations research formulas (ROP, SS, EOQ, Priority)
│   ├── analytics.py                      # Executive KPIs, trends, category & supplier metrics
│   └── app.py                            # Streamlit multi-tab interactive dashboard
├── tests/                                # Pytest test suites
│   ├── __init__.py
│   ├── test_data_loader.py               # Schema & data loading tests
│   ├── test_inventory_logic.py           # Inventory mathematical formula tests
│   └── test_forecasting.py               # Feature pipeline & ML inference tests
├── .gitignore                            # Environment and cache ignore rules
├── pytest.ini                            # Pytest configuration
├── requirements.txt                      # Project dependency definitions
└── README.md                             # Documentation and user manual
```

---

## 📊 Dataset Specification & Empirical Analysis

The master dataset located at `data/dataset.csv` contains **5,000 records** spanning 500 unique product SKUs across all 12 months of calendar year 2024.

### Schema Fields
| # | Column | Data Type | Description |
|---|---|---|---|
| 1 | `date` | datetime | Transaction date (`2024-01-01` to `2024-12-31`) |
| 2 | `product_id` | string | Unique SKU identifier (`SKU0001` - `SKU0500`) |
| 3 | `category` | string | Product classification (`Grocery`, `Electronics`, `Apparel`, `HomeDecor`) |
| 4 | `season` | string | Seasonal tag (`Winter`, `Summer`, `Monsoon`, `Festive`) |
| 5 | `stock_qty` | integer | Current physical on-hand inventory (units) |
| 6 | `reorder_level`| integer | Existing reorder point threshold |
| 7 | `daily_sales` | integer | Recorded sales volume for that date |
| 8 | `unit_cost` | float | Unit wholesale procurement price ($) |
| 9 | `supplier_id` | string | Supplier identifier (`SUP01` - `SUP30`) |
| 10| `lead_days` | integer | Supplier replenishment delivery time (days) |

### Empirical Seasonal Findings
Rather than assuming theoretical seasonal demand, empirical analysis of the 5,000 records proves:
* **Festive Season (Oct - Nov)**: Highest daily sales volume across all categories, averaging **37.58 units/day** (+40.7% higher than Monsoon; Grocery leads at 49.8 units/day).
* **Summer Season (Mar - Jun)**: Secondary sales surge averaging **30.84 units/day** (+15.5% over Monsoon).
* **Winter Season (Jan, Feb, Dec)**: Moderate baseline demand averaging **28.25 units/day**.
* **Monsoon Season (Jul - Sep)**: Lowest overall sales volume averaging **26.71 units/day**.
* **Season-Month Consistency**: 100% strict mapping with zero date-season contradictions.

---

## 🧮 Inventory Optimization Formulas

### 1. Safety Stock ($SS$)
Protects against demand surges and delayed supplier shipments:
$$SS = Z \times \sigma_d \times \sqrt{L}$$
* $Z$: Service factor ($1.65 \approx 95\%$ in-stock probability; configurable in sidebar).
* $\sigma_d$: Historical daily sales standard deviation for the product.
* $L$: Supplier replenishment lead time in days.
* *Fallback*: If historical standard deviation is unavailable or zero, applies $0.20 \times \bar{D} \times L$ (20% lead-time buffer).

### 2. Reorder Point ($ROP$)
Replenishment trigger threshold:
$$ROP = (\bar{D} \times L) + SS$$
where $\bar{D}$ is expected daily demand.

### 3. Suggested Order Quantity ($Q_{\text{order}}$)
Replenishment batch sizing to cover expected demand over a planning horizon $T$ (e.g. 14 days):
$$Q_{\text{order}} = \max\Big(0, \big(\bar{D} \times (L + T) + SS\big) - \text{Current Stock}\Big)$$
Guaranteed to never produce negative replenishment quantities.

### 4. Economic Order Quantity ($EOQ$)
Balances ordering setup costs against warehouse holding capital:
$$EOQ = \sqrt{\frac{2 \times D_{\text{annual}} \times S}{H}}$$
* $D_{\text{annual}} = \bar{D} \times 365$
* $S$: Administrative ordering cost per purchase order (default $\$50.00$, configurable in UI).
* $H$: Annual holding cost per unit ($H = \text{unit\_cost} \times \text{holding\_rate}$, default 20%).

### 5. Priority Classification Matrix
* **`CRITICAL`**: Current Stock $= 0$ OR Current Stock $<$ Lead Time Demand ($\bar{D} \times L$).
* **`HIGH`**: Current Stock $\le ROP$.
* **`MEDIUM`**: $ROP < \text{Current Stock} \le 1.5 \times ROP$.
* **`LOW`**: Current Stock $> 1.5 \times ROP$ (healthy inventory cushion).

---

## 🤖 Demand Forecasting Evaluation

To prevent temporal data leakage, models are evaluated using a rigorous **3-way chronological split**:
* **Training Set**: 3,627 observations (`2024-01-01` to `2024-09-12`)
* **Validation Set**: 676 observations (`2024-09-13` to `2024-11-06`) — *Used strictly for model selection*
* **Holdout Test Set**: 697 observations (`2024-11-07` to `2024-12-31`) — *Final unbiased generalization evaluation*

### 1. Model Selection on Validation Set (Pre-Test Decision)
| Candidate Model / Baseline | MAE (units/day) | RMSE (units/day) | $R^2$ Score | WAPE (%) |
|---|---|---|---|---|
| **Dummy Mean Baseline** | 10.706 | 15.986 | -0.1346 | 31.13% |
| **Rolling-7 Baseline** | 11.090 | 16.295 | -0.1790 | 32.24% |
| **HistGradientBoosting** | 10.186 | 15.400 | -0.0530 | 29.61% |
| **Random Forest (Selected)** | **9.987** | **15.109** | **-0.0135** | **29.04%** |

*Random Forest achieved the lowest validation RMSE (15.109) and was selected as the production model prior to evaluating the holdout test set.*

### 2. Generalization Performance on Unseen Holdout Test Set
| Model / Benchmark | MAE (units/day) | RMSE (units/day) | $R^2$ Score | WAPE (%) | Outcome |
|---|---|---|---|---|---|
| **Dummy Mean Baseline** | 10.194 | 16.052 | -0.0382 | 31.89% | Naive mean benchmark |
| **Rolling-7 Baseline** | 11.266 | 16.536 | -0.1017 | 35.24% | Lagged rolling benchmark |
| **HistGradientBoosting** | 9.429 | 15.917 | -0.0208 | 29.49% | Gradient boosting candidate |
| **Random Forest (Production)** | **9.195** | **15.553** | **0.0254** | **28.76%** | **Best test performance** |

> [!NOTE]
> **Heuristic Prediction Interval & Accuracy Bounds Disclosure**:
> 1. Real-world retail sales with 500 SKUs exhibit high daily intermittent variability ($\sigma \approx 14.3$ units).
> 2. The Random Forest model reduces MAE by ~1.0 unit/day and RMSE by ~0.5 unit/day compared to the naive mean baseline.
> 3. Prediction intervals ($\pm 1.96 \times \text{RMSE}$) are approximate error-based heuristic intervals under an assumed normal residual distribution, not statistically calibrated 95% conformal prediction intervals.

---

## ⚡ Quick Start & Run Guide

### 1. Activate Environment

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt / PowerShell)
.venv\Scripts\activate
```

### 2. Verify / Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests

Execute the Pytest test suite:

```bash
pytest -v
```

Expected result: `17 passed in ~1.2s`.

### 4. Train or Retrain Forecast Models

```bash
python -m src.train
```

Serializes model to `models/demand_forecast_model.joblib` and generates `reports/model_evaluation_report.txt`.

### 5. Launch the Streamlit Web Application

```bash
streamlit run src/app.py
```

The web dashboard will open in your default browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Dashboard Modules

1. **📊 Executive Overview**: Real-time KPI summary (Total Products, Units, Capital Value, Reorder Deficits, Critical Stockouts), weekly sales trajectory line charts, and category capital distribution pie charts.
2. **📦 Inventory Stock Management**: Searchable and filterable SKU ledger with priority filtering, supplier breakdown, and CSV export.
3. **📈 ML Demand Forecasting**: Interactive product-level demand forecasting for 7, 14, or 30 days ahead with historical comparison and 95% confidence interval bands ($\pm 1.96 \cdot \text{RMSE}$).
4. **🛒 Reorder Recommendations**: Actionable procurement list specifying suggested reorder quantities, EOQ recommendations, and order justifications.
5. **🌦️ Seasonal Analytics**: Empirical comparison across Festive, Summer, Monsoon, and Winter seasons alongside top-velocity and slow-moving product rankings.
6. **🔍 Data Quality & Diagnostics**: Real-time audit of schema conformance, null count, exact duplicates, physical boundaries, and downloadable official validation report.

---

## ⚠️ Known Limitations & Future Enhancements

* **Dataset Granularity**: The dataset consists of periodic SKU snapshot records rather than continuous second-by-second cashier scanner receipts.
* **Supplier Constraints**: Lead times are modeled as static product-level attributes; future work could model stochastic lead-time distributions.
* **Future Roadmap**:
  * Multi-echelon inventory optimization across central warehouse and satellite stores.
  * Integration with live ERP APIs (SAP, Oracle NetSuite).
  * Automated replenishment purchase order generation via email/webhook notifications.
# optimized-retail-inventory-management
