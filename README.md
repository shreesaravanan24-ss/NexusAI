# NexusAI: Big Data Engineering, Analytics & AI Platform

NexusAI is a unified, local-first Big Data Engineering, SQL Analytics, Machine Learning, and Grounded AI platform built on authentic retail transaction data.

---

## 1. Architecture Overview

```
[Raw CSV / Excel] ──► [PySpark ETL] ──► [Silver Parquet] ──► [Gold Parquet]
(1,067,371 rows)       (Data Hygiene)    (Partitioned)        (Aggregations)
                                                                    │
           ┌────────────────────────────────────────────────────────┤
           ▼                                                        ▼
  [SQLite Analytics Store]                                  [Machine Learning]
  (Reconciled DDL & Views)                                  (Ridge / IsoForest)
           │                                                        │
           └────────────────────────┬───────────────────────────────┘
                                    ▼
                      [Streamlit Dashboard & AI Assistant]
                      (KPIs, Trends, Forecasting, Safe Chat)
```

---

## 2. Dataset Attribution & License

* **Source:** [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI Machine Learning Repository, ID 502)
* **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**)
* **Scope:** 1,067,371 authentic transactions (2009–2011) from a UK-based non-store online retailer.
* **Fields:** `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`.

---

## 3. Verified Financial & Accounting Ground Truth

The PySpark ETL and SQLite SQL Analytics layers reconcile to the exact penny:

$$\text{Net Revenue} = \text{Gross Revenue} - \text{Returns Loss}$$

| Financial Metric | Verified Value | Notes |
|:---|:---|:---|
| **Gross Revenue** | **£20,814,291.98** | Sales where $\text{Quantity} > 0$ |
| **Returns Loss** | **£1,527,041.43** | Returns / cancellations where $\text{Quantity} < 0$ or Invoice starts with 'C' |
| **Net Revenue** | **£19,287,250.55** | Net sales ($\text{Gross} - \text{Returns}$) |
| **Units Sold** | **11,672,570** | Total merchandise units sold |
| **Units Returned** | **1,064,078** | Total merchandise units returned (9.12% unit return rate) |
| **Total Invoices** | **53,628** | Across 604 trading days |

---

## 4. Implemented Features by Phase

### Phase 1 & 2: Big Data Engineering (PySpark & Parquet)
* **Zero Silent Drops:** Preserves 100% of raw transactions (1,067,371 records) into Silver Parquet without artificial price caps or row deletion.
* **Data Hygiene & Flags:** Explicitly computes `is_return`, `is_guest_customer`, and `is_zero_or_negative_price`.
* **Stock Code Normalization:** Safely converts numeric codes like `22423.0` to `22423` while preserving alphanumeric IDs (e.g. `85123A`, `DOT`).
* **Partitioned Parquet:** Silver and Gold daily tables partitioned by `invoice_year` (2009, 2010, 2011).

### Phase 3: SQL Analytics (SQLite Local-First)
* High-performance SQLite database (`data/nexusai.db`) populated directly from Parquet.
* Reusable, tested query functions in `pipelines/db_loader.py` for KPIs, trends, top products, and country distributions.
* DDL schemas (`sql/schema.sql`) and analytical queries (`sql/views.sql`).

### Phase 4: Interactive Dashboard (Streamlit & Plotly)
* **Executive KPIs:** Real-time Gross, Returns, Net, and Units KPI cards.
* **Date Range Filtering:** Interactive daily revenue and returns time-series.
* **Monthly Performance:** Grouped bar charts visualizing gross sales vs. return loss.
* **Product & Market Performance:** Top products bar charts with return rate metrics and country geographic distribution.

### Phase 5: Machine Learning
* **Sales Demand Forecasting:** Chronological 80/20 train/test split (no future lookahead / data leakage).
  * **Model:** Ridge Regression with lag features ($t-1, t-2, t-7, t-14$) and calendar features.
  * **Evaluation:** Model MAE **£11,652.18** vs. Naive Persistence Baseline MAE **£17,749.07** (**+34.35% improvement**).
* **Transaction Outlier Detection:** Multivariate Isolation Forest ($0.49\%$ detected anomaly rate on quantity, price, and net amount).
  * *Important limitation note:* Statistical anomalies represent atypical transactions (e.g. bulk purchases, administrative entries) and do NOT constitute confirmed fraud or physical stockouts.

### Phase 6: Grounded AI Analytics Assistant
* Safe natural language Q&A grounded 100% in verified SQLite/Gold data.
* **Security Guardrails:** AST/Regex validator blocks non-SELECT queries, DDL/DML (`DROP`, `DELETE`, `INSERT`, `UPDATE`), and SQL injection attempts.
* **Intent Allowlist:** Maps queries to verified metrics (KPIs, return rates, top products, country ranking, monthly trends).

---

## 5. Execution Commands

### Prerequisites
* Windows OS with Python 3.12.
* Portable JDK 17 (managed automatically by `pipelines/env_setup.py`).

### Activate Environment
```powershell
cd C:\Users\User\Downloads\NexusAI
.\.venv\Scripts\Activate.ps1
```

### Run Full Test Suite
```powershell
# 1. Pipeline & Ingestion Regression Test (7-row fixture)
python tests/test_pipeline.py

# 2. SQL Analytics, ML Evaluation & Security Guardrails Test
python tests/test_analytics.py
```

### Run Batch ETL Pipeline (Optional Re-run)
```powershell
python -m pipelines.spark_etl --input data/raw/online_retail_raw.csv
```

### Load SQLite Analytics Store
```powershell
python pipelines/db_loader.py
```

### Launch Interactive Dashboard
```powershell
streamlit run app/streamlit_app.py
```

---

## 6. Limitations & Scope Clarifications

1. **Physical Inventory:** The UCI dataset captures transaction receipts, not warehouse bin balances. Demand metrics represent order volume, not on-hand stock levels.
2. **Anomaly Classification:** Anomaly detection isolates statistical outliers; it does not replace fraud investigation workflows.
3. **Forecasting Scope:** Daily aggregate forecasting operates on macroscopic order volumes; individual SKU demand requires larger per-item history.

---

## 7. Recommended Dashboard Screenshots (For Portfolio / GitHub)

When publishing to GitHub, capture and include screenshots in `assets/`:
1. `assets/kpi_overview.png`: Tab 1 Executive Financial Summary cards and daily trend line chart.
2. `assets/product_analytics.png`: Tab 2 Top products horizontal bar chart and return rate table.
3. `assets/ml_forecasting.png`: Tab 3 Actual vs. Model vs. Naive Baseline time series plot.
4. `assets/ai_assistant.png`: Tab 4 Grounded AI Assistant answering natural language retail questions.
