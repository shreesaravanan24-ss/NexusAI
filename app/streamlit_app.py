"""NexusAI Unified Interactive Analytics & AI Platform.

Streamlit dashboard connecting:
- Real-time KPI Cards (Gross, Returns, Net Revenue, Units Sold/Returned)
- Interactive Date & Country Filters
- Sales & Return Trends over Time (Plotly)
- Top Products Performance & Return Rates
- Machine Learning Sales Forecasting (Evaluation vs Naive Baseline)
- Statistical Anomaly Detection (Isolation Forest)
- Safe Natural Language Analytics Assistant (Query Allowlist)
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ai_assistant.assistant import GroundedAnalyticsAssistant
from ml.models import detect_transaction_anomalies, train_and_evaluate_forecasting
from pipelines.db_loader import (
    DEFAULT_DB_PATH,
    get_connection,
    get_country_performance,
    get_daily_trends,
    get_monthly_trends,
    get_overall_kpis,
    get_top_products,
    load_parquet_to_sqlite,
)

# Page configuration
st.set_page_config(
    page_title="NexusAI — Retail Big Data Analytics & AI Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def ensure_database_ready() -> bool:
    """Ensures SQLite analytical database is populated from Gold Parquet."""
    if not DEFAULT_DB_PATH.exists() or DEFAULT_DB_PATH.stat().st_size == 0:
        with st.spinner("Initializing analytical SQLite store from Gold Parquet..."):
            load_parquet_to_sqlite()
    return True


@st.cache_data(ttl=3600)
def load_kpis():
    conn = get_connection()
    try:
        return get_overall_kpis(conn)
    finally:
        conn.close()


@st.cache_data(ttl=3600)
def load_daily_data(start_date=None, end_date=None):
    conn = get_connection()
    try:
        return get_daily_trends(conn, start_date, end_date)
    finally:
        conn.close()


@st.cache_data(ttl=3600)
def load_monthly_data():
    conn = get_connection()
    try:
        return get_monthly_trends(conn)
    finally:
        conn.close()


@st.cache_data(ttl=3600)
def load_products(limit=15):
    conn = get_connection()
    try:
        return get_top_products(conn, limit)
    finally:
        conn.close()


@st.cache_data(ttl=3600)
def load_countries(limit=20):
    conn = get_connection()
    try:
        return get_country_performance(conn, limit)
    finally:
        conn.close()


@st.cache_data(ttl=3600)
def load_ml_forecast():
    return train_and_evaluate_forecasting()


@st.cache_data(ttl=3600)
def load_anomalies(sample_size=10000):
    return detect_transaction_anomalies(sample_size=sample_size)


# Main App Execution
def main():
    ensure_database_ready()

    st.title("🛍️ NexusAI Analytics Platform")
    st.caption("End-to-End Big Data Engineering (PySpark / Parquet) • SQL Analytics (SQLite) • ML Forecasting • Grounded AI Assistant")

    tabs = st.tabs([
        "📊 Executive KPI & Trends",
        "📦 Product & Market Performance",
        "🤖 ML Forecasting & Anomalies",
        "💬 AI Analytics Assistant",
    ])

    # TAB 1: EXECUTIVE KPIS & TRENDS
    with tabs[0]:
        st.subheader("Executive Financial Summary")
        kpis = load_kpis()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gross Revenue", f"£{kpis['gross_revenue']:,.2f}")
        c2.metric("Returns Loss", f"£{kpis['returns_loss']:,.2f}", delta=f"-{(kpis['returns_loss']/kpis['gross_revenue'])*100:.2f}%", delta_color="inverse")
        c3.metric("Net Revenue", f"£{kpis['net_revenue']:,.2f}")
        c4.metric("Units Sold / Returned", f"{kpis['units_sold']:,}", f"-{kpis['units_returned']:,} ret")

        st.divider()

        # Date filter
        all_daily = load_daily_data()
        min_date = pd.to_datetime(all_daily["invoice_date"]).min().date()
        max_date = pd.to_datetime(all_daily["invoice_date"]).max().date()

        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            date_range = st.date_input("Filter Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)

        start_str = date_range[0].strftime("%Y-%m-%d") if len(date_range) > 0 else None
        end_str = date_range[1].strftime("%Y-%m-%d") if len(date_range) > 1 else None

        filtered_daily = load_daily_data(start_str, end_str)

        if filtered_daily.empty:
            st.warning("No data found for the selected date range.")
        else:
            fig_trend = px.line(
                filtered_daily,
                x="invoice_date",
                y=["gross_revenue", "returns_loss", "net_revenue"],
                labels={"value": "Revenue (£)", "invoice_date": "Date", "variable": "Metric"},
                title=f"Daily Revenue & Returns Trend ({start_str} to {end_str})",
                color_discrete_map={
                    "gross_revenue": "#2b5c8f",
                    "returns_loss": "#e63946",
                    "net_revenue": "#2a9d8f",
                },
            )
            fig_trend.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_trend, use_container_width=True)

        # Monthly aggregation
        st.subheader("Monthly Performance Overview")
        monthly_df = load_monthly_data()
        fig_monthly = px.bar(
            monthly_df,
            x="year_month",
            y=["gross_revenue", "returns_loss"],
            barmode="group",
            labels={"value": "Amount (£)", "year_month": "Month", "variable": "Revenue Component"},
            title="Monthly Gross Sales vs. Returns Loss",
            color_discrete_map={"gross_revenue": "#457b9d", "returns_loss": "#e63946"},
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    # TAB 2: PRODUCTS & MARKETS
    with tabs[1]:
        st.subheader("Product-Level Analytics")
        prod_limit = st.slider("Number of Top Products to Display", 5, 50, 10)
        prod_df = load_products(prod_limit)

        fig_prod = px.bar(
            prod_df,
            x="net_sales",
            y="product_description",
            orientation="h",
            title=f"Top {prod_limit} Products by Net Sales (£)",
            labels={"net_sales": "Net Sales (£)", "product_description": "Product"},
            color="net_sales",
            color_continuous_scale="Viridis",
        )
        fig_prod.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_prod, use_container_width=True)

        st.dataframe(
            prod_df[["stock_code", "product_description", "units_sold", "units_returned", "gross_sales", "net_sales", "return_rate_pct"]],
            use_container_width=True,
            column_config={
                "gross_sales": st.column_config.NumberColumn(format="£%.2f"),
                "net_sales": st.column_config.NumberColumn(format="£%.2f"),
                "return_rate_pct": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

        st.divider()
        st.subheader("Geographic Market Distribution")
        country_df = load_countries(15)

        fig_country = px.bar(
            country_df,
            x="country",
            y="net_revenue",
            title="Top 15 Countries by Net Revenue (£)",
            labels={"net_revenue": "Net Revenue (£)", "country": "Country"},
            color="net_revenue",
            color_continuous_scale="Teal",
        )
        st.plotly_chart(fig_country, use_container_width=True)

    # TAB 3: MACHINE LEARNING
    with tabs[2]:
        st.subheader("Sales Demand Forecasting")
        st.caption("Chronological train/test validation (80% train, 20% test) using Ridge regression vs naive persistence baseline.")

        with st.spinner("Evaluating forecasting model..."):
            ml_results = load_ml_forecast()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model MAE", f"£{ml_results['model_mae']:,.2f}")
        m2.metric("Baseline MAE", f"£{ml_results['baseline_mae']:,.2f}")
        m3.metric("MAE Improvement", f"{ml_results['mae_improvement_pct']:.2f}%", delta=f"{ml_results['mae_improvement_pct']:.2f}%")
        m4.metric("Model RMSE", f"£{ml_results['model_rmse']:,.2f}")

        pred_df = ml_results["test_predictions"]
        fig_ml = go.Figure()
        fig_ml.add_trace(go.Scatter(x=pred_df["date"], y=pred_df["actual"], mode="lines", name="Actual Net Revenue", line=dict(color="#1f77b4")))
        fig_ml.add_trace(go.Scatter(x=pred_df["date"], y=pred_df["predicted"], mode="lines", name="Model Predicted", line=dict(color="#2ca02c", dash="dash")))
        fig_ml.add_trace(go.Scatter(x=pred_df["date"], y=pred_df["baseline"], mode="lines", name="Naive Baseline (t-1)", line=dict(color="#ff7f0e", dash="dot")))
        fig_ml.update_layout(title="Out-of-Sample Forecasting: Actual vs Predicted vs Naive Baseline", hovermode="x unified")
        st.plotly_chart(fig_ml, use_container_width=True)

        st.info("⚠️ **Methodology Note:** Train date range: " + ml_results["train_date_range"][0] + " to " + ml_results["train_date_range"][1] +
                " | Test date range: " + ml_results["test_date_range"][0] + " to " + ml_results["test_date_range"][1] +
                ". Zero future lookahead ensures strict temporal validity.")

        st.divider()
        st.subheader("Transaction Outlier Detection (Isolation Forest)")
        st.caption("Identifies multivariate statistical anomalies in quantity, unit price, and net amount. Does NOT claim confirmed fraud or inventory stockouts.")

        anom_df = load_anomalies()
        st.dataframe(
            anom_df[["invoice_no", "stock_code", "description", "quantity", "unit_price", "net_amount", "country"]].head(25),
            use_container_width=True,
            column_config={
                "unit_price": st.column_config.NumberColumn(format="£%.2f"),
                "net_amount": st.column_config.NumberColumn(format="£%.2f"),
            },
        )

    # TAB 4: GROUNDED AI ASSISTANT
    with tabs[3]:
        st.subheader("💬 NexusAI Grounded Analytics Assistant")
        st.caption("Natural language analytics grounded 100% in verified project data. Operates under strict read-only template enforcement.")

        assistant = GroundedAnalyticsAssistant()

        # Quick question buttons
        st.write("**Suggested Questions:**")
        qc1, qc2, qc3 = st.columns(3)
        sample_q = ""
        if qc1.button("📊 What is total revenue and returns?"):
            sample_q = "What is total revenue and returns?"
        if qc2.button("🏆 What are the top 5 products?"):
            sample_q = "What are the top 5 products?"
        if qc3.button("🌍 Which countries generate the most sales?"):
            sample_q = "Which countries generate the most sales?"

        user_query = st.text_input("Ask a question about the retail data:", value=sample_q, placeholder="e.g. What is the return rate?")

        if user_query:
            with st.spinner("Analyzing verified metrics..."):
                resp = assistant.ask(user_query)
            st.success(resp["answer"])
            if resp["data"]:
                with st.expander("View Source Data"):
                    st.json(resp["data"][:5])


if __name__ == "__main__":
    main()
