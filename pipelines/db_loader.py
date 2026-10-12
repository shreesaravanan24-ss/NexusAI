"""NexusAI SQL Analytics Store and Database Loader.

Loads Gold and Silver Parquet tables into local SQLite database.
Provides reusable, tested analytical queries with zero duplicate records.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NexusAI.DBLoader")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "nexusai.db"


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def load_parquet_to_sqlite(
    db_path: Path | str = DEFAULT_DB_PATH,
    gold_daily_dir: Path | str = BASE_DIR / "data" / "gold" / "daily_sales_summary",
    gold_product_dir: Path | str = BASE_DIR / "data" / "gold" / "product_sales_summary",
    silver_dir: Path | str = BASE_DIR / "data" / "silver" / "retail_transactions",
) -> dict[str, int]:
    """Repeatably loads Parquet data into SQLite replacing existing tables."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {}
    with sqlite3.connect(str(db_path)) as conn:
        # 1. Load Daily Sales Summary
        logger.info("Loading daily sales from %s...", gold_daily_dir)
        daily_df = pq.read_table(str(gold_daily_dir)).to_pandas()
        daily_df["invoice_date"] = pd.to_datetime(daily_df["invoice_date"]).dt.strftime("%Y-%m-%d")
        daily_df.to_sql("daily_sales_summary", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_sales_summary(invoice_date)")
        counts["daily_sales_summary"] = len(daily_df)

        # 2. Load Product Sales Summary
        logger.info("Loading product sales from %s...", gold_product_dir)
        product_df = pq.read_table(str(gold_product_dir)).to_pandas()
        product_df["stock_code"] = product_df["stock_code"].astype(str)
        product_df.to_sql("product_sales_summary", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_product_stock ON product_sales_summary(stock_code)")
        counts["product_sales_summary"] = len(product_df)

        # 3. Load Country Aggregates from Silver
        logger.info("Computing and loading country sales from %s...", silver_dir)
        silver_table = pq.read_table(
            str(silver_dir),
            columns=["country", "invoice_no", "quantity", "gross_amount", "return_amount", "net_amount", "customer_id"],
        )
        silver_df = silver_table.to_pandas()
        country_df = (
            silver_df.groupby("country", dropna=False)
            .agg(
                total_invoices=("invoice_no", "nunique"),
                unique_customers=("customer_id", lambda s: s.dropna().nunique()),
                units_sold=("quantity", lambda s: int(s[s > 0].sum())),
                units_returned=("quantity", lambda s: int(abs(s[s < 0].sum()))),
                gross_revenue=("gross_amount", lambda s: round(float(s.sum()), 2)),
                returns_loss=("return_amount", lambda s: round(float(s.sum()), 2)),
                net_revenue=("net_amount", lambda s: round(float(s.sum()), 2)),
            )
            .reset_index()
            .sort_values("net_revenue", ascending=False)
        )
        country_df.to_sql("country_sales_summary", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_country_name ON country_sales_summary(country)")
        counts["country_sales_summary"] = len(country_df)

    logger.info("SQLite loading complete: %s", counts)
    return counts


# --- Reusable Query Functions ---

def get_overall_kpis(conn: sqlite3.Connection) -> dict[str, Any]:
    """Returns top-level KPIs reconciled against Gold data."""
    cur = conn.cursor()
    cur.execute("""
        SELECT
            ROUND(SUM(gross_revenue), 2) AS gross_revenue,
            ROUND(SUM(returns_loss), 2)  AS returns_loss,
            ROUND(SUM(net_revenue), 2)   AS net_revenue,
            SUM(units_sold)              AS units_sold,
            SUM(units_returned)          AS units_returned,
            SUM(total_invoices)          AS total_invoices,
            COUNT(DISTINCT invoice_date) AS total_trading_days
        FROM daily_sales_summary
    """)
    row = dict(cur.fetchone())
    return row


def get_daily_trends(
    conn: sqlite3.Connection,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Returns daily trend data with optional date filtering."""
    query = """
        SELECT
            invoice_date,
            total_invoices,
            units_sold,
            units_returned,
            gross_revenue,
            returns_loss,
            net_revenue,
            unique_customers,
            invoice_year
        FROM daily_sales_summary
        WHERE 1=1
    """
    params: list[Any] = []
    if start_date:
        query += " AND invoice_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND invoice_date <= ?"
        params.append(end_date)
    query += " ORDER BY invoice_date ASC"

    return pd.read_sql_query(query, conn, params=params)


def get_monthly_trends(conn: sqlite3.Connection) -> pd.DataFrame:
    """Aggregates daily figures to monthly sales trends."""
    query = """
        SELECT
            SUBSTR(invoice_date, 1, 7) AS year_month,
            SUM(total_invoices)        AS total_invoices,
            SUM(units_sold)            AS units_sold,
            SUM(units_returned)        AS units_returned,
            ROUND(SUM(gross_revenue), 2) AS gross_revenue,
            ROUND(SUM(returns_loss), 2)  AS returns_loss,
            ROUND(SUM(net_revenue), 2)   AS net_revenue,
            SUM(unique_customers)      AS active_customers
        FROM daily_sales_summary
        GROUP BY SUBSTR(invoice_date, 1, 7)
        ORDER BY year_month ASC
    """
    return pd.read_sql_query(query, conn)


def get_top_products(conn: sqlite3.Connection, limit: int = 10) -> pd.DataFrame:
    """Returns top products by net sales."""
    query = """
        SELECT
            stock_code,
            product_description,
            order_frequency,
            units_sold,
            units_returned,
            gross_sales,
            net_sales,
            return_rate_pct
        FROM product_sales_summary
        ORDER BY net_sales DESC
        LIMIT ?
    """
    return pd.read_sql_query(query, conn, params=[limit])


def get_country_performance(conn: sqlite3.Connection, limit: int = 15) -> pd.DataFrame:
    """Returns sales performance by country."""
    query = """
        SELECT
            country,
            total_invoices,
            unique_customers,
            units_sold,
            units_returned,
            gross_revenue,
            returns_loss,
            net_revenue
        FROM country_sales_summary
        ORDER BY net_revenue DESC
        LIMIT ?
    """
    return pd.read_sql_query(query, conn, params=[limit])


if __name__ == "__main__":
    res = load_parquet_to_sqlite()
    print("Database built successfully:", res)
    conn = get_connection()
    kpis = get_overall_kpis(conn)
    print("Overall KPIs:", kpis)
    top = get_top_products(conn, 3)
    print("\nTop 3 Products:\n", top)
    conn.close()
