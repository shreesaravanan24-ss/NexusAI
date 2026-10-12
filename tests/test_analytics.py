"""Comprehensive Verification Suite for Phases 3, 4, 5 and 6.

Validates:
1. SQL Analytics Reconciliation against Gold Parquet ground truth.
2. ML Model Temporal Validity (no data leakage, valid metrics, outperforms baseline).
3. AI Assistant Security Guardrails (rejection of DROP/DELETE/INSERT, valid intent routing).
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pyarrow.parquet as pq
from ai_assistant.assistant import GroundedAnalyticsAssistant, validate_read_only_sql
from ml.models import prepare_forecasting_data, train_and_evaluate_forecasting
from pipelines.db_loader import (
    DEFAULT_DB_PATH,
    get_connection,
    get_country_performance,
    get_daily_trends,
    get_overall_kpis,
    get_top_products,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NexusAI.AnalyticsTests")


def test_sql_reconciliation():
    """Reconciles SQLite KPIs against Gold Parquet data."""
    logger.info("Test 1: Reconciling SQLite KPIs against Gold Parquet...")
    conn = get_connection(DEFAULT_DB_PATH)
    try:
        sql_kpis = get_overall_kpis(conn)

        # Ground truth directly from Gold Parquet
        gold_df = pq.read_table(str(BASE_DIR / "data" / "gold" / "daily_sales_summary")).to_pandas()
        gold_gross = round(float(gold_df["gross_revenue"].sum()), 2)
        gold_returns = round(float(gold_df["returns_loss"].sum()), 2)
        gold_net = round(float(gold_df["net_revenue"].sum()), 2)

        assert abs(sql_kpis["gross_revenue"] - gold_gross) < 0.01, f"Gross mismatch: {sql_kpis['gross_revenue']} vs {gold_gross}"
        assert abs(sql_kpis["returns_loss"] - gold_returns) < 0.01, f"Returns mismatch: {sql_kpis['returns_loss']} vs {gold_returns}"
        assert abs(sql_kpis["net_revenue"] - gold_net) < 0.01, f"Net mismatch: {sql_kpis['net_revenue']} vs {gold_net}"
        # Invariant check
        diff = abs(round(sql_kpis["gross_revenue"] - sql_kpis["returns_loss"], 2) - sql_kpis["net_revenue"])
        assert diff < 0.01, f"Accounting invariant violated in SQLite: diff={diff}"
        logger.info("  Reconciliation PASSED: Gross=£%.2f, Returns=£%.2f, Net=£%.2f (Invariant diff=%.4f)",
                    sql_kpis["gross_revenue"], sql_kpis["returns_loss"], sql_kpis["net_revenue"], diff)
    finally:
        conn.close()


def test_ml_data_leakage_and_metrics():
    """Validates temporal splitting and forecasting performance against naive baseline."""
    logger.info("Test 2: Validating ML temporal split and evaluation...")
    df = prepare_forecasting_data()

    # Verify chronological ordering
    assert df["invoice_date"].is_monotonic_increasing, "Data is not chronologically ordered!"

    res = train_and_evaluate_forecasting(test_ratio=0.20)
    train_end = res["train_date_range"][1]
    test_start = res["test_date_range"][0]

    # Verify no temporal overlap (train ends before test begins)
    assert train_end < test_start, f"Data leakage detected! Train end ({train_end}) >= Test start ({test_start})"

    # Verify model metrics
    assert res["model_mae"] > 0, "Model MAE must be positive."
    assert res["baseline_mae"] > 0, "Baseline MAE must be positive."
    assert res["model_mae"] < res["baseline_mae"], f"Model ({res['model_mae']}) must outperform naive baseline ({res['baseline_mae']})"
    assert res["mae_improvement_pct"] > 0, "Improvement percentage must be positive."
    logger.info("  ML Evaluation PASSED: Model MAE £%.2f beats Baseline £%.2f by %.2f%%",
                res["model_mae"], res["baseline_mae"], res["mae_improvement_pct"])


def test_ai_assistant_guardrails():
    """Validates security guardrails and intent grounding."""
    logger.info("Test 3: Validating AI Assistant security guardrails and query restriction...")

    # Malicious queries must be blocked
    malicious = [
        "DROP TABLE daily_sales_summary",
        "DELETE FROM product_sales_summary WHERE stock_code='22423'",
        "INSERT INTO daily_sales_summary VALUES ('2026-01-01', 1, 1, 0, 100, 0, 100, 1, 2026)",
        "UPDATE daily_sales_summary SET net_revenue=0",
        "SELECT * FROM daily_sales_summary; DROP TABLE country_sales_summary;",
        "ATTACH DATABASE 'evil.db' AS evil",
    ]
    for q in malicious:
        valid, msg = validate_read_only_sql(q)
        assert not valid, f"Security failure: malicious query was allowed: {q}"

    # Safe SELECT queries must pass
    safe_q = "SELECT stock_code, net_sales FROM product_sales_summary ORDER BY net_sales DESC LIMIT 5"
    valid, msg = validate_read_only_sql(safe_q)
    assert valid, f"Safe query was rejected: {msg}"

    # Assistant intent checks
    assistant = GroundedAnalyticsAssistant()
    r1 = assistant.ask("What is total gross and net revenue?")
    assert r1["intent"] == "overall_kpis"
    assert "Net Revenue: £19,287,250.55" in r1["answer"]

    r2 = assistant.ask("What are the top 3 best seller products?")
    assert r2["intent"] == "top_products"
    assert len(r2["data"]) == 3

    logger.info("  AI Assistant Guardrails PASSED!")


if __name__ == "__main__":
    test_sql_reconciliation()
    test_ml_data_leakage_and_metrics()
    test_ai_assistant_guardrails()
    print("\nALL PHASE 3-6 TESTS PASSED SUCCESSFULLY!")
