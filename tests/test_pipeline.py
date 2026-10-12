"""Automated Test Suite for NexusAI Ingestion & Data Quality Milestone."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipelines.env_setup import get_spark_session
from pipelines.fetch_dataset import generate_unit_test_sample
from pipelines.spark_etl import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NexusAI.Test")


def run_test_suite() -> bool:
    """Executes deterministic test on the pipeline and verifies all invariants."""
    raw_dir = BASE_DIR / "data" / "raw"
    silver_dir = BASE_DIR / "data" / "silver" / "test_retail_transactions"
    gold_daily_dir = BASE_DIR / "data" / "gold" / "test_daily_sales_summary"
    gold_product_dir = BASE_DIR / "data" / "gold" / "test_product_sales_summary"

    sample_csv = raw_dir / "sample_test.csv"
    generate_unit_test_sample(sample_csv)

    spark = get_spark_session("NexusAI-UnitTest")

    try:
        logger.info("Executing pipeline on unit test sample...")
        report = run_pipeline(
            input_csv=sample_csv,
            silver_out=silver_dir,
            gold_daily_out=gold_daily_dir,
            gold_product_out=gold_product_dir,
            spark=spark,
        )

        logger.info("Verifying Data Quality Invariants...")
        assert report["raw_record_count"] == 7, f"Expected 7 raw records, got {report['raw_record_count']}"
        assert report["silver_record_count"] == 7, f"Expected 7 silver records, got {report['silver_record_count']}"
        assert report["records_preserved"] is True, "Data loss detected! Silver count must equal Raw count."
        assert report["returns_count"] == 2, f"Expected 2 return records, got {report['returns_count']}"
        assert report["guest_customer_count"] == 1, f"Expected 1 guest customer, got {report['guest_customer_count']}"
        assert report["price_adjustments_or_zero_count"] == 1, f"Expected 1 zero price item, got {report['price_adjustments_or_zero_count']}"

        # Verify Silver Parquet output
        silver_read = spark.read.parquet(str(silver_dir))
        assert silver_read.count() == 7, "Silver Parquet on disk does not match expected row count."

        # Verify Gold Daily Sales Parquet output
        gold_daily_read = spark.read.parquet(str(gold_daily_dir))
        daily_count = gold_daily_read.count()
        assert daily_count == 1, f"Expected 1 day in test aggregation, got {daily_count}"
        daily_row = gold_daily_read.first()
        logger.info("Gold Daily Summary: Gross=%.2f, Returns=%.2f, Net=%.2f",
                    daily_row["gross_revenue"], daily_row["returns_loss"], daily_row["net_revenue"])
        assert daily_row["gross_revenue"] > 0, "Gross revenue should be positive."
        assert daily_row["returns_loss"] > 0, "Returns loss should be positive for return entries."
        assert round(daily_row["gross_revenue"] - daily_row["returns_loss"], 2) == round(daily_row["net_revenue"], 2), \
            "Accounting invariant violated: Net revenue must equal Gross revenue minus Returns loss."

        # Verify Gold Product Sales Parquet output
        gold_prod_read = spark.read.parquet(str(gold_product_dir))
        assert gold_prod_read.count() > 0, "Gold Product Parquet has no records."

        logger.info("ALL INVARIANTS AND PARQUET PERSISTENCE TESTS PASSED!")
        return True

    finally:
        spark.stop()


if __name__ == "__main__":
    success = run_test_suite()
    if not success:
        sys.exit(1)
