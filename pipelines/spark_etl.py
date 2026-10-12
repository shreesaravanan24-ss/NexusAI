"""NexusAI Core PySpark Batch ETL Pipeline.

Transforms raw retail transactions into partitioned Silver and Gold Parquet layers.
Adheres strictly to data preservation:
- Preserves raw transaction records (no silent dropping of returns or manual price capping).
- Explicitly flags returns (is_return), guest users (is_guest_customer), and price anomalies.
- Calculates pure sales analytics (net_amount, gross_amount, return_amount).
- Produces clean Data Quality metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from pipelines.env_setup import get_spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NexusAI.SparkETL")


RAW_SCHEMA = StructType([
    StructField("Invoice", StringType(), True),
    StructField("StockCode", StringType(), True),
    StructField("Description", StringType(), True),
    # Quantity read as DoubleType because Calamine serialises Excel integers as floats
    # (e.g. "6.0"), which Spark IntegerType cannot parse. Cast to IntegerType in transform.
    StructField("Quantity", DoubleType(), True),
    StructField("InvoiceDate", TimestampType(), True),
    StructField("Price", DoubleType(), True),
    StructField("Customer ID", StringType(), True),
    StructField("Country", StringType(), True),
])


def load_raw_csv(spark: SparkSession, csv_path: Path) -> DataFrame:
    """Reads raw CSV using predefined schema without data loss."""
    logger.info("Loading raw CSV: %s", csv_path)
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .schema(RAW_SCHEMA)
        .load(str(csv_path))
    )


def transform_silver_layer(raw_df: DataFrame) -> DataFrame:
    """Cleans schema, normalizes column names, and adds business flags without discarding records."""
    logger.info("Transforming raw data to Silver layer...")

    silver_df = (
        raw_df
        .withColumnRenamed("Invoice", "invoice_no")
        .withColumnRenamed("StockCode", "stock_code")
        .withColumnRenamed("Description", "description")
        .withColumnRenamed("Quantity", "quantity")
        .withColumnRenamed("InvoiceDate", "invoice_timestamp")
        .withColumnRenamed("Price", "unit_price")
        .withColumnRenamed("Customer ID", "customer_id")
        .withColumnRenamed("Country", "country")
        # Cast quantity from Double (Calamine serialises Excel integers as floats) to Integer
        .withColumn("quantity", F.col("quantity").cast(IntegerType()))
        # Normalize strings and safely strip .0 from numeric stock codes (e.g. '22423.0' -> '22423')
        .withColumn("invoice_no", F.trim(F.col("invoice_no")))
        .withColumn(
            "stock_code",
            F.when(
                F.trim(F.col("stock_code")).endswith(".0"),
                F.regexp_replace(F.trim(F.col("stock_code")), "\\.0$", ""),
            ).otherwise(F.trim(F.col("stock_code"))),
        )
        .withColumn("description", F.trim(F.col("description")))
        .withColumn("country", F.trim(F.col("country")))
        .withColumn(
            "customer_id",
            F.when(F.trim(F.col("customer_id")) == "", None)
            .otherwise(F.trim(F.col("customer_id"))),
        )
        # Business and audit flags (Documented cleaning rules - NO SILENT DROPS)
        .withColumn(
            "is_return",
            (F.col("quantity") < 0)
            | F.col("invoice_no").startswith("C")
            | F.col("invoice_no").startswith("c"),
        )
        .withColumn("is_guest_customer", F.col("customer_id").isNull())
        .withColumn("is_zero_or_negative_price", F.coalesce(F.col("unit_price") <= 0.0, F.lit(False)))
        # Sales economics
        .withColumn(
            "gross_amount",
            F.when(F.col("quantity") > 0, F.round(F.col("quantity") * F.col("unit_price"), 2))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "return_amount",
            F.when(F.col("quantity") < 0, F.round(F.abs(F.col("quantity")) * F.col("unit_price"), 2))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "net_amount",
            F.round(F.col("quantity") * F.col("unit_price"), 2),
        )
        # Date partitions
        .withColumn("invoice_date", F.to_date(F.col("invoice_timestamp")))
        .withColumn("invoice_year", F.year(F.col("invoice_timestamp")))
        .withColumn("invoice_month", F.month(F.col("invoice_timestamp")))
    )
    return silver_df


def build_gold_daily_sales(silver_df: DataFrame) -> DataFrame:
    """Aggregates daily sales, returns, and distinct customer counts."""
    logger.info("Building Gold Layer: daily_sales_summary...")
    return (
        silver_df.filter(F.col("invoice_date").isNotNull())
        .groupBy("invoice_date", "invoice_year")
        .agg(
            F.countDistinct("invoice_no").alias("total_invoices"),
            F.sum(F.when(F.col("quantity") > 0, F.col("quantity")).otherwise(0)).alias("units_sold"),
            F.sum(F.when(F.col("quantity") < 0, F.abs(F.col("quantity"))).otherwise(0)).alias("units_returned"),
            F.round(F.sum("gross_amount"), 2).alias("gross_revenue"),
            F.round(F.sum("return_amount"), 2).alias("returns_loss"),
            F.round(F.sum("net_amount"), 2).alias("net_revenue"),
            F.countDistinct("customer_id").alias("unique_customers"),
        )
        .orderBy("invoice_date")
    )


def build_gold_product_sales(silver_df: DataFrame) -> DataFrame:
    """Aggregates product-level sales performance and return ratios."""
    logger.info("Building Gold Layer: product_sales_summary...")
    return (
        silver_df.filter(F.col("stock_code").isNotNull())
        .groupBy("stock_code")
        .agg(
            F.first("description", ignorenulls=True).alias("product_description"),
            F.countDistinct("invoice_no").alias("order_frequency"),
            F.sum(F.when(F.col("quantity") > 0, F.col("quantity")).otherwise(0)).alias("units_sold"),
            F.sum(F.when(F.col("quantity") < 0, F.abs(F.col("quantity"))).otherwise(0)).alias("units_returned"),
            F.round(F.sum("gross_amount"), 2).alias("gross_sales"),
            F.round(F.sum("net_amount"), 2).alias("net_sales"),
        )
        .withColumn(
            "return_rate_pct",
            F.when(F.col("units_sold") > 0, F.round((F.col("units_returned") / F.col("units_sold")) * 100, 2))
            .otherwise(F.lit(0.0)),
        )
        .orderBy(F.desc("net_sales"))
    )


def compute_data_quality_report(raw_df: DataFrame, silver_df: DataFrame) -> dict[str, Any]:
    """Generates an empirical audit report verifying data hygiene and integrity."""
    raw_count = raw_df.count()
    silver_count = silver_df.count()

    returns_count = silver_df.filter(F.col("is_return")).count()
    guest_count = silver_df.filter(F.col("is_guest_customer")).count()
    zero_price_count = silver_df.filter(F.col("is_zero_or_negative_price")).count()
    distinct_products = silver_df.select("stock_code").distinct().count()
    distinct_countries = silver_df.select("country").distinct().count()

    metrics = {
        "raw_record_count": raw_count,
        "silver_record_count": silver_count,
        "records_preserved": silver_count == raw_count,
        "returns_count": returns_count,
        "returns_percentage": round((returns_count / silver_count) * 100, 2) if silver_count else 0.0,
        "guest_customer_count": guest_count,
        "guest_percentage": round((guest_count / silver_count) * 100, 2) if silver_count else 0.0,
        "price_adjustments_or_zero_count": zero_price_count,
        "distinct_products": distinct_products,
        "distinct_countries": distinct_countries,
    }
    return metrics


def run_pipeline(
    input_csv: Path,
    silver_out: Path,
    gold_daily_out: Path,
    gold_product_out: Path,
    spark: SparkSession | None = None,
) -> dict[str, Any]:
    """Executes the full local Parquet ETL pipeline."""
    managed_session = spark is None
    if spark is None:
        spark = get_spark_session("NexusAI-BatchPipeline")

    try:
        raw_df = load_raw_csv(spark, input_csv)
        silver_df = transform_silver_layer(raw_df)

        # Compute Data Quality metrics before caching or writing
        dq_report = compute_data_quality_report(raw_df, silver_df)
        logger.info("Data Quality Audit Report: %s", dq_report)

        # Write Silver Layer partitioned by invoice_year
        logger.info("Writing Silver layer Parquet to: %s", silver_out)
        (
            silver_df.write.mode("overwrite")
            .partitionBy("invoice_year")
            .parquet(str(silver_out))
        )

        # Build & Write Gold Daily Sales partitioned by invoice_year
        daily_sales = build_gold_daily_sales(silver_df)
        logger.info("Writing Gold Daily Sales Parquet to: %s", gold_daily_out)
        (
            daily_sales.write.mode("overwrite")
            .partitionBy("invoice_year")
            .parquet(str(gold_daily_out))
        )

        # Build & Write Gold Product Sales
        product_sales = build_gold_product_sales(silver_df)
        logger.info("Writing Gold Product Sales Parquet to: %s", gold_product_out)
        (
            product_sales.write.mode("overwrite")
            .parquet(str(gold_product_out))
        )

        logger.info("ETL Pipeline completed successfully.")
        return dq_report
    finally:
        if managed_session:
            spark.stop()


if __name__ == "__main__":
    import argparse

    base = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="NexusAI Spark ETL Pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=base / "data" / "raw" / "sample_test.csv",
        help="Path to raw input CSV (default: data/raw/sample_test.csv)",
    )
    parser.add_argument(
        "--silver-out",
        type=Path,
        default=base / "data" / "silver" / "retail_transactions",
        help="Output path for Silver Parquet layer",
    )
    parser.add_argument(
        "--gold-daily-out",
        type=Path,
        default=base / "data" / "gold" / "daily_sales_summary",
        help="Output path for Gold daily sales Parquet",
    )
    parser.add_argument(
        "--gold-product-out",
        type=Path,
        default=base / "data" / "gold" / "product_sales_summary",
        help="Output path for Gold product sales Parquet",
    )
    args = parser.parse_args()

    logger.info("Input CSV     : %s", args.input)
    logger.info("Silver output : %s", args.silver_out)
    logger.info("Gold daily    : %s", args.gold_daily_out)
    logger.info("Gold product  : %s", args.gold_product_out)

    report = run_pipeline(args.input, args.silver_out, args.gold_daily_out, args.gold_product_out)
    print("\n--- Pipeline Data Quality Audit Report ---")
    for k, v in report.items():
        print(f"  {k}: {v}")
