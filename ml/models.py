"""NexusAI Machine Learning Module.

Practical Sales Forecasting and Transaction Anomaly Detection:
1. Daily Net Sales Forecasting: Ridge / Linear regression with chronological split
   (train on 80% earlier days, test on final 20% unseen days, evaluated against naive baseline).
2. Transaction Anomaly Detection: Isolation Forest identifying statistical outliers
   (high unit price, extreme return sizes, large quantities).
   Clearly documented as statistical anomalies, NOT confirmed fraud or stockouts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NexusAI.ML")

BASE_DIR = Path(__file__).resolve().parent.parent


def prepare_forecasting_data(daily_parquet_path: Path | str = BASE_DIR / "data" / "gold" / "daily_sales_summary") -> pd.DataFrame:
    """Reads daily gold aggregates and builds lag and calendar features."""
    df = pq.read_table(str(daily_parquet_path)).to_pandas()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df = df.sort_values("invoice_date").reset_index(drop=True)

    # Feature engineering using only past data (no future leakage)
    df["day_of_week"] = df["invoice_date"].dt.dayofweek
    df["day_of_month"] = df["invoice_date"].dt.day
    df["month"] = df["invoice_date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Lags of net_revenue ( strictly t-1, t-2, t-7, t-14 )
    df["lag_1"] = df["net_revenue"].shift(1)
    df["lag_2"] = df["net_revenue"].shift(2)
    df["lag_7"] = df["net_revenue"].shift(7)
    df["lag_14"] = df["net_revenue"].shift(14)
    # Rolling past 7-day average (excluding current day: shift(1).rolling(7))
    df["rolling_mean_7"] = df["net_revenue"].shift(1).rolling(7).mean()

    # Drop initial rows with NaN from lags
    df = df.dropna().reset_index(drop=True)
    return df


def train_and_evaluate_forecasting(
    daily_parquet_path: Path | str = BASE_DIR / "data" / "gold" / "daily_sales_summary",
    test_ratio: float = 0.20,
) -> dict[str, Any]:
    """Chronological train/test split forecasting model evaluated against baseline.

    Strictly avoids data leakage by cutting time series chronologically.
    """
    df = prepare_forecasting_data(daily_parquet_path)
    n = len(df)
    split_idx = int(n * (1.0 - test_ratio))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    features = ["day_of_week", "day_of_month", "month", "is_weekend", "lag_1", "lag_2", "lag_7", "lag_14", "rolling_mean_7"]
    target = "net_revenue"

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # Model: Ridge Regression
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Baseline: Naive persistence (predict lag_1, i.e. yesterday's net revenue)
    y_baseline = test_df["lag_1"].values

    mae_model = mean_absolute_error(y_test, y_pred)
    rmse_model = root_mean_squared_error(y_test, y_pred)
    mae_base = mean_absolute_error(y_test, y_baseline)
    rmse_base = root_mean_squared_error(y_test, y_baseline)

    # Comparison
    mae_improvement_pct = round(((mae_base - mae_model) / mae_base) * 100, 2)

    result = {
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_date_range": (train_df["invoice_date"].min().strftime("%Y-%m-%d"), train_df["invoice_date"].max().strftime("%Y-%m-%d")),
        "test_date_range": (test_df["invoice_date"].min().strftime("%Y-%m-%d"), test_df["invoice_date"].max().strftime("%Y-%m-%d")),
        "model_mae": round(float(mae_model), 2),
        "model_rmse": round(float(rmse_model), 2),
        "baseline_mae": round(float(mae_base), 2),
        "baseline_rmse": round(float(rmse_base), 2),
        "mae_improvement_pct": mae_improvement_pct,
        "test_predictions": pd.DataFrame({
            "date": test_df["invoice_date"].dt.strftime("%Y-%m-%d"),
            "actual": y_test.values,
            "predicted": y_pred,
            "baseline": y_baseline,
        }),
    }
    logger.info("Forecasting Evaluation: Model MAE=£%.2f vs Baseline MAE=£%.2f (%.2f%% improvement)",
                mae_model, mae_base, mae_improvement_pct)
    return result


def detect_transaction_anomalies(
    silver_parquet_path: Path | str = BASE_DIR / "data" / "silver" / "retail_transactions",
    contamination: float = 0.005,
    sample_size: int = 50000,
) -> pd.DataFrame:
    """Detects statistical outliers using Isolation Forest on transaction metrics.

    NOTE: Statistical outliers represent unusual values (e.g. bulk orders, manual credit memos,
    atypical prices) and DO NOT constitute confirmed fraud or physical stockout events.
    """
    logger.info("Sampling %d records from Silver for anomaly detection...", sample_size)
    table = pq.read_table(
        str(silver_parquet_path),
        columns=["invoice_no", "stock_code", "description", "quantity", "unit_price", "net_amount", "country", "is_return"],
    )
    df = table.to_pandas()
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42).copy()
    else:
        df_sample = df.copy()

    # Features: quantity magnitude, unit price, net amount magnitude
    features = pd.DataFrame({
        "abs_quantity": df_sample["quantity"].abs(),
        "unit_price": df_sample["unit_price"].fillna(0.0),
        "abs_net_amount": df_sample["net_amount"].abs().fillna(0.0),
        "is_return_num": df_sample["is_return"].astype(int),
    })

    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    df_sample["anomaly_score"] = iso.fit_predict(features)
    # -1 is anomaly, 1 is inlier
    anomalies = df_sample[df_sample["anomaly_score"] == -1].copy()
    anomalies = anomalies.sort_values("net_amount", key=abs, ascending=False)
    logger.info("Detected %d anomalies out of %d sampled transactions (%.2f%%)",
                len(anomalies), len(df_sample), (len(anomalies) / len(df_sample)) * 100)
    return anomalies


if __name__ == "__main__":
    fc = train_and_evaluate_forecasting()
    print("Forecasting results:")
    for k, v in fc.items():
        if k != "test_predictions":
            print(f"  {k}: {v}")
    anom = detect_transaction_anomalies(sample_size=10000)
    print("\nTop 5 transaction anomalies detected:\n", anom[["invoice_no", "stock_code", "quantity", "unit_price", "net_amount"]].head(5))
