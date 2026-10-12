"""Dataset Ingestion & Extraction for Online Retail II (UCI ID 502).

Verified Source: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
import urllib.request
import zipfile
from python_calamine import CalamineWorkbook

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NexusAI.FetchDataset")

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def get_data_dirs() -> tuple[Path, Path, Path]:
    base = Path(__file__).resolve().parent.parent
    raw_dir = base / "data" / "raw"
    silver_dir = base / "data" / "silver"
    gold_dir = base / "data" / "gold"
    raw_dir.mkdir(parents=True, exist_ok=True)
    silver_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, silver_dir, gold_dir


def download_uci_dataset(dest_zip: Path) -> Path:
    """Downloads the official Online Retail II archive if not present."""
    if dest_zip.exists() and dest_zip.stat().st_size > 1000:
        logger.info("Dataset archive already exists at: %s", dest_zip)
        return dest_zip

    logger.info("Downloading Online Retail II from %s ...", UCI_DATASET_URL)
    req = urllib.request.Request(UCI_DATASET_URL, headers={"User-Agent": "Mozilla/5.0 (NexusAI/1.0)"})
    with urllib.request.urlopen(req) as resp, open(dest_zip, "wb") as f:
        # Download in chunks with progress
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                logger.info("Progress: %.1f MB / %.1f MB", downloaded / (1024 * 1024), total / (1024 * 1024))
    logger.info("Download complete: %s (%d bytes)", dest_zip, dest_zip.stat().st_size)
    return dest_zip


def extract_and_convert_to_csv(
    zip_path: Path,
    output_csv: Path,
    sample_limit: int | None = None,
) -> Path:
    """Extracts Excel workbook from zip and streams rows into CSV using Calamine."""
    if output_csv.exists() and output_csv.stat().st_size > 0 and sample_limit is None:
        logger.info("Raw CSV already exists at: %s", output_csv)
        return output_csv

    # Extract xlsx to raw dir
    raw_dir = zip_path.parent
    xlsx_path = raw_dir / "online_retail_II.xlsx"
    if not xlsx_path.exists():
        logger.info("Extracting %s from zip...", xlsx_path.name)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw_dir)

    logger.info("Reading Excel sheets with Calamine and streaming to CSV: %s", output_csv)
    wb = CalamineWorkbook.from_path(str(xlsx_path))
    headers = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]

    rows_written = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(headers)

        for sheet_name in wb.sheet_names:
            logger.info("Processing worksheet: %s", sheet_name)
            sheet = wb.get_sheet_by_name(sheet_name)
            sheet_rows = sheet.to_python()
            if not sheet_rows:
                continue

            # First row is header in each sheet
            for row in sheet_rows[1:]:
                if sample_limit is not None and rows_written >= sample_limit:
                    break
                # Normalize: cast Quantity (col 3) to int — Calamine returns Excel numbers as floats
                row_out = list(row[:8])
                if row_out[3] is not None:
                    try:
                        row_out[3] = int(row_out[3])
                    except (ValueError, TypeError):
                        pass
                writer.writerow(row_out)
                rows_written += 1

            if sample_limit is not None and rows_written >= sample_limit:
                break

    logger.info("Completed CSV conversion. Total records written: %d", rows_written)
    return output_csv


def generate_unit_test_sample(sample_csv_path: Path) -> Path:
    """Generates a small, deterministic test fixture for testing ETL rules and edge cases."""
    records = [
        # Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
        ["536365", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", "6", "2010-12-01 08:26:00", "2.55", "17850", "United Kingdom"],
        ["536365", "71053", "WHITE METAL LANTERN", "6", "2010-12-01 08:26:00", "3.39", "17850", "United Kingdom"],
        # Cancellation/Return: negative quantity & 'C' prefix
        ["C536379", "D", "Discount", "-1", "2010-12-01 09:41:00", "27.50", "14527", "United Kingdom"],
        # Negative quantity without C prefix (damaged/return record)
        ["536380", "84406B", "CREAM CUPID HEARTS COAT HANGER", "-2", "2010-12-01 09:45:00", "3.20", "15311", "United Kingdom"],
        # Guest customer (null customer ID)
        ["536381", "22423", "REGENCY CAKESTAND 3 TIER", "3", "2010-12-01 10:00:00", "12.75", "", "France"],
        # Zero price / adjustment record
        ["536382", "POST", "POSTAGE", "1", "2010-12-01 10:15:00", "0.00", "12431", "Germany"],
        # Another standard sale in Germany
        ["536383", "22728", "ALARM CLOCK BAKELIKE PINK", "12", "2010-12-01 11:30:00", "3.75", "12431", "Germany"],
    ]

    headers = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    with open(sample_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(records)

    logger.info("Generated unit test sample at: %s", sample_csv_path)
    return sample_csv_path


if __name__ == "__main__":
    raw_dir, _, _ = get_data_dirs()
    zip_dest = raw_dir / "online_retail_ii.zip"
    csv_dest = raw_dir / "online_retail_raw.csv"

    download_uci_dataset(zip_dest)
    extract_and_convert_to_csv(zip_dest, csv_dest)
    generate_unit_test_sample(raw_dir / "sample_test.csv")
