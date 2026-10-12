-- Schema definition for NexusAI SQLite Analytical Store

CREATE TABLE IF NOT EXISTS daily_sales_summary (
    invoice_date TEXT PRIMARY KEY,
    total_invoices INTEGER,
    units_sold INTEGER,
    units_returned INTEGER,
    gross_revenue REAL,
    returns_loss REAL,
    net_revenue REAL,
    unique_customers INTEGER,
    invoice_year INTEGER
);

CREATE TABLE IF NOT EXISTS product_sales_summary (
    stock_code TEXT PRIMARY KEY,
    product_description TEXT,
    order_frequency INTEGER,
    units_sold INTEGER,
    units_returned INTEGER,
    gross_sales REAL,
    net_sales REAL,
    return_rate_pct REAL
);

CREATE TABLE IF NOT EXISTS country_sales_summary (
    country TEXT PRIMARY KEY,
    total_invoices INTEGER,
    unique_customers INTEGER,
    units_sold INTEGER,
    units_returned INTEGER,
    gross_revenue REAL,
    returns_loss REAL,
    net_revenue REAL
);

CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_sales_summary(invoice_date);
CREATE INDEX IF NOT EXISTS idx_product_stock ON product_sales_summary(stock_code);
CREATE INDEX IF NOT EXISTS idx_country_name ON country_sales_summary(country);
