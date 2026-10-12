-- Analytical views and queries for NexusAI

-- 1. Monthly Revenue & Return Performance
SELECT
    SUBSTR(invoice_date, 1, 7) AS year_month,
    SUM(total_invoices) AS total_invoices,
    SUM(units_sold) AS units_sold,
    SUM(units_returned) AS units_returned,
    ROUND(SUM(gross_revenue), 2) AS gross_revenue,
    ROUND(SUM(returns_loss), 2) AS returns_loss,
    ROUND(SUM(net_revenue), 2) AS net_revenue,
    ROUND(SUM(returns_loss) / SUM(gross_revenue) * 100, 2) AS return_rate_pct
FROM daily_sales_summary
GROUP BY SUBSTR(invoice_date, 1, 7)
ORDER BY year_month ASC;

-- 2. Top 10 Products by Net Revenue
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
LIMIT 10;

-- 3. Top 10 Countries by Net Sales
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
LIMIT 10;
