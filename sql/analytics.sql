-- Example PostgreSQL schema for a normalized transaction table.
CREATE TABLE IF NOT EXISTS retail_transactions (
    invoice_no TEXT,
    stock_code TEXT,
    description TEXT,
    quantity INTEGER,
    invoice_date TIMESTAMP,
    unit_price NUMERIC(12, 4),
    customer_id TEXT,
    country TEXT,
    revenue NUMERIC(14, 4),
    is_cancellation BOOLEAN
);

-- Once data has been loaded into retail_transactions:
SELECT country, ROUND(SUM(revenue), 2) AS revenue
FROM retail_transactions
WHERE NOT is_cancellation AND quantity > 0 AND unit_price > 0
GROUP BY country
ORDER BY revenue DESC
LIMIT 10;

SELECT customer_id,
       COUNT(DISTINCT invoice_no) AS order_count,
       ROUND(SUM(revenue), 2) AS total_spend,
       MAX(invoice_date) AS last_purchase
FROM retail_transactions
WHERE customer_id IS NOT NULL
  AND NOT is_cancellation
  AND quantity > 0
  AND unit_price > 0
GROUP BY customer_id
ORDER BY total_spend DESC
LIMIT 20;
