#!/usr/bin/env python3
"""Print every output block used in Chapter 13, from shop.db.

Run from the project root:

    python scripts/chapter13_outputs.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "raw" / "shop.db"
CON = sqlite3.connect(DB)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 80)


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def q(sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, CON)


def scalar(sql: str):
    return q(sql).iloc[0, 0]


# ============================================ 13.1 relational thinking
def section_131() -> None:
    show("R1 sqlite version", {"sqlite": sqlite3.sqlite_version})
    show("R2 tables", q("""
        SELECT name, type FROM sqlite_master
        WHERE type IN ('table','view') ORDER BY name
    """).to_string(index=False))

    show("R3 row counts", q("""
        SELECT 'customers' AS t, COUNT(*) AS n FROM customers
        UNION ALL SELECT 'orders', COUNT(*) FROM orders
        UNION ALL SELECT 'order_items', COUNT(*) FROM order_items
        UNION ALL SELECT 'products', COUNT(*) FROM products
        UNION ALL SELECT 'targets', COUNT(*) FROM targets
    """).to_string(index=False))

    show("R4 primary keys", q("""
        SELECT
          (SELECT COUNT(*) FROM customers) -
          (SELECT COUNT(DISTINCT customer_id) FROM customers) AS customer_dupes,
          (SELECT COUNT(*) FROM orders) -
          (SELECT COUNT(DISTINCT order_id) FROM orders) AS order_dupes,
          (SELECT COUNT(*) FROM order_items) -
          (SELECT COUNT(DISTINCT order_id || '-' || product_id)
           FROM order_items) AS line_dupes
    """).to_string(index=False))

    show("R5 dialect notes", [
        "SQLite: ILIKE unsupported, use LOWER + LIKE",
        "SQLite: DATE_TRUNC unsupported, use strftime",
        "SQLite: RIGHT/FULL OUTER supported from 3.39",
        "PostgreSQL: DATE_TRUNC, ILIKE, GENERATED columns",
        "BigQuery: backticks, STRUCT, partition clustering",
    ])


# =============================================== 13.2 query mechanics
def section_132() -> None:
    show("Q1 select and alias", q("""
        SELECT order_id AS id, status AS state
        FROM orders LIMIT 4
    """).to_string(index=False))

    show("Q2 where before group by", q("""
        SELECT status, COUNT(*) AS n
        FROM orders
        WHERE order_date >= '2025-01-01'
        GROUP BY status
    """).to_string(index=False))

    show("Q3 having filters groups", q("""
        SELECT channel, COUNT(*) AS orders
        FROM orders
        GROUP BY channel
        HAVING COUNT(*) > 1000
    """).to_string(index=False))

    show("Q4 case expression", q("""
        SELECT order_id,
               CASE WHEN status = 'shipped' THEN 'good'
                    WHEN status = 'returned' THEN 'reverse'
                    ELSE 'excluded' END AS bucket
        FROM orders LIMIT 5
    """).to_string(index=False))

    show("Q5 coalesce and nullif", q("""
        SELECT COALESCE(NULL, 'fallback') AS filled,
               NULLIF(5, 5) AS nulled,
               COALESCE(NULLIF(0, 0), 99) AS both
    """).to_string(index=False))

    show("Q6 cast", q("""
        SELECT CAST('42' AS INTEGER) + 1 AS i,
               CAST('3.5' AS REAL) * 2 AS r
    """).to_string(index=False))


# ============================================= 13.3 filtering/sorting
def section_133() -> None:
    show("F1 in and between", q("""
        SELECT COUNT(*) AS n FROM orders
        WHERE channel IN ('web','app')
          AND order_date BETWEEN '2025-01-01' AND '2025-03-31'
    """).to_string(index=False))

    show("F2 like and lower", q("""
        SELECT campaign_name FROM campaigns
        WHERE LOWER(campaign_name) LIKE '%sale%'
    """).to_string(index=False))

    show("F3 NULL comparison trap", q("""
        SELECT
          (SELECT COUNT(*) FROM staging WHERE cust_key = NULL) AS equals_null,
          (SELECT COUNT(*) FROM staging WHERE cust_key IS NULL) AS is_null,
          (SELECT COUNT(*) FROM staging
            WHERE cust_key NOT IN ('C-00001')) AS not_in_with_null
    """).to_string(index=False))

    show("F4 order by multiple keys", q("""
        SELECT region, segment, COUNT(*) AS n
        FROM customers GROUP BY region, segment
        ORDER BY region, n DESC LIMIT 6
    """).to_string(index=False))

    show("F5 nulls sort first", q("""
        SELECT order_ref, cust_key FROM staging
        ORDER BY cust_key ASC
    """).to_string(index=False))

    show("F6 deterministic limit", q("""
        SELECT order_id, order_date FROM orders
        ORDER BY order_date DESC, order_id ASC LIMIT 4
    """).to_string(index=False))


# ========================================== 13.4 aggregation/grouping
def section_134() -> None:
    show("A1 count star vs col", q("""
        SELECT COUNT(*) AS all_rows,
               COUNT(cust_key) AS non_null_keys,
               COUNT(DISTINCT cust_key) AS distinct_keys
        FROM staging
    """).to_string(index=False))

    show("A2 sums and averages", q("""
        SELECT status,
               COUNT(*) AS lines,
               ROUND(SUM(quantity * unit_price *
                 (1 - discount_pct/100.0)), 2) AS revenue
        FROM order_items i JOIN orders o USING (order_id)
        GROUP BY status
    """).to_string(index=False))

    show("A3 integer division", q("""
        SELECT 7 / 2 AS int_div, 7 / 2.0 AS real_div
    """).to_string(index=False))

    show("A4 group by two keys", q("""
        SELECT region, segment, COUNT(*) AS customers
        FROM customers GROUP BY region, segment
        ORDER BY region, segment LIMIT 8
    """).to_string(index=False))

    show("A5 rollup emulated", q("""
        SELECT region, SUM(n) AS customers FROM (
          SELECT region, segment, COUNT(*) AS n FROM customers
          GROUP BY region, segment
          UNION ALL
          SELECT 'ALL', NULL, COUNT(*) FROM customers
        ) GROUP BY region ORDER BY region
    """).to_string(index=False))

    show("A6 avg ignores nulls", q("""
        SELECT AVG(amount_num) AS avg_amount FROM (
          SELECT CASE WHEN amount = 'N/A' THEN NULL
                      ELSE CAST(REPLACE(amount, '$','') AS REAL) END AS amount_num
          FROM staging)
    """).to_string(index=False))

    show("A7 min and max", q("""
        SELECT MIN(order_date) AS first_order,
               MAX(order_date) AS last_order
        FROM orders
    """).to_string(index=False))


# ======================================================== 13.5 joins
def section_135() -> None:
    show("J1 inner join", q("""
        SELECT COUNT(*) AS rows_joined
        FROM orders o JOIN customers c ON o.customer_id = c.customer_id
    """).to_string(index=False))

    show("J2 left join unmatched", q("""
        SELECT COUNT(*) AS rows_joined,
               SUM(CASE WHEN c.customer_id IS NULL THEN 1 ELSE 0 END) AS unmatched
        FROM orders o LEFT JOIN customers c
          ON o.customer_id = c.customer_id
    """).to_string(index=False))

    show("J3 fan-out from duplicate dimension", q("""
        SELECT COUNT(*) AS joined_rows
        FROM orders o JOIN customers_dup c
          ON o.customer_id = c.customer_id
    """).to_string(index=False))

    show("J4 fan-out revenue", q("""
        SELECT ROUND(SUM(i.quantity * i.unit_price *
                 (1 - i.discount_pct/100.0)), 2) AS gross
        FROM orders o JOIN customers_dup c
          ON o.customer_id = c.customer_id
        JOIN order_items i ON o.order_id = i.order_id
    """).to_string(index=False))

    show("J5 correct revenue", q("""
        SELECT ROUND(SUM(i.quantity * i.unit_price *
                 (1 - i.discount_pct/100.0)), 2) AS gross
        FROM orders o JOIN customers c
          ON o.customer_id = c.customer_id
        JOIN order_items i ON o.order_id = i.order_id
    """).to_string(index=False))

    show("J6 anti join", q("""
        SELECT COUNT(*) AS customers_without_orders
        FROM customers c
        WHERE NOT EXISTS (SELECT 1 FROM orders o
                          WHERE o.customer_id = c.customer_id)
    """).to_string(index=False))

    show("J7 semi join", q("""
        SELECT COUNT(*) AS customers_with_orders
        FROM customers c
        WHERE EXISTS (SELECT 1 FROM orders o
                      WHERE o.customer_id = c.customer_id)
    """).to_string(index=False))

    show("J8 cross join size", q("""
        SELECT COUNT(*) AS pairs FROM
          (SELECT 1 UNION SELECT 2) a CROSS JOIN
          (SELECT 1 UNION SELECT 2 UNION SELECT 3) b
    """).to_string(index=False))


# ============================================== 13.6 subqueries and CTEs
def section_136() -> None:
    show("CT1 scalar subquery", q("""
        SELECT (SELECT COUNT(*) FROM orders) AS orders,
               (SELECT COUNT(*) FROM customers) AS customers
    """).to_string(index=False))

    show("CT2 derived table", q("""
        SELECT status, MAX(n) AS largest FROM (
          SELECT status, COUNT(*) AS n FROM orders GROUP BY status
        ) GROUP BY status
    """).to_string(index=False))

    show("CT3 correlated subquery", q("""
        SELECT c.customer_id,
               (SELECT COUNT(*) FROM orders o
                WHERE o.customer_id = c.customer_id) AS n_orders
        FROM customers c LIMIT 4
    """).to_string(index=False))

    show("CT4 chained CTEs", q("""
        WITH per_month AS (
          SELECT substr(order_date,1,7) AS month,
                 status, COUNT(*) AS n
          FROM orders GROUP BY 1,2
        ), good AS (
          SELECT month, SUM(n) AS orders FROM per_month
          WHERE status = 'shipped' GROUP BY month
        )
        SELECT * FROM good ORDER BY month LIMIT 4
    """).to_string(index=False))

    show("CT5 recursive CTE", q("""
        WITH RECURSIVE seq(n) AS (
          SELECT 1 UNION ALL SELECT n+1 FROM seq WHERE n < 5
        ) SELECT * FROM seq
    """).to_string(index=False))

    show("CT6 view", q("""
        SELECT COUNT(*) AS view_rows FROM v_order_lines
    """).to_string(index=False))


# ============================================== 13.7 window functions
def section_137() -> None:
    show("W1 row_number", q("""
        SELECT customer_id, order_id, order_date,
               ROW_NUMBER() OVER (PARTITION BY customer_id
                                  ORDER BY order_date) AS rn
        FROM orders
        WHERE customer_id IN (SELECT customer_id FROM customers LIMIT 2)
        ORDER BY customer_id, rn LIMIT 6
    """).to_string(index=False))

    show("W2 rank and dense_rank", q("""
        SELECT revenue,
               RANK() OVER (ORDER BY revenue DESC) AS rk,
               DENSE_RANK() OVER (ORDER BY revenue DESC) AS dr
        FROM (SELECT ROUND(SUM(quantity*unit_price),0) AS revenue
              FROM order_items GROUP BY order_id)
        LIMIT 5
    """).to_string(index=False))

    show("W3 lag", q("""
        SELECT month, orders,
               LAG(orders) OVER (ORDER BY month) AS prev_month
        FROM (SELECT substr(order_date,1,7) AS month, COUNT(*) AS orders
              FROM orders GROUP BY 1)
        ORDER BY month LIMIT 4
    """).to_string(index=False))

    show("W4 running total", q("""
        SELECT month, orders,
               SUM(orders) OVER (ORDER BY month
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum
        FROM (SELECT substr(order_date,1,7) AS month, COUNT(*) AS orders
              FROM orders GROUP BY 1)
        ORDER BY month LIMIT 4
    """).to_string(index=False))

    show("W5 moving average", q("""
        SELECT month, orders,
               ROUND(AVG(orders) OVER (ORDER BY month
                     ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 1) AS ma3
        FROM (SELECT substr(order_date,1,7) AS month, COUNT(*) AS orders
              FROM orders GROUP BY 1)
        ORDER BY month LIMIT 5
    """).to_string(index=False))

    show("W6 cumulative share", q("""
        SELECT region, customers,
               ROUND(1.0 * customers / SUM(customers) OVER (), 4) AS share
        FROM (SELECT region, COUNT(*) AS customers
              FROM customers GROUP BY region)
        ORDER BY customers DESC
    """).to_string(index=False))

    show("W7 ntile", q("""
        SELECT customer_id, n_orders,
               NTILE(4) OVER (ORDER BY n_orders DESC) AS quartile
        FROM (SELECT customer_id, COUNT(*) AS n_orders
              FROM orders GROUP BY customer_id)
        LIMIT 5
    """).to_string(index=False))

    show("W8 first_value", q("""
        SELECT order_id, order_date,
               FIRST_VALUE(order_date) OVER (
                 PARTITION BY customer_id ORDER BY order_date
                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS first_date
        FROM orders LIMIT 4
    """).to_string(index=False))


# ============================================= 13.8 analytical patterns
def section_138() -> None:
    show("P1 dedupe with row_number", q("""
        WITH ranked AS (
          SELECT *, ROW_NUMBER() OVER (
            PARTITION BY customer_id ORDER BY customer_id) AS rn
          FROM customers_dup
        ) SELECT COUNT(*) AS kept FROM ranked WHERE rn = 1
    """).to_string(index=False))

    show("P2 top three per group", q("""
        WITH line_rev AS (
          SELECT i.product_id, p.category,
                 SUM(i.quantity*i.unit_price) AS revenue,
                 ROW_NUMBER() OVER (PARTITION BY p.category
                                    ORDER BY SUM(i.quantity*i.unit_price) DESC) AS rn
          FROM order_items i JOIN products p USING (product_id)
          GROUP BY i.product_id, p.category
        ) SELECT category, product_id, ROUND(revenue,2) AS revenue
          FROM line_rev WHERE rn <= 3 ORDER BY category, revenue DESC LIMIT 8
    """).to_string(index=False))

    show("P3 month over month", q("""
        WITH m AS (
          SELECT substr(order_date,1,7) AS month, COUNT(*) AS orders
          FROM orders GROUP BY 1
        ) SELECT month, orders,
                 LAG(orders) OVER (ORDER BY month) AS prev,
                 ROUND(1.0*orders/LAG(orders) OVER (ORDER BY month)-1,4) AS mom
          FROM m ORDER BY month LIMIT 4
    """).to_string(index=False))

    show("P4 year over year", q("""
        WITH m AS (
          SELECT substr(order_date,1,4) AS yr,
                 substr(order_date,6,2) AS mo, COUNT(*) AS orders
          FROM orders GROUP BY 1,2
        ), yoy AS (
          SELECT yr, mo, orders,
                 LAG(orders,12) OVER (ORDER BY yr, mo) AS same_month_last_year
          FROM m
        )
        SELECT * FROM yoy WHERE yr = '2025' ORDER BY mo LIMIT 4
    """).to_string(index=False))

    show("P5 cohort sizes", q("""
        SELECT substr(signup_date,1,7) AS cohort, COUNT(*) AS customers
        FROM customers GROUP BY 1 ORDER BY 1 LIMIT 5
    """).to_string(index=False))

    show("P6 funnel", q("""
        SELECT step, COUNT(DISTINCT user_id) AS users
        FROM funnel_events GROUP BY step
        ORDER BY CASE step WHEN 'visit' THEN 1 WHEN 'product' THEN 2
                           WHEN 'cart' THEN 3 WHEN 'checkout' THEN 4
                           ELSE 5 END
    """).to_string(index=False))

    show("P7 gaps and islands", q("""
        WITH days AS (
          SELECT DISTINCT order_date AS d FROM orders
          WHERE customer_id = 'C-00001'
        ), numbered AS (
          SELECT d, julianday(d) - ROW_NUMBER() OVER (ORDER BY d) AS grp
          FROM days
        ) SELECT MIN(d) AS start, MAX(d) AS finish, COUNT(*) AS run_length
          FROM numbered GROUP BY grp ORDER BY start LIMIT 4
    """).to_string(index=False))

    show("P8 period to date", q("""
        WITH m AS (
          SELECT substr(order_date,1,7) AS month, COUNT(*) AS orders
          FROM orders GROUP BY 1
        ) SELECT month, orders,
                 SUM(orders) OVER (PARTITION BY substr(month,1,4)
                     ORDER BY month ROWS UNBOUNDED PRECEDING) AS ytd
          FROM m ORDER BY month LIMIT 5
    """).to_string(index=False))


# ========================================== 13.9 dates, strings, NULLs
def section_139() -> None:
    show("DT1 extraction", q("""
        SELECT strftime('%Y', order_date) AS yr,
               strftime('%m', order_date) AS mo,
               strftime('%w', order_date) AS dow
        FROM orders LIMIT 3
    """).to_string(index=False))

    show("DT2 arithmetic", q("""
        SELECT date('2026-01-31','+1 month') AS plus_month,
               date('2026-03-01','-1 day') AS minus_day,
               julianday('2026-03-31') - julianday('2026-01-01') AS days
    """).to_string(index=False))

    show("DT3 string functions", q("""
        SELECT UPPER('west') AS up, LOWER('WEST') AS low,
               LENGTH('  padded  ') AS raw_len,
               LENGTH(TRIM('  padded  ')) AS trimmed_len
    """).to_string(index=False))

    show("DT4 replace", q("""
        SELECT cust_key,
               TRIM(COALESCE(cust_key,'')) AS cleaned
        FROM staging
    """).to_string(index=False))

    show("DT5 null in aggregate", q("""
        SELECT SUM(x) AS sum_x, AVG(x) AS avg_x, COUNT(x) AS count_x,
               COUNT(*) AS count_star
        FROM (SELECT 1 AS x UNION ALL SELECT 2 UNION ALL SELECT NULL)
    """).to_string(index=False))

    show("DT6 date string trap", q("""
        SELECT COUNT(*) AS march_orders FROM orders
        WHERE order_date >= '2025-03-01' AND order_date < '2025-04-01'
    """).to_string(index=False))


# ======================================== 13.10 performance/integration
def section_1310() -> None:
    show("PF1 explain", q("EXPLAIN QUERY PLAN SELECT * FROM orders WHERE customer_id = 'C-00001'").to_string(index=False))

    show("PF2 without index", q("EXPLAIN QUERY PLAN SELECT * FROM orders WHERE status = 'shipped'").to_string(index=False))

    show("PF3 sargable vs not", q("EXPLAIN QUERY PLAN SELECT * FROM orders WHERE substr(order_date,1,4) = '2025'").to_string(index=False))

    show("PF4 view vs table", {
        "view_rows": int(scalar("SELECT COUNT(*) FROM v_order_lines")),
        "order_items_rows": int(scalar("SELECT COUNT(*) FROM order_items")),
    })

    show("PF5 push work down", q("""
        SELECT channel, COUNT(*) AS n, ROUND(AVG(quantity),2) AS avg_qty
        FROM v_order_lines GROUP BY channel
    """).to_string(index=False))

    show("PF6 pandas equality", {
        "sql_orders": int(scalar("SELECT COUNT(*) FROM orders")),
        "pandas_orders": len(pd.read_csv(ROOT / 'data' / 'raw' / 'shop_orders.csv')),
    })


# ================================================ 13.11 example bank
def section_1311() -> None:
    show("E1 counts", q("SELECT COUNT(*) AS n FROM orders").to_string(index=False))
    show("E2 distinct", q("SELECT COUNT(DISTINCT customer_id) AS n FROM orders").to_string(index=False))
    show("E3 null keys", q("SELECT COUNT(*) AS n FROM staging WHERE cust_key IS NULL").to_string(index=False))
    show("E4 shipped", q("SELECT COUNT(*) AS n FROM orders WHERE status='shipped'").to_string(index=False))
    show("E5 top channel", q("SELECT channel, COUNT(*) n FROM orders GROUP BY channel ORDER BY n DESC LIMIT 1").to_string(index=False))
    show("E6 nullif", q("SELECT NULLIF('', '') IS NULL AS flag").to_string(index=False))
    show("E7 row number", q("SELECT MAX(rn) AS max_rn FROM (SELECT ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) rn FROM orders)").to_string(index=False))
    show("E8 month count", q("SELECT COUNT(DISTINCT substr(order_date,1,7)) AS n FROM orders").to_string(index=False))


# =================================================== 13.12 capstone
def section_1312() -> None:
    show("K1 first-last query", q("""
        SELECT customer_id,
               MIN(order_date) AS first_order,
               MAX(order_date) AS last_order,
               COUNT(*) AS order_count,
               julianday(MAX(order_date)) - julianday(MIN(order_date))
                 AS days_between
        FROM orders GROUP BY customer_id ORDER BY customer_id LIMIT 5
    """).to_string(index=False))

    show("K2 validation totals", q("""
        SELECT COUNT(*) AS customers,
               SUM(n) AS orders
        FROM (SELECT customer_id, COUNT(*) AS n FROM orders GROUP BY customer_id)
    """).to_string(index=False))

    show("K3 cross-check pandas", {
        "sql_first_order": scalar("SELECT MIN(order_date) FROM orders"),
        "sql_last_order": scalar("SELECT MAX(order_date) FROM orders"),
        "sql_total_orders": int(scalar("SELECT COUNT(*) FROM orders")),
    })

    show("K4 artifacts", ["shop.db", "twelve_queries.sql",
                           "cohort_retention.csv", "query_review.md"])


def section_extra() -> None:
    """Additional business queries for the expanded Chapter 13 sections."""
    show("X1 monthly revenue", q("""
        SELECT substr(o.order_date,1,7) AS month,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(SUM(i.quantity*i.unit_price*
                 (1-i.discount_pct/100.0)),2) AS revenue
        FROM orders o JOIN order_items i USING (order_id)
        WHERE o.status='shipped'
        GROUP BY 1 ORDER BY 1 LIMIT 5
    """).to_string(index=False))

    show("X2 category revenue", q("""
        SELECT p.category, COUNT(*) AS lines,
               ROUND(SUM(i.quantity*i.unit_price),2) AS gross
        FROM order_items i JOIN products p USING (product_id)
        GROUP BY 1 ORDER BY gross DESC
    """).to_string(index=False))

    show("X3 tier and value", q("""
        SELECT c.loyalty_tier, COUNT(DISTINCT c.customer_id) AS customers,
               ROUND(AVG(1.0*o.rev/o.n),2) AS avg_order_value
        FROM (SELECT customer_id, COUNT(*) n,
                     SUM(quantity*unit_price) rev
              FROM order_items i JOIN orders o USING (order_id)
              GROUP BY customer_id) o
        JOIN customers c USING (customer_id)
        GROUP BY 1 ORDER BY customers DESC
    """).to_string(index=False))

    show("X4 campaign response", q("""
        SELECT campaign_id, COUNT(*) AS exposures,
               SUM(responded) AS responses,
               ROUND(1.0*SUM(responded)/COUNT(*),4) AS rate
        FROM exposures GROUP BY 1 ORDER BY 1
    """).to_string(index=False))

    show("X5 target vs actual", q("""
        SELECT t.month, t.region, t.revenue_target,
               COALESCE(a.revenue,0) AS actual
        FROM targets t LEFT JOIN (
          SELECT substr(o.order_date,1,7) AS month, c.region,
                 SUM(i.quantity*i.unit_price) AS revenue
          FROM orders o JOIN order_items i USING (order_id)
          JOIN customers c USING (customer_id)
          GROUP BY 1,2) a
          ON t.month=a.month AND t.region=a.region
        ORDER BY t.month LIMIT 5
    """).to_string(index=False))

    show("X6 status amounts", q("""
        SELECT o.status, COUNT(*) AS orders,
               ROUND(SUM(i.quantity*i.unit_price*
                 (1-i.discount_pct/100.0)),2) AS revenue
        FROM orders o JOIN order_items i USING (order_id)
        GROUP BY 1 ORDER BY revenue DESC
    """).to_string(index=False))

    show("X7 discounts", q("""
        SELECT discount_pct, COUNT(*) AS lines
        FROM order_items GROUP BY 1 ORDER BY 1 LIMIT 6
    """).to_string(index=False))

    show("X8 lines per order", q("""
        SELECT lines_per_order, COUNT(*) AS orders FROM (
          SELECT order_id, COUNT(*) AS lines_per_order
          FROM order_items GROUP BY 1
        ) GROUP BY 1 ORDER BY 1
    """).to_string(index=False))

    show("X9 tenure buckets", q("""
        SELECT CASE
                 WHEN julianday('2026-01-01') - julianday(signup_date) < 365
                      THEN '<1 year'
                 WHEN julianday('2026-01-01') - julianday(signup_date) < 730
                      THEN '1-2 years'
                 ELSE '2+ years' END AS tenure,
               COUNT(*) AS customers
        FROM customers GROUP BY 1 ORDER BY customers DESC
    """).to_string(index=False))

    show("X10 repeat purchase", q("""
        SELECT CASE WHEN n=1 THEN 'one order' ELSE 'repeat' END AS grp_label,
               COUNT(*) AS customers
        FROM (SELECT customer_id, COUNT(*) n FROM orders GROUP BY 1)
        GROUP BY 1
    """).to_string(index=False))

    show("X11 monthly active customers", q("""
        SELECT substr(order_date,1,7) AS month,
               COUNT(DISTINCT customer_id) AS active
        FROM orders GROUP BY 1 ORDER BY 1 LIMIT 4
    """).to_string(index=False))

    show("X12 channel by region", q("""
        SELECT c.region, o.channel, COUNT(*) AS orders,
               ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (PARTITION BY c.region),2)
                 AS pct
        FROM orders o JOIN customers c USING (customer_id)
        GROUP BY 1,2 ORDER BY 1, orders DESC LIMIT 8
    """).to_string(index=False))

    show("X13 funnel by segment", q("""
        SELECT segment,
               SUM(CASE WHEN step='visit' THEN 1 ELSE 0 END) AS visits,
               SUM(CASE WHEN step='purchase' THEN 1 ELSE 0 END) AS purchases,
               ROUND(1.0*SUM(CASE WHEN step='purchase' THEN 1 ELSE 0 END) /
                     SUM(CASE WHEN step='visit' THEN 1 ELSE 0 END),4) AS cvr
        FROM funnel_events GROUP BY 1 ORDER BY 1
    """).to_string(index=False))

    show("X14 first cohort by channel", q("""
        WITH firsts AS (
          SELECT customer_id, MIN(order_date) AS first_date
          FROM orders GROUP BY 1
        ) SELECT substr(f.first_date,1,7) AS cohort, o.channel,
                 COUNT(*) AS customers
          FROM firsts f JOIN orders o
            ON f.customer_id=o.customer_id AND f.first_date=o.order_date
          GROUP BY 1,2 ORDER BY 1 LIMIT 5
    """).to_string(index=False))

    show("X15 table sizes", q("""
        SELECT 'customers' t, COUNT(*) n FROM customers
        UNION ALL SELECT 'orders', COUNT(*) FROM orders
        UNION ALL SELECT 'funnel_events', COUNT(*) FROM funnel_events
    """).to_string(index=False))

    show("X16 null-safe left join", q("""
        SELECT COUNT(*) AS rows_after,
               SUM(CASE WHEN c.customer_id IS NULL THEN 1 ELSE 0 END) AS unmatched
        FROM staging s LEFT JOIN customers c
          ON UPPER(TRIM(s.cust_key)) = c.customer_id
    """).to_string(index=False))

    show("X17 top-N third place", q("""
        WITH ranked AS (
          SELECT p.category, i.product_id, SUM(i.quantity*i.unit_price) rev,
                 ROW_NUMBER() OVER (PARTITION BY p.category
                   ORDER BY SUM(i.quantity*i.unit_price) DESC) rn
          FROM order_items i JOIN products p USING (product_id)
          GROUP BY 1,2
        ) SELECT category, ROUND(MAX(CASE WHEN rn=3 THEN rev END),2) AS third
          FROM ranked GROUP BY 1 ORDER BY 1
    """).to_string(index=False))

    show("X18 union dedupe", {
        "union_rows": int(scalar("SELECT COUNT(*) FROM (SELECT 1 x UNION SELECT 1)")),
        "union_all_rows": int(scalar("SELECT COUNT(*) FROM (SELECT 1 x UNION ALL SELECT 1)")),
    })

    show("X19 distinct order value", q("""
        SELECT ROUND(AVG(order_value),2) AS avg_revenue,
               ROUND(MIN(order_value),2) AS min_revenue,
               ROUND(MAX(order_value),2) AS max_revenue
        FROM (SELECT SUM(quantity*unit_price) AS order_value
              FROM order_items GROUP BY order_id)
    """).to_string(index=False))

    show("X20 final scorecard", {
        "orders": int(scalar("SELECT COUNT(*) FROM orders")),
        "customers": int(scalar("SELECT COUNT(*) FROM customers")),
        "lines": int(scalar("SELECT COUNT(*) FROM order_items")),
        "months": int(scalar("SELECT COUNT(DISTINCT substr(order_date,1,7)) FROM orders")),
    })


def main() -> None:
    print("python", "pandas", pd.__version__)
    print()
    for section in (
        section_131, section_132, section_133, section_134, section_135,
        section_136, section_137, section_138, section_139, section_1310,
        section_1311, section_1312, section_extra,
    ):
        print("=" * 78)
        section()


if __name__ == "__main__":
    main()
