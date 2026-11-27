#!/usr/bin/env python3
"""Generate the audit fixtures that Chapter 5 profiles.

Run from the project root:

    python scripts/generate_chapter05_sources.py

Outputs (all under data/raw/audit/):

    orders_sample_dirty.csv     ~40 hand-built order lines with planted defects
    customers_dirty.csv         ~25 messy customer records for the validation suite
    finance_monthly_net.csv     a finance net-revenue export derived from kestrel.db

Everything is deterministic. The finance file is computed from the SQLite
database written by generate_chapter04_sources.py, so run that first if
data/raw/kestrel.db is missing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
AUDIT = RAW / "audit"

COLUMNS = [
    "order_id", "order_date", "customer_id", "channel", "region", "category",
    "product_id", "quantity", "unit_price", "discount_pct", "shipping_cost",
    "returned", "revenue", "store_id",
]


def dirty_orders() -> None:
    """A small orders file whose defects are the subject of the audit."""
    rows = [
        # ---- 8 clean lines, to give the profile something normal to mix with
        ("O-900001", "2026-01-04", "C-000101", "Web", "Midwest", "Camping", "P-1001", "2", "58.53", "0.0", "3.95", "0", "117.06", "S-01"),
        ("O-900002", "2026-01-05", "C-000102", "App", "South", "Footwear", "P-2003", "1", "129.00", "10.0", "0.00", "0", "116.10", "S-02"),
        ("O-900003", "2026-01-06", "C-000103", "Phone", "West", "Apparel", "P-3002", "3", "39.95", "0.0", "7.95", "0", "119.85", "S-03"),
        ("O-900004", "2026-01-07", "C-000104", "Web", "Northeast", "Accessories", "P-4001", "1", "24.50", "0.0", "0.00", "0", "24.50", "S-04"),
        ("O-900005", "2026-01-08", "C-000105", "App", "Midwest", "Camping", "P-1005", "4", "44.63", "0.0", "12.95", "1", "165.57", "S-05"),
        ("O-900006", "2026-01-09", "C-000106", "Web", "West", "Footwear", "P-2001", "1", "88.29", "5.0", "3.95", "0", "87.83", "S-06"),
        ("O-900007", "2026-01-10", "C-000107", "Phone", "South", "Apparel", "P-3004", "2", "65.00", "20.0", "0.00", "0", "104.00", "S-07"),
        ("O-900008", "2026-01-11", "C-000108", "Web", "Northeast", "Accessories", "P-4002", "1", "31.42", "0.0", "0.00", "0", "31.42", "S-08"),
        # ---- planted defects below
        # 1. exact duplicate of O-900001
        ("O-900001", "2026-01-04", "C-000101", "Web", "Midwest", "Camping", "P-1001", "2", "58.53", "0.0", "3.95", "0", "117.06", "S-01"),
        # 2. duplicate order_id with a conflicting quantity
        ("O-900002", "2026-01-05", "C-000102", "App", "South", "Footwear", "P-2003", "2", "129.00", "10.0", "0.00", "0", "232.20", "S-02"),
        # 3. missing order_id
        ("", "2026-01-12", "C-000109", "Web", "West", "Camping", "P-1002", "1", "44.00", "0.0", "0.00", "0", "44.00", "S-09"),
        # 4. unparseable date
        ("O-900010", "unknown", "C-000110", "Web", "South", "Camping", "P-1003", "1", "58.53", "0.0", "0.00", "0", "58.53", "S-01"),
        # 5. impossible calendar date (Feb 30)
        ("O-900011", "2026-02-30", "C-000111", "App", "Midwest", "Footwear", "P-2004", "1", "99.00", "0.0", "0.00", "0", "99.00", "S-02"),
        # 6. future date
        ("O-900012", "2030-05-01", "C-000112", "Web", "West", "Apparel", "P-3001", "1", "75.00", "0.0", "0.00", "0", "75.00", "S-03"),
        # 7. missing region
        ("O-900013", "2026-01-13", "C-000113", "Web", "", "Camping", "P-1004", "1", "44.00", "0.0", "0.00", "0", "44.00", "S-04"),
        # 8. non-breaking space in region
        ("O-900014", "2026-01-14", "C-000114", "App", "West\u00a0", "Footwear", "P-2002", "1", "88.29", "0.0", "0.00", "0", "88.29", "S-05"),
        # 9. case variant in category
        ("O-900015", "2026-01-15", "C-000115", "Web", "South", "camping", "P-1006", "2", "44.63", "0.0", "0.00", "0", "89.26", "S-06"),
        # 10. whitespace-padded category
        ("O-900016", "2026-01-16", "C-000116", "Phone", "Northeast", " Camping ", "P-1007", "1", "58.53", "0.0", "0.00", "0", "58.53", "S-07"),
        # 11. sentinel customer_id
        ("O-900017", "2026-01-17", "-1", "Web", "West", "Apparel", "P-3003", "1", "75.00", "0.0", "0.00", "0", "75.00", "S-08"),
        # 12. sentinel quantity
        ("O-900018", "2026-01-18", "C-000117", "App", "South", "Camping", "P-1008", "9999", "44.00", "0.0", "0.00", "0", "439956.00", "S-09"),
        # 13. negative quantity
        ("O-900019", "2026-01-19", "C-000118", "Web", "Midwest", "Footwear", "P-2005", "-2", "99.00", "0.0", "0.00", "0", "-198.00", "S-10"),
        # 14. zero unit_price
        ("O-900020", "2026-01-20", "C-000119", "Web", "West", "Apparel", "P-3005", "1", "0.00", "0.0", "0.00", "0", "0.00", "S-11"),
        # 15. negative unit_price
        ("O-900021", "2026-01-21", "C-000120", "App", "South", "Camping", "P-1009", "1", "-10.00", "0.0", "0.00", "0", "-10.00", "S-12"),
        # 16. currency-formatted unit_price
        ("O-900022", "2026-01-22", "C-000121", "Web", "Northeast", "Footwear", "P-2006", "1", "$102.10", "0.0", "0.00", "0", "102.10", "S-01"),
        # 17. discount over 100
        ("O-900023", "2026-01-23", "C-000122", "Phone", "West", "Apparel", "P-3006", "1", "75.00", "150.0", "0.00", "0", "-37.50", "S-02"),
        # 18. negative discount
        ("O-900024", "2026-01-24", "C-000123", "Web", "Midwest", "Camping", "P-1010", "1", "58.53", "-10.0", "0.00", "0", "64.38", "S-03"),
        # 19. returned out of domain
        ("O-900025", "2026-01-25", "C-000124", "App", "South", "Footwear", "P-2007", "1", "88.29", "0.0", "0.00", "2", "88.29", "S-04"),
        # 20. broken product foreign key
        ("O-900026", "2026-01-26", "C-000125", "Web", "West", "Camping", "P-9999", "1", "44.00", "0.0", "0.00", "0", "44.00", "S-05"),
        # 21. broken store foreign key
        ("O-900027", "2026-01-27", "C-000126", "Phone", "Northeast", "Apparel", "P-3007", "1", "75.00", "0.0", "0.00", "0", "75.00", "S-99"),
        # 22. revenue inconsistent with the formula
        ("O-900028", "2026-01-28", "C-000127", "Web", "South", "Camping", "P-1001", "2", "58.53", "0.0", "0.00", "0", "999.99", "S-06"),
        # 23. negative shipping
        ("O-900029", "2026-01-29", "C-000130", "App", "Midwest", "Accessories", "P-4003", "1", "31.42", "0.0", "-3.95", "0", "31.42", "S-07"),
    ]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    AUDIT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(AUDIT / "orders_sample_dirty.csv", index=False)
    print(f"  orders_sample_dirty.csv: {len(frame)} rows")


def dirty_customers() -> None:
    """Customer records with inconsistent text, FKs, and sentinels."""
    rows = [
        ("C-000101", "2024-03-12", "Gold", "Midwest"),
        ("C-000102", "2024-05-01", "Silver", "South"),
        ("C-000103", "2023-11-20", "Bronze", "West"),
        ("C-000104", "2025-01-15", "Platinum", "Northeast"),
        ("C-000105", "2024-07-07", "gold", "Midwest"),          # case variant tier
        ("C-000106", " 2024-08-02", "Bronze", "South"),          # leading whitespace in date
        ("C-000107", "2024-09-09", "SILVER", "West"),            # case variant tier
        ("C-000108", "2024-10-10", "Bronze", "northeast"),       # case variant region
        ("C-000109", "2024-11-11", "Bronze", "North East"),      # spelling variant region
        ("C-000110", "2024-12-12", "Bronze", ""),                # missing region
        ("", "2025-02-01", "Bronze", "West"),                    # missing key
        ("C-000111", "2025-02-02", "Unknown", "West"),           # sentinel tier
        ("C-000112", "2025-02-03", "Bronze", "West"),
        ("C-000102", "2024-05-01", "Silver", "South"),           # exact duplicate
        ("C-000113", "2025-02-04", "Bronze", "South"),
        ("C-000114", "1900-01-01", "Bronze", "Midwest"),         # sentinel date
        ("C-000115", "2025-02-05", "Platinum", "Northeast"),
        ("C-000116", "2025-13-40", "Bronze", "West"),            # impossible date
        ("C-000117", "2025-02-06", "Gold", "Midwest"),
        ("C-000118", "2025-02-07", "Bronze", "South"),
        ("C-000119", "2025-02-08", "Bronze", "West"),
        ("C-000120", "2025-02-09", "Silver", "Northeast"),
        ("C-000121", "2025-02-10", "Bronze", "South"),
        ("C-000122", "2025-02-11", "Bronze", "Midwest"),
        ("C-000123", "2025-02-12", "Bronze", "West"),
    ]
    frame = pd.DataFrame(rows, columns=["customer_id", "signup_date", "loyalty_tier", "home_region"])
    frame.to_csv(AUDIT / "customers_dirty.csv", index=False)
    print(f"  customers_dirty.csv: {len(frame)} rows")


def finance_export() -> None:
    """A finance net-revenue export derived from kestrel.db, with one adjustment."""
    db = RAW / "kestrel.db"
    if not db.exists():
        raise SystemExit("data/raw/kestrel.db missing; run generate_chapter04_sources.py first.")
    con = sqlite3.connect(db)
    monthly = pd.read_sql(
        "SELECT substr(order_date, 1, 7) AS month, "
        "       ROUND(SUM(revenue), 2)                 AS gross, "
        "       ROUND(SUM(revenue * returned), 2)      AS returns, "
        "       ROUND(SUM(shipping_cost), 2)           AS shipping "
        "FROM orders GROUP BY month ORDER BY month",
        con,
    )
    con.close()
    monthly["net_revenue"] = (monthly["gross"] - monthly["returns"] - monthly["shipping"]).round(2)
    monthly["note"] = ""
    # One deliberate manual adjustment, with an explanatory note.
    idx = monthly.index[monthly["month"] == "2026-02"][0]
    monthly.loc[idx, "net_revenue"] = round(monthly.loc[idx, "net_revenue"] + 15000.00, 2)
    monthly.loc[idx, "note"] = "manual accrual reversal"
    out = monthly[["month", "net_revenue", "note"]]
    out.to_csv(AUDIT / "finance_monthly_net.csv", index=False)
    print(f"  finance_monthly_net.csv: {len(out)} months")


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    print("generating Chapter 5 audit fixtures ...")
    dirty_orders()
    dirty_customers()
    finance_export()
    print("done.")


if __name__ == "__main__":
    main()
