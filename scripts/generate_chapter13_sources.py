#!/usr/bin/env python3
"""Generate the SQL fixtures that Chapter 13 uses.

Run from the project root:

    python scripts/generate_chapter13_sources.py

Outputs:

    data/raw/shop.db                     a small retail database
    data/raw/shop_customers.csv          customer extract for pandas validation
    data/raw/shop_orders.csv             order extract for pandas validation
    data/raw/shop_order_items.csv        line extract for pandas validation

Everything is deterministic. The database is rebuilt from scratch on each run.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB = RAW / "shop.db"
SEED = 1307


def build() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)

    regions = ["Midwest", "Northeast", "South", "West"]
    segments = ["Basic", "Plus", "Premium"]
    tiers = ["Bronze", "Silver", "Gold", "Platinum"]

    customers = pd.DataFrame({
        "customer_id": [f"C-{i:05d}" for i in range(1, 2001)],
        "signup_date": pd.to_datetime("2023-01-01")
                       + pd.to_timedelta(rng.integers(0, 900, size=2000), unit="D"),
        "segment": rng.choice(segments, size=2000, p=[0.5, 0.3, 0.2]),
        "region": rng.choice(regions, size=2000),
        "loyalty_tier": rng.choice(tiers, size=2000),
    })
    customers["signup_date"] = customers["signup_date"].dt.strftime("%Y-%m-%d")

    cats = {"Camping": (12, 180.0), "Footwear": (8, 120.0),
            "Apparel": (10, 80.0), "Accessories": (6, 40.0)}
    prows = []
    pid = 0
    for cat, (count, base) in cats.items():
        for j in range(count):
            pid += 1
            prows.append((f"P-{pid:04d}", cat,
                          round(float(base * rng.uniform(0.7, 1.4)), 2)))
    products = pd.DataFrame(prows, columns=["product_id", "category", "list_price"])

    order_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 731, size=8000), unit="D")
    orders = pd.DataFrame({
        "order_id": [f"O-{i:06d}" for i in range(1, 8001)],
        "customer_id": rng.choice(customers["customer_id"], size=8000),
        "order_date": order_dates.strftime("%Y-%m-%d"),
        "status": rng.choice(["shipped", "returned", "cancelled"],
                             size=8000, p=[0.86, 0.10, 0.04]),
        "channel": rng.choice(["web", "app", "phone"],
                              size=8000, p=[0.55, 0.3, 0.15]),
    })

    lines = []
    for oid in orders["order_id"]:
        k = int(rng.choice([1, 2, 3], p=[0.65, 0.28, 0.07]))
        chosen = rng.choice(products["product_id"], size=k, replace=False)
        for prod in chosen:
            qty = int(rng.integers(1, 5))
            price = float(products.loc[products["product_id"] == prod,
                                       "list_price"].iloc[0])
            disc = float(rng.choice([0, 0, 0, 10, 15, 20]))
            lines.append((oid, prod, qty, round(price, 2), disc))
    order_items = pd.DataFrame(
        lines, columns=["order_id", "product_id", "quantity",
                        "unit_price", "discount_pct"])

    campaigns = pd.DataFrame([
        ("CMP-01", "spring_sale", "2024-03-01", "2024-03-31"),
        ("CMP-02", "summer_pack", "2024-06-01", "2024-06-30"),
        ("CMP-03", "loyalty_boost", "2025-01-01", "2025-01-31"),
        ("CMP-04", "autumn_push", "2025-09-01", "2025-09-30"),
    ], columns=["campaign_id", "campaign_name", "start_date", "end_date"])

    exposures = pd.DataFrame({
        "customer_id": rng.choice(customers["customer_id"], size=3200),
        "campaign_id": rng.choice(campaigns["campaign_id"], size=3200),
        "exposed_at": pd.to_datetime("2024-03-01")
                      + pd.to_timedelta(rng.integers(0, 640, size=3200), unit="D"),
        "responded": rng.binomial(1, 0.14, size=3200),
    })
    exposures["exposed_at"] = exposures["exposed_at"].dt.strftime("%Y-%m-%d")

    months = pd.date_range("2024-01-01", periods=24, freq="MS").strftime("%Y-%m")
    targets = pd.DataFrame([
        (m, r, round(float(50000 * (1 + 0.005 * i)
                           * rng.uniform(0.9, 1.1)), 2))
        for i, m in enumerate(months) for r in regions
    ], columns=["month", "region", "revenue_target"])

    steps = ["visit", "product", "cart", "checkout", "purchase"]
    funnel = []
    for i in range(6000):
        seg = rng.choice(segments, p=[0.5, 0.3, 0.2])
        depth = int(rng.choice([1, 2, 3, 4, 5], p=[0.25, 0.25, 0.2, 0.15, 0.15]))
        for s in steps[:depth]:
            funnel.append((f"U-{i:05d}", s, seg,
                           "2025-01-01" if s == "visit" else ""))
    funnel_events = pd.DataFrame(
        funnel, columns=["user_id", "step", "segment", "event_date"])

    # a staging table with the defects that cause fan-out and null traps
    staging = pd.DataFrame({
        "order_ref": ["O-000001", "O-000002", "O-000003", "O-000004"],
        "cust_key": [" C-00001", "c-00002", "C-00003", None],
        "amount": ["100.00", "$120.50", "N/A", "80.00"],
    })

    # a duplicate-key dimension for the fan-out demonstration
    dupe = customers.head(12).copy()
    dupe["loyalty_tier"] = "MIGRATED"
    customers_dup = pd.concat([customers, dupe], ignore_index=True)

    return {
        "customers": customers,
        "customers_dup": customers_dup,
        "products": products,
        "orders": orders,
        "order_items": order_items,
        "campaigns": campaigns,
        "exposures": exposures,
        "targets": targets,
        "funnel_events": funnel_events,
        "staging": staging,
    }


def write_db(t: dict[str, pd.DataFrame]) -> None:
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    for name in ["customers", "customers_dup", "products", "orders",
                 "order_items", "campaigns", "exposures", "targets",
                 "funnel_events", "staging"]:
        t[name].to_sql(name, con, index=False)
    con.executescript("""
        CREATE INDEX idx_orders_customer ON orders(customer_id);
        CREATE INDEX idx_orders_date ON orders(order_date);
        CREATE INDEX idx_items_order ON order_items(order_id);
        CREATE VIEW v_order_lines AS
          SELECT o.order_id, o.order_date, o.customer_id, o.status, o.channel,
                 i.product_id, i.quantity, i.unit_price, i.discount_pct,
                 ROUND(i.quantity * i.unit_price * (1 - i.discount_pct/100.0), 2)
                   AS revenue
          FROM orders o JOIN order_items i ON o.order_id = i.order_id;
    """)
    con.commit()
    con.close()

    t["customers"].to_csv(RAW / "shop_customers.csv", index=False)
    t["orders"].to_csv(RAW / "shop_orders.csv", index=False)
    t["order_items"].to_csv(RAW / "shop_order_items.csv", index=False)
    tables = ", ".join(f"{k}={len(v)}" for k, v in t.items())
    print(f"  shop.db: {DB.stat().st_size:,} bytes; {tables}")


def main() -> None:
    print("generating Chapter 13 SQL fixtures ...")
    write_db(build())
    print("done.")


if __name__ == "__main__":
    main()
