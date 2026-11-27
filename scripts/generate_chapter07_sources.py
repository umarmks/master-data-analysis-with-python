#!/usr/bin/env python3
"""Generate the join/reshape fixtures that Chapter 7 uses.

Run from the project root:

    python scripts/generate_chapter07_sources.py

Outputs (all under data/raw/audit/):

    price_history.csv           product prices with effective dates (as-of join)
    customers_join_dirty.csv    keys with case/space/leading-zero variants
    customers_duplicate_keys.csv dimension with duplicate keys (fan-out trap)
    monthly_report_wide.csv     spreadsheet-style 36-column report
    campaign_customer.csv       campaign exposure per customer

Everything is deterministic. The other inputs Chapter 7 reads (the 12 monthly
sales exports, kestrel_orders.csv, kestrel_customers.csv, kestrel_products.csv,
and data/raw/campaigns.jsonl) already ship with the book.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
AUDIT = RAW / "audit"

SEED = 707


def price_history() -> None:
    """Three effective-dated prices for every product in the catalogue."""
    products = pd.read_csv(RAW / "kestrel_products.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for _, p in products.iterrows():
        base = float(p["list_price"])
        for date, factor in [("2024-01-01", 1.00),
                             ("2025-01-01", 1.04),
                             ("2026-01-01", 0.97)]:
            price = round(base * factor * float(rng.normal(1.0, 0.01)), 2)
            rows.append((p["product_id"], date, price))
    out = pd.DataFrame(rows, columns=["product_id", "effective_from", "unit_price"])
    out = out.sort_values(["product_id", "effective_from"]).reset_index(drop=True)
    out.to_csv(AUDIT / "price_history.csv", index=False)
    print(f"  price_history.csv: {len(out)} prices for {out['product_id'].nunique()} products")


def customers_join_dirty() -> None:
    """The same 20 customers written five different ways each."""
    rnd = random.Random(SEED)
    styles = [
        lambda k: k,                              # clean
        lambda k: k.lower(),                       # case variant
        lambda k: " " + k,                         # leading space
        lambda k: k + " ",                         # trailing space
        lambda k: "C-" + str(int(k[2:])),          # lost leading zeros
    ]
    rows = []
    for i in range(1, 21):
        key = f"C-{i:06d}"
        for style in styles:
            rows.append((style(key), f"Customer {i}",
                         rnd.choice(["Bronze", "Silver", "Gold"])))
    out = pd.DataFrame(rows, columns=["customer_id", "customer_name", "tier"])
    out.to_csv(AUDIT / "customers_join_dirty.csv", index=False)
    print(f"  customers_join_dirty.csv: {len(out)} rows, {out['customer_id'].nunique()} keys")


def customers_duplicate_keys() -> None:
    """A dimension with a repeated key: the fan-out trap."""
    base = pd.read_csv(RAW / "kestrel_customers.csv").head(50).copy()
    dupes = base.head(12).copy()
    dupes["region"] = "MIGRATED"
    out = pd.concat([base, dupes], ignore_index=True)
    out.to_csv(AUDIT / "customers_duplicate_keys.csv", index=False)
    print(f"  customers_duplicate_keys.csv: {len(out)} rows, "
          f"{out['customer_id'].nunique()} distinct keys")


def monthly_report_wide() -> None:
    """A 36-column spreadsheet-style report: 12 months x 3 metrics."""
    rng = np.random.default_rng(SEED)
    months = pd.date_range("2025-01-01", periods=12, freq="MS").strftime("%Y-%m")
    rows = []
    for region in ["Midwest", "Northeast", "South", "West"]:
        row = {"region": region}
        base = rng.uniform(80_000, 160_000)
        for i, month in enumerate(months):
            revenue = round(float(base * (1 + 0.008 * i) * rng.normal(1, 0.04)), 2)
            orders = int(revenue / rng.uniform(170, 230))
            row[f"revenue_{month}"] = revenue
            row[f"orders_{month}"] = orders
            row[f"margin_{month}"] = round(revenue * float(rng.uniform(0.18, 0.31)), 2)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(AUDIT / "monthly_report_wide.csv", index=False)
    print(f"  monthly_report_wide.csv: {len(out)} rows, {len(out.columns)} columns")


def campaign_customer() -> None:
    """One campaign exposure row per customer for the Customer 360 project."""
    rng = np.random.default_rng(SEED)
    customers = pd.read_csv(RAW / "kestrel_customers.csv")
    chosen = customers.sample(1200, random_state=SEED)["customer_id"].tolist()
    out = pd.DataFrame({
        "customer_id": chosen,
        "campaign": rng.choice(["spring_sale", "summer_pack", "loyalty_boost"],
                               size=len(chosen), p=[0.4, 0.35, 0.25]),
        "contact_date": pd.to_datetime("2025-01-01")
                        + pd.to_timedelta(rng.integers(0, 360, size=len(chosen)), unit="D"),
        "spend": np.round(rng.uniform(5, 250, size=len(chosen)), 2),
    })
    out["contact_date"] = out["contact_date"].dt.strftime("%Y-%m-%d")
    out.to_csv(AUDIT / "campaign_customer.csv", index=False)
    print(f"  campaign_customer.csv: {len(out)} exposures")


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    print("generating Chapter 7 join/reshape fixtures ...")
    price_history()
    customers_join_dirty()
    customers_duplicate_keys()
    monthly_report_wide()
    campaign_customer()
    print("done.")


if __name__ == "__main__":
    main()
