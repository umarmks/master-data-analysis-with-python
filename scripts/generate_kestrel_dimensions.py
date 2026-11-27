#!/usr/bin/env python3
"""Generate the customer and product dimension files for the Kestrel case study.

Deterministic: the same orders file always produces the same dimensions, so the
join examples in Chapter 3 and the analysis tables in later chapters are stable.

    python scripts/generate_kestrel_dimensions.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ORDERS = RAW / "kestrel_orders.csv"
CUSTOMERS = RAW / "kestrel_customers.csv"
PRODUCTS = RAW / "kestrel_products.csv"

REGIONS = ["Midwest", "Northeast", "South", "West"]
TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
CATEGORY_NAMES = {
    "Camping": ["Ridge Tent", "Trail Lantern", "Sleeping Bag", "Camp Stove", "Hiking Pole"],
    "Footwear": ["Trail Runner", "Summit Boot", "River Sandal", "Trail Shoe", "Alpine Boot"],
    "Apparel": ["Alpine Jacket", "Storm Shell", "Base Layer", "Fleece Vest", "Rain Pant"],
    "Accessories": ["Daypack", "Water Bottle", "Headlamp", "Trek Pole", "Dry Bag"],
}


def main() -> None:
    orders = pd.read_csv(ORDERS, dtype={"customer_id": "string"})
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    orders = orders.drop_duplicates().dropna(subset=["order_date"])

    rng = np.random.default_rng(20260402)

    # --- products -------------------------------------------------------
    products = (
        orders[["product_id", "category", "unit_price"]]
        .groupby(["product_id", "category"], as_index=False)
        .agg(list_price=("unit_price", "median"))
        .sort_values("product_id")
        .reset_index(drop=True)
    )
    names = []
    counters: dict[str, int] = {}
    for category in products["category"]:
        index = counters.get(category, 0)
        pool = CATEGORY_NAMES[category]
        names.append(f"{pool[index % len(pool)]}")
        counters[category] = index + 1
    products["product_name"] = names
    products["list_price"] = products["list_price"].round(2)
    products = products[["product_id", "product_name", "category", "list_price"]]
    products.to_csv(PRODUCTS, index=False)

    # --- customers ------------------------------------------------------
    customers = (
        orders[orders["customer_id"].notna()]["customer_id"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
        .to_frame()
    )
    n = len(customers)
    signup = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 900, size=n), unit="D"
    )
    customers["signup_date"] = signup.strftime("%Y-%m-%d")
    customers["loyalty_tier"] = rng.choice(TIERS, size=n, p=[0.5, 0.3, 0.15, 0.05])
    customers["home_region"] = rng.choice(REGIONS, size=n)

    # Deliberately omit 50 customers so the left-join indicator has work to do.
    omit = set(rng.choice(customers["customer_id"], size=50, replace=False))
    customers = customers[~customers["customer_id"].isin(omit)].reset_index(drop=True)
    customers.to_csv(CUSTOMERS, index=False)

    print("customers:", len(customers), "->", CUSTOMERS.name)
    print("products: ", len(products), "->", PRODUCTS.name)
    print("omitted customers (by design):", len(omit))


if __name__ == "__main__":
    main()
