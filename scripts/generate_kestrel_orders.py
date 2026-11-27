#!/usr/bin/env python3
"""Generate the synthetic Kestrel Outfitters order-line dataset.

The output is deterministic for a given seed and is shaped for Chapter 1 of
*Data Analysis with Python*. It contains the same structural defects used in
the chapter: duplicate migration rows, invalid dates, missing regions, guest
checkouts, and blank discounts.

Run from anywhere:

    python scripts/generate_kestrel_orders.py

Output:

    data/raw/kestrel_orders.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260418
N_ORDERS = 213_040
N_UNIQUE_LINES = 487_936
N_DUPLICATES = 1_184
N_ROWS = N_UNIQUE_LINES + N_DUPLICATES
N_INVALID_DATES = 14

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "kestrel_orders.csv"

CATEGORIES = ["Camping", "Footwear", "Apparel", "Accessories"]
CHANNELS = ["Web", "App", "Phone"]
REGIONS = ["Midwest", "Northeast", "South", "West"]

# Orders per month. The totals are exact:
#   2024 = 91,009; 2025 = 99,141; 2026 Q1 = 22,890; total = 213,040.
MONTHLY_ORDERS = {
    "2024-01": 7_000,
    "2024-02": 6_700,
    "2024-03": 7_400,
    "2024-04": 7_100,
    "2024-05": 7_300,
    "2024-06": 7_200,
    "2024-07": 7_400,
    "2024-08": 7_500,
    "2024-09": 7_100,
    "2024-10": 7_900,
    "2024-11": 9_100,
    "2024-12": 9_309,
    "2025-01": 7_600,
    "2025-02": 7_300,
    "2025-03": 8_741,
    "2025-04": 7_800,
    "2025-05": 7_900,
    "2025-06": 7_800,
    "2025-07": 7_900,
    "2025-08": 8_000,
    "2025-09": 7_700,
    "2025-10": 8_500,
    "2025-11": 9_700,
    "2025-12": 10_200,
    "2026-01": 7_630,
    "2026-02": 7_211,
    "2026-03": 8_049,
}

# Product catalogue: product ID, list price, relative popularity.
CATALOG = {
    "Camping": [
        ("P-1001", 59.0, 0.18),
        ("P-1002", 89.0, 0.13),
        ("P-1003", 45.0, 0.18),
        ("P-1004", 139.0, 0.10),
        ("P-1005", 35.0, 0.15),
        ("P-1006", 169.0, 0.07),
        ("P-1007", 55.0, 0.11),
        ("P-1008", 349.0, 0.04),
        ("P-1009", 1299.0, 0.01),
        ("P-1010", 45.0, 0.03),
    ],
    "Footwear": [
        ("P-2001", 59.0, 0.18),
        ("P-2002", 79.0, 0.16),
        ("P-2003", 49.0, 0.17),
        ("P-2004", 99.0, 0.10),
        ("P-2005", 119.0, 0.08),
        ("P-2006", 69.0, 0.13),
        ("P-2007", 55.0, 0.09),
        ("P-2008", 149.0, 0.04),
        ("P-2009", 89.0, 0.04),
        ("P-2010", 39.0, 0.01),
    ],
    "Apparel": [
        ("P-3001", 35.0, 0.18),
        ("P-3002", 45.0, 0.16),
        ("P-3003", 29.0, 0.17),
        ("P-3004", 69.0, 0.10),
        ("P-3005", 79.0, 0.08),
        ("P-3006", 55.0, 0.12),
        ("P-3007", 35.0, 0.10),
        ("P-3008", 99.0, 0.04),
        ("P-3009", 69.0, 0.04),
        ("P-3010", 25.0, 0.01),
    ],
    "Accessories": [
        ("P-4001", 19.0, 0.19),
        ("P-4002", 29.0, 0.17),
        ("P-4003", 15.0, 0.17),
        ("P-4004", 39.0, 0.11),
        ("P-4005", 45.0, 0.08),
        ("P-4006", 25.0, 0.12),
        ("P-4007", 35.0, 0.08),
        ("P-4008", 59.0, 0.04),
        ("P-4009", 69.0, 0.03),
        ("P-4010", 12.0, 0.01),
    ],
}

# Chapter 1 targets, in net revenue (after returns and shipping).
Q1_TARGETS = {
    "2025Q1": {
        "category": {
            "Camping": 1_490_000,
            "Footwear": 1_180_000,
            "Apparel": 1_020_000,
            "Accessories": 702_118,
        },
        "channel": {"Web": 2_702_000, "App": 1_318_000, "Phone": 372_118},
    },
    "2026Q1": {
        "category": {
            "Camping": 1_324_000,
            "Footwear": 1_170_000,
            "Apparel": 1_000_000,
            "Accessories": 625_998,
        },
        "channel": {"Web": 2_470_000, "App": 1_300_000, "Phone": 349_998},
    },
}


def _order_dates(rng: np.random.Generator) -> pd.DatetimeIndex:
    parts: list[pd.DatetimeIndex] = []
    for month, count in MONTHLY_ORDERS.items():
        period = pd.Period(month, freq="M")
        days = rng.integers(1, period.days_in_month + 1, size=count)
        start = pd.Timestamp(f"{month}-01")
        parts.append(start + pd.to_timedelta(days - 1, unit="D"))
    return pd.DatetimeIndex(np.concatenate([p.values for p in parts]))


def _line_counts(rng: np.random.Generator) -> np.ndarray:
    # Mean is about 2.29 lines per order, close to the chapter's 2.3.
    counts = rng.choice(
        np.arange(1, 9),
        size=N_ORDERS,
        p=[0.35, 0.30, 0.18, 0.09, 0.04, 0.02, 0.015, 0.005],
    ).astype(np.int64)

    diff = N_UNIQUE_LINES - int(counts.sum())
    while diff != 0:
        if diff > 0:
            candidates = np.flatnonzero(counts < 8)
            take = min(diff, len(candidates), 20_000)
            chosen = rng.choice(candidates, size=take, replace=False)
            counts[chosen] += 1
            diff -= take
        else:
            candidates = np.flatnonzero(counts > 1)
            take = min(-diff, len(candidates), 20_000)
            chosen = rng.choice(candidates, size=take, replace=False)
            counts[chosen] -= 1
            diff += take
    return counts


def _assign_channels(
    rng: np.random.Generator, quarters: pd.Index
) -> np.ndarray:
    channel = rng.choice(CHANNELS, size=N_ORDERS, p=[0.61, 0.30, 0.09])
    channel = channel.astype(object)

    overrides = {
        "2025Q1": [0.615, 0.300, 0.085],
        "2026Q1": [0.600, 0.316, 0.084],
    }
    for quarter, probs in overrides.items():
        mask = quarters == quarter
        channel[mask] = rng.choice(CHANNELS, size=int(mask.sum()), p=probs)
    return channel


def _assign_categories(
    rng: np.random.Generator, quarters: pd.Index
) -> np.ndarray:
    category = rng.choice(
        CATEGORIES, size=len(quarters), p=[0.34, 0.27, 0.23, 0.16]
    ).astype(object)

    overrides = {
        "2025Q1": [0.339, 0.269, 0.232, 0.160],
        "2026Q1": [0.321, 0.284, 0.243, 0.152],
    }
    for quarter, probs in overrides.items():
        mask = quarters == quarter
        category[mask] = rng.choice(
            CATEGORIES, size=int(mask.sum()), p=probs
        )
    return category


def _assign_products(
    rng: np.random.Generator, lines: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    product_id = np.empty(len(lines), dtype=object)
    unit_price = np.empty(len(lines), dtype=float)

    for category, items in CATALOG.items():
        mask = (lines["category"] == category).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        ids = np.array([item[0] for item in items], dtype=object)
        prices = np.array([item[1] for item in items], dtype=float)
        weights = np.array([item[2] for item in items], dtype=float)
        weights = weights / weights.sum()
        idx = rng.choice(len(items), size=n, p=weights)

        # A little price variation keeps repeated lines from being exact
        # duplicates while preserving the product's price band.
        jitter = rng.uniform(0.97, 1.03, size=n)
        product_id[mask] = ids[idx]
        unit_price[mask] = np.round(prices[idx] * jitter, 2)

    return product_id, unit_price


def _return_probability(category: np.ndarray, quarter: np.ndarray) -> np.ndarray:
    probs = np.full(len(category), 0.078, dtype=float)

    base = {
        "Camping": 0.105,
        "Footwear": 0.064,
        "Apparel": 0.070,
        "Accessories": 0.057,
    }
    for cat, value in base.items():
        probs[category == cat] = value

    overrides = {
        ("2025Q1", "Camping"): 0.098,
        ("2025Q1", "Footwear"): 0.061,
        ("2025Q1", "Apparel"): 0.068,
        ("2025Q1", "Accessories"): 0.052,
        ("2026Q1", "Camping"): 0.118,
        ("2026Q1", "Footwear"): 0.064,
        ("2026Q1", "Apparel"): 0.072,
        ("2026Q1", "Accessories"): 0.061,
    }
    for (qtr, cat), value in overrides.items():
        probs[(quarter == qtr) & (category == cat)] = value
    return probs


def _calibrate_quarter(
    lines: pd.DataFrame,
    mask: np.ndarray,
    category_targets: dict[str, float],
    channel_targets: dict[str, float],
) -> None:
    """Scale prices so category and channel net revenue match the targets.

    Each line's net revenue is:
        multiplier * (revenue - return_value) - shipping_cost

    We solve for a category-by-channel multiplier with iterative proportional
    fitting, then apply a final correction so the quarter total is exact.
    """
    sub = lines.loc[mask]
    cat_idx = pd.Categorical(sub["category"], categories=CATEGORIES).codes
    ch_idx = pd.Categorical(sub["channel"], categories=CHANNELS).codes

    base = (
        sub["revenue"].to_numpy() * (1 - sub["returned"].to_numpy())
    )
    ship = sub["shipping_cost"].to_numpy()

    base_matrix = np.zeros((len(CATEGORIES), len(CHANNELS)))
    ship_matrix = np.zeros_like(base_matrix)
    np.add.at(base_matrix, (cat_idx, ch_idx), base)
    np.add.at(ship_matrix, (cat_idx, ch_idx), ship)

    row_targets = np.array([category_targets[c] for c in CATEGORIES])
    col_targets = np.array([channel_targets[c] for c in CHANNELS])

    # Iterative proportional fitting with the fixed shipping offset.
    col_factor = np.ones(len(CHANNELS))
    row_factor = np.ones(len(CATEGORIES))
    for _ in range(500):
        row_factor = (row_targets + ship_matrix.sum(axis=1)) / np.maximum(
            (base_matrix * col_factor).sum(axis=1), 1e-9
        )
        new_col = (col_targets + ship_matrix.sum(axis=0)) / np.maximum(
            (row_factor[:, None] * base_matrix).sum(axis=0), 1e-9
        )
        if np.max(np.abs(new_col - col_factor)) < 1e-12:
            col_factor = new_col
            break
        col_factor = new_col

    multipliers = row_factor[:, None] * col_factor[None, :]
    line_multiplier = multipliers[cat_idx, ch_idx]

    discount = sub["discount_pct"].fillna(0).to_numpy()
    quantity = sub["quantity"].to_numpy()
    returned = sub["returned"].to_numpy()
    price = np.round(sub["unit_price"].to_numpy() * line_multiplier, 2)
    revenue = np.round(quantity * price * (1 - discount / 100), 2)

    target_total = float(row_targets.sum())
    base_total = float((revenue * (1 - returned)).sum())
    factor = (target_total + float(ship.sum())) / max(base_total, 1e-9)
    price = np.round(price * factor, 2)
    revenue = np.round(quantity * price * (1 - discount / 100), 2)

    # Correct the remaining few cents on the largest line.
    net = revenue * (1 - returned) - ship
    diff = target_total - float(net.sum())
    if abs(diff) > 0.005:
        j = int(np.argmax(revenue))
        revenue[j] = round(revenue[j] + diff, 2)
        denom = quantity[j] * (1 - discount[j] / 100)
        price[j] = round(revenue[j] / denom, 2)
        revenue[j] = round(quantity[j] * price[j] * (1 - discount[j] / 100), 2)

    lines.loc[mask, "unit_price"] = price
    lines.loc[mask, "revenue"] = revenue


def _verify(lines: pd.DataFrame) -> None:
    q = lines["order_date"].dt.to_period("Q").astype(str)
    for quarter in ["2025Q1", "2026Q1"]:
        sub = lines.loc[q == quarter]
        net = sub["revenue"] * (1 - sub["returned"]) - sub["shipping_cost"]
        print(f"\n{quarter}")
        print(f"  orders:       {sub['order_id'].nunique():,}")
        print(f"  lines:        {len(sub):,}")
        print(f"  net revenue:  ${net.sum():,.2f}")
        print(f"  AOV:          ${net.sum() / sub['order_id'].nunique():,.2f}")
        print(f"  return rate:  {sub['returned'].mean():.3f}")
        print("  by category:")
        print(
            sub.assign(net=net)
            .groupby("category")["net"]
            .sum()
            .sort_values(ascending=False)
            .round(0)
            .to_string()
        )
        print("  by channel:")
        print(
            sub.assign(net=net)
            .groupby("channel")["net"]
            .sum()
            .sort_values(ascending=False)
            .round(0)
            .to_string()
        )


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("Generating orders...")
    order_dates = _order_dates(rng)
    quarters = order_dates.to_period("Q").astype(str)

    orders = pd.DataFrame(
        {
            "order_id": [f"O-{100001 + i:06d}" for i in range(N_ORDERS)],
            "order_date": order_dates,
            "customer_id": rng.choice(
                np.array([f"C-{i:06d}" for i in range(1, 60_001)]),
                size=N_ORDERS,
                p=(1.0 / np.arange(1, 60_001) ** 0.85)
                / (1.0 / np.arange(1, 60_001) ** 0.85).sum(),
            ),
            "channel": _assign_channels(rng, quarters),
            "region": rng.choice(
                REGIONS, size=N_ORDERS, p=[0.24, 0.22, 0.32, 0.22]
            ),
            "store_id": rng.choice(
                np.array([f"S-{i:02d}" for i in range(1, 13)]), size=N_ORDERS
            ),
        }
    )

    # Guest checkouts: blank the customer on enough orders to produce about
    # 2,141 missing customer IDs across order lines.
    guest_orders = rng.choice(N_ORDERS, size=935, replace=False)
    orders.loc[guest_orders, "customer_id"] = None

    # Missing regions: 1.8% of orders, which is about 8,800 order lines.
    missing_region = rng.random(N_ORDERS) < 0.018
    orders.loc[missing_region, "region"] = None

    print("Expanding order lines...")
    counts = _line_counts(rng)
    line_order_idx = np.repeat(np.arange(N_ORDERS), counts)
    lines = orders.iloc[line_order_idx].reset_index(drop=True)
    assert len(lines) == N_UNIQUE_LINES

    line_quarters = lines["order_date"].dt.to_period("Q").astype(str).to_numpy()

    print("Assigning products and prices...")
    lines["category"] = _assign_categories(rng, pd.Index(line_quarters))
    product_id, unit_price = _assign_products(rng, lines)
    lines["product_id"] = product_id
    lines["unit_price"] = unit_price

    lines["quantity"] = rng.choice(
        np.arange(1, 9),
        size=len(lines),
        p=[0.65, 0.25, 0.07, 0.02, 0.005, 0.003, 0.001, 0.001],
    )

    # Big-ticket items are almost always bought one at a time. Without this
    # cap a single line can dominate the revenue distribution.
    expensive = lines["unit_price"] > 250
    lines.loc[expensive, "quantity"] = 1

    discount_values = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 70.0])
    discount = rng.choice(
        discount_values,
        size=len(lines),
        p=[0.57, 0.08, 0.10, 0.10, 0.07, 0.04, 0.02, 0.01, 0.007, 0.003],
    )
    # Blank discount means "no discount was recorded", not zero.
    blank_discount = rng.random(len(lines)) < 0.0253
    discount[blank_discount] = np.nan
    lines["discount_pct"] = discount

    ret_prob = _return_probability(
        lines["category"].to_numpy(), line_quarters
    )
    lines["returned"] = (rng.random(len(lines)) < ret_prob).astype(int)

    lines["revenue"] = np.round(
        lines["quantity"]
        * lines["unit_price"]
        * (1 - lines["discount_pct"].fillna(0) / 100),
        2,
    )

    # Shipping is charged on kept orders only; a returned order refunds its
    # shipping. Small orders ship free, mid-size orders carry at most $3.95,
    # and the $49.95 express option is rare enough to leave a handful of
    # genuinely loss-making lines.
    revenue_for_shipping = lines["revenue"].to_numpy()
    returned_flag = lines["returned"].to_numpy()
    shipping = np.zeros(len(lines), dtype=float)

    mid = (returned_flag == 0) & (revenue_for_shipping >= 15) & (
        revenue_for_shipping < 35
    )
    shipping[mid] = rng.choice(
        np.array([0.0, 3.95]), size=int(mid.sum()), p=[0.45, 0.55]
    )

    high = (returned_flag == 0) & (revenue_for_shipping >= 35)
    shipping[high] = rng.choice(
        np.array([0.0, 3.95, 7.95, 9.95, 12.95, 49.95]),
        size=int(high.sum()),
        p=[0.30, 0.36, 0.23, 0.06, 0.04965, 0.00035],
    )
    lines["shipping_cost"] = shipping

    print("Calibrating Q1 totals...")
    for quarter, targets in Q1_TARGETS.items():
        mask = (line_quarters == quarter)
        _calibrate_quarter(
            lines,
            mask,
            targets["category"],
            targets["channel"],
        )

    # Convert dates to text so the deliberately invalid dates survive the CSV.
    lines["order_date"] = lines["order_date"].dt.strftime("%Y-%m-%d")

    # Invalid dates live outside Q1 2025/2026 so they do not disturb the
    # calibrated totals. They are excluded from the duplicate sample.
    candidates = np.flatnonzero(~np.isin(line_quarters, ["2025Q1", "2026Q1"]))
    invalid_idx = rng.choice(candidates, size=N_INVALID_DATES, replace=False)
    invalid_values = [
        "not available",
        "2026-02-30",
        "2026-13-01",
        "2024-04-31",
        "2024-02-30",
        "unknown",
        "2025-00-10",
        "2025-11-31",
        "2024-06-31",
        "2025-02-29",
        "no-date",
        "2024-09-31",
        "2025-04-31",
        "2024-12-32",
    ]
    lines.loc[invalid_idx, "order_date"] = invalid_values

    # Make sure the 487,936 unique lines really are unique before adding the
    # intentional duplicate rows. Only the later copy of each identical pair is
    # nudged; nudging the whole group keeps the pair identical to each other.
    attempts = 0
    while lines.duplicated().any() and attempts < 20:
        dup_mask = lines.duplicated(keep="first")
        bump = 0.01 * (attempts + 1)
        lines.loc[dup_mask, "unit_price"] = (
            lines.loc[dup_mask, "unit_price"] + bump
        ).round(2)
        lines.loc[dup_mask, "revenue"] = (
            lines.loc[dup_mask, "quantity"]
            * lines.loc[dup_mask, "unit_price"]
            * (1 - lines.loc[dup_mask, "discount_pct"].fillna(0) / 100)
        ).round(2)
        attempts += 1

    print("Adding migration duplicates...")
    duplicate_candidates = np.setdiff1d(
        np.arange(len(lines)), invalid_idx, assume_unique=True
    )
    duplicate_idx = rng.choice(
        duplicate_candidates, size=N_DUPLICATES, replace=False
    )
    duplicates = lines.iloc[duplicate_idx].copy()

    raw = pd.concat([lines, duplicates], ignore_index=True)
    raw = raw.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    raw = raw[
        [
            "order_id",
            "order_date",
            "customer_id",
            "channel",
            "region",
            "category",
            "product_id",
            "quantity",
            "unit_price",
            "discount_pct",
            "shipping_cost",
            "returned",
            "revenue",
            "store_id",
        ]
    ]

    assert len(raw) == N_ROWS
    assert raw["order_id"].nunique() == N_ORDERS
    assert int(raw.duplicated().sum()) == N_DUPLICATES

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUT_PATH, index=False, na_rep="", float_format="%.2f")

    print(f"\nWrote {OUT_PATH}")
    print(f"  rows:      {len(raw):,}")
    print(f"  columns:   {len(raw.columns)}")
    print(f"  orders:    {raw['order_id'].nunique():,}")
    print(f"  duplicates:{int(raw.duplicated().sum()):,}")
    print(f"  size:      {OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    # Verification is run on the de-duplicated, date-valid data, matching the
    # Chapter 1 analysis sequence.
    clean = raw.drop_duplicates().copy()
    clean["order_date"] = pd.to_datetime(clean["order_date"], errors="coerce")
    clean = clean.dropna(subset=["order_date"])
    _verify(clean)


if __name__ == "__main__":
    main()
