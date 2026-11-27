#!/usr/bin/env python3
"""Print the exact output blocks used in Chapter 1, from the current CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"

pd.set_option("display.width", 200)

orders = pd.read_csv(DATA_RAW / "kestrel_orders.csv")
print("SHAPE")
print(orders.shape)
print("\nDTYPES")
print(orders.dtypes)
print("\nHEAD")
print(orders.head().to_string())
print("\nMISSING")
print(orders.isna().sum().sort_values(ascending=False).head(8))
print("\nDESCRIBE")
print(orders.describe(include="number").T)
print("\nGRAIN")
print("Rows:", len(orders))
print("Unique order IDs:", orders["order_id"].nunique())
print("Exact duplicate rows:", orders.duplicated().sum())

orders = orders.drop_duplicates().copy()
print("\nAFTER DEDUPE")
print("Rows after removing exact duplicates:", len(orders))

orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
bad = orders["order_date"].isna().sum()
print("\nDATES")
print("Unparseable dates:", bad)
print("First month:", orders["order_date"].min())
print("Last month:", orders["order_date"].max())
orders = orders.dropna(subset=["order_date"]).copy()
print("Rows after dropping invalid dates:", len(orders))

orders["order_month"] = orders["order_date"].dt.to_period("M")
orders["quarter"] = orders["order_date"].dt.to_period("Q")

orders["return_value"] = orders["revenue"] * orders["returned"]
orders["net_revenue"] = (
    orders["revenue"] - orders["return_value"] - orders["shipping_cost"]
)
print("\nNET REVENUE DESCRIBE")
print(
    orders[["revenue", "return_value", "shipping_cost", "net_revenue"]]
    .describe()
    .T
)
print("\nNEGATIVE LINES")
print("Lines with negative net revenue:", (orders["net_revenue"] < 0).sum())
print(
    "Share of net revenue from returned lines:",
    orders["return_value"].sum() / orders["revenue"].sum(),
)

monthly = orders.groupby("order_month").agg(
    net_revenue=("net_revenue", "sum"),
    orders=("order_id", "nunique"),
    lines=("order_id", "size"),
    return_rate=("returned", "mean"),
)
print("\nMONTHLY")
print(monthly.round(2))

q1_2025 = orders[orders["quarter"].astype(str) == "2025Q1"]
q1_2026 = orders[orders["quarter"].astype(str) == "2026Q1"]
print("\nQ1 SUMMARY")
print(f"Q1 2025 net revenue: ${q1_2025['net_revenue'].sum():,.0f}")
print(f"Q1 2026 net revenue: ${q1_2026['net_revenue'].sum():,.0f}")
change = (q1_2026["net_revenue"].sum() / q1_2025["net_revenue"].sum() - 1) * 100
print(f"Change: {change:+.1f}%")
print("\nQ1 ORDERS AND RETURN RATE")
for name, frame in [("2025Q1", q1_2025), ("2026Q1", q1_2026)]:
    print(
        name,
        frame["order_id"].nunique(),
        round(frame["returned"].mean(), 3),
    )

q1 = orders[orders["quarter"].astype(str).isin(["2025Q1", "2026Q1"])]
by_channel = (
    q1.groupby(["channel", "quarter"])["net_revenue"].sum().unstack("quarter")
)
by_channel.columns = by_channel.columns.astype(str)
by_channel["change"] = by_channel["2026Q1"] - by_channel["2025Q1"]
by_channel["pct_change"] = by_channel["change"] / by_channel["2025Q1"] * 100
by_channel["share_of_gap"] = by_channel["change"] / by_channel["change"].sum() * 100
print("\nBY CHANNEL")
print(by_channel.sort_values("change").round(1))

by_category = (
    q1.groupby(["category", "quarter"])["net_revenue"].sum().unstack("quarter")
)
by_category.columns = by_category.columns.astype(str)
by_category["change"] = by_category["2026Q1"] - by_category["2025Q1"]
by_category["pct_change"] = by_category["change"] / by_category["2025Q1"] * 100
by_category["share_of_gap"] = by_category["change"] / by_category["change"].sum() * 100
print("\nBY CATEGORY")
print(by_category.sort_values("change").round(1))

returns_by_category = (
    q1.groupby(["category", "quarter"])["returned"].mean().unstack("quarter").mul(100)
)
returns_by_category.columns = returns_by_category.columns.astype(str)
returns_by_category["change_pp"] = (
    returns_by_category["2026Q1"] - returns_by_category["2025Q1"]
)
print("\nRETURNS BY CATEGORY")
print(returns_by_category.sort_values("change_pp", ascending=False).round(1))

aov_by_channel = (
    q1.groupby(["channel", "quarter", "order_id"])["net_revenue"]
    .sum()
    .groupby(["channel", "quarter"])
    .mean()
    .unstack("quarter")
)
aov_by_channel.columns = aov_by_channel.columns.astype(str)
aov_by_channel["pct_change"] = (
    aov_by_channel["2026Q1"] / aov_by_channel["2025Q1"] - 1
) * 100
print("\nAOV BY CHANNEL")
print(aov_by_channel.round(2))

q1 = q1.assign(
    discounted=q1["discount_pct"].fillna(0) > 0,
    discount_value=q1["discount_pct"].fillna(0),
)
mix = q1.groupby(["channel", "quarter"]).agg(
    avg_unit_price=("unit_price", "mean"),
    discount_share=("discounted", "mean"),
    avg_discount=("discount_value", "mean"),
)
print("\nMIX")
print(mix.round(2))


def q1_yoy(frame: pd.DataFrame, col: str) -> float:
    totals = (
        frame[frame["quarter"].astype(str).isin(["2025Q1", "2026Q1"])]
        .groupby("quarter")[col]
        .sum()
        .sort_index()
    )
    return float((totals.iloc[1] / totals.iloc[0] - 1) * 100)


orders["gross_revenue"] = orders["revenue"]
orders["net_of_returns"] = orders["revenue"] - orders["return_value"]
orders["half_returns"] = orders["revenue"] - 0.5 * orders["return_value"] - orders["shipping_cost"]
print("\nSENSITIVITY")
print(
    pd.Series(
        {
            "Gross revenue": q1_yoy(orders, "gross_revenue"),
            "Net of returns": q1_yoy(orders, "net_of_returns"),
            "Net of returns and shipping (base)": q1_yoy(orders, "net_revenue"),
            "Half of returns treated as exchanges": q1_yoy(orders, "half_returns"),
            "Excluding guest checkouts": q1_yoy(
                orders[orders["customer_id"].notna()], "net_revenue"
            ),
        }
    ).round(1)
)
