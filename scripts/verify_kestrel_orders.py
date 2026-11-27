#!/usr/bin/env python3
"""Print the Chapter 1 analysis checks for data/raw/kestrel_orders.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "kestrel_orders.csv"


def main() -> None:
    pd.set_option("display.width", 200)

    raw = pd.read_csv(DATA)
    print("RAW")
    print("shape:", raw.shape)
    print("unique orders:", raw["order_id"].nunique())
    print("duplicates:", int(raw.duplicated().sum()))
    print("\nmissing:")
    print(raw.isna().sum().sort_values(ascending=False).head(8).to_string())

    dates = pd.to_datetime(raw["order_date"], errors="coerce")
    print("\ndate range:", dates.min(), "to", dates.max())
    print("invalid dates:", int(dates.isna().sum()))

    df = raw.drop_duplicates().copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"]).copy()
    print("\nCLEAN")
    print("rows:", len(df))
    print("negative net revenue:", int((df["revenue"] * (1 - df["returned"]) - df["shipping_cost"] < 0).sum()))
    print("return share:", round(float((df["revenue"] * df["returned"]).sum() / df["revenue"].sum()), 4))

    df["return_value"] = df["revenue"] * df["returned"]
    df["net_revenue"] = df["revenue"] - df["return_value"] - df["shipping_cost"]
    print("\nnet revenue profile:")
    print(df[["revenue", "return_value", "shipping_cost", "net_revenue"]].describe().T.round(2).to_string())

    df["order_month"] = df["order_date"].dt.to_period("M")
    monthly = (
        df.groupby("order_month")
        .agg(
            net_revenue=("net_revenue", "sum"),
            orders=("order_id", "nunique"),
            lines=("order_id", "size"),
            return_rate=("returned", "mean"),
        )
        .sort_index()
    )
    print("\nmonthly tail:")
    print(monthly.tail(6).round(2).to_string())

    df["quarter"] = df["order_date"].dt.to_period("Q")
    quarterly = df.groupby("quarter").agg(
        net_revenue=("net_revenue", "sum"),
        orders=("order_id", "nunique"),
        return_rate=("returned", "mean"),
    )
    print("\nQ1 summary:")
    print(quarterly.loc[["2025Q1", "2026Q1"]].round(2).to_string())

    q1 = df[df["quarter"].astype(str).isin(["2025Q1", "2026Q1"])]
    aov = (
        q1.groupby(["quarter", "order_id"])["net_revenue"]
        .sum()
        .groupby("quarter")
        .mean()
    )
    print("\nAOV:", aov.round(2).to_dict())

    by_channel = (
        q1.groupby(["channel", "quarter"])["net_revenue"]
        .sum()
        .unstack("quarter")
    )
    by_channel.columns = by_channel.columns.astype(str)
    by_channel["change"] = by_channel["2026Q1"] - by_channel["2025Q1"]
    by_channel["pct_change"] = by_channel["change"] / by_channel["2025Q1"] * 100
    by_channel["share_of_gap"] = by_channel["change"] / by_channel["change"].sum() * 100
    print("\nchannel:")
    print(by_channel.round(1).to_string())

    by_category = (
        q1.groupby(["category", "quarter"])["net_revenue"]
        .sum()
        .unstack("quarter")
    )
    by_category.columns = by_category.columns.astype(str)
    by_category["change"] = by_category["2026Q1"] - by_category["2025Q1"]
    by_category["pct_change"] = by_category["change"] / by_category["2025Q1"] * 100
    by_category["share_of_gap"] = by_category["change"] / by_category["change"].sum() * 100
    print("\ncategory:")
    print(by_category.sort_values("change").round(1).to_string())

    returns = (
        q1.groupby(["category", "quarter"])["returned"]
        .mean()
        .unstack("quarter")
        .mul(100)
    )
    returns.columns = returns.columns.astype(str)
    returns["change_pp"] = returns["2026Q1"] - returns["2025Q1"]
    print("\nreturns by category:")
    print(returns.round(1).to_string())

    aov_channel = (
        q1.groupby(["channel", "quarter", "order_id"])["net_revenue"]
        .sum()
        .groupby(["channel", "quarter"])
        .mean()
        .unstack("quarter")
    )
    aov_channel.columns = aov_channel.columns.astype(str)
    aov_channel["pct_change"] = (
        aov_channel["2026Q1"] / aov_channel["2025Q1"] - 1
    ) * 100
    print("\nAOV by channel:")
    print(aov_channel.round(2).to_string())

    q1 = q1.assign(
        discounted=q1["discount_pct"].fillna(0) > 0,
        discount_value=q1["discount_pct"].fillna(0),
    )
    mix = q1.groupby(["channel", "quarter"]).agg(
        avg_unit_price=("unit_price", "mean"),
        discount_share=("discounted", "mean"),
        avg_discount=("discount_value", "mean"),
    )
    print("\nmix:")
    print(mix.round(2).to_string())

    def yoy(frame: pd.DataFrame, col: str) -> float:
        totals = (
            frame[frame["quarter"].astype(str).isin(["2025Q1", "2026Q1"])]
            .groupby("quarter")[col]
            .sum()
            .sort_index()
        )
        return float((totals.iloc[1] / totals.iloc[0] - 1) * 100)

    df["gross_revenue"] = df["revenue"]
    df["net_of_returns"] = df["revenue"] - df["return_value"]
    df["half_returns"] = df["revenue"] - 0.5 * df["return_value"] - df["shipping_cost"]

    scenarios = pd.Series(
        {
            "Gross revenue": yoy(df, "gross_revenue"),
            "Net of returns": yoy(df, "net_of_returns"),
            "Net of returns and shipping (base)": yoy(df, "net_revenue"),
            "Half of returns treated as exchanges": yoy(df, "half_returns"),
            "Excluding guest checkouts": yoy(
                df[df["customer_id"].notna()], "net_revenue"
            ),
        }
    )
    print("\nsensitivity:")
    print(scenarios.round(1).to_string())


if __name__ == "__main__":
    main()
