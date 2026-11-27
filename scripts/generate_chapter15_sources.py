#!/usr/bin/env python3
"""Generate the business-analytics fixtures that Chapter 15 uses.

Run from the project root:

    python scripts/generate_chapter15_sources.py

Outputs (all under data/raw/business/):

    biz_monthly.csv      24 months x 3 channels x 4 regions
    biz_products.csv     product performance and quarterly prices
    biz_campaigns.csv    platform-reported and warehouse-attributed conversions
    biz_customers.csv    customer-level revenue, margin and recency
    biz_events.csv       product usage events
    biz_operations.csv   12-step fulfilment process
    biz_finance.csv      budget vs actual with margin and opex

Everything is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "business"
SEED = 1507


def monthly() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    rows = []
    channels = [("web", 0.55, 0.118), ("app", 0.30, 0.098), ("phone", 0.15, 0.082)]
    regions = ["Midwest", "Northeast", "South", "West"]
    for i, m in enumerate(months):
        season = 1 + 0.15 * np.sin(2 * np.pi * (m.month - 3) / 12)
        growth = 1 + 0.006 * i
        for ch, share, cvr in channels:
            for reg in regions:
                sessions = int(48000 * share / 4 * growth * season
                               * float(rng.normal(1, 0.05)))
                orders = int(sessions * cvr * float(rng.normal(1, 0.04)))
                aov = 172 * (1 + 0.003 * i) * float(rng.normal(1, 0.03))
                rev = round(orders * aov, 2)
                cogs = round(rev * float(rng.uniform(0.58, 0.66)), 2)
                returns = round(rev * float(rng.uniform(0.04, 0.09)), 2)
                shipping = round(orders * float(rng.uniform(3.5, 6.0)), 2)
                rows.append((m.strftime("%Y-%m"), ch, reg, sessions, orders,
                             round(aov, 2), rev, cogs, returns, shipping,
                             round(rev - cogs - returns - shipping, 2)))
    return pd.DataFrame(rows, columns=[
        "month", "channel", "region", "sessions", "orders", "aov",
        "revenue", "cogs", "returns", "shipping", "margin"])


def products(m: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    cats = {"Camping": (8, 180.0), "Footwear": (7, 120.0),
            "Apparel": (9, 85.0), "Accessories": (6, 45.0)}
    rows = []
    pid = 0
    for cat, (count, base) in cats.items():
        for j in range(count):
            pid += 1
            price = round(base * float(rng.uniform(0.85, 1.20)), 2)
            cost = round(price * float(rng.uniform(0.52, 0.68)), 2)
            units = int(rng.integers(90, 900))
            q4_units = int(units * float(rng.uniform(0.75, 1.35)))
            rows.append((f"P-{pid:04d}", f"{cat} item {j+1}", cat, price,
                         cost, units, q4_units, round(units * price, 2),
                         round(q4_units * price * 1.04, 2)))
    return pd.DataFrame(rows, columns=[
        "product_id", "product_name", "category", "price", "unit_cost",
        "units_q3", "units_q4", "revenue_q3", "revenue_q4"])


def campaigns() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    months = pd.date_range("2025-01-01", periods=12, freq="MS").strftime("%Y-%m")
    rows = []
    for platform, base_cvr, cpc in [
        ("Meta", 0.021, 1.35), ("Google", 0.031, 2.10), ("TikTok", 0.014, 0.95)
    ]:
        for m in months:
            spend = round(float(rng.uniform(18000, 42000)), 2)
            clicks = int(spend / cpc)
            platform_conv = int(clicks * base_cvr * float(rng.normal(1, 0.08)))
            first = int(platform_conv * float(rng.uniform(0.62, 0.78)))
            last = int(platform_conv * float(rng.uniform(0.55, 0.72)))
            linear = round((first + last) / 2)
            rows.append((m, platform, spend, clicks, platform_conv,
                         first, last, linear))
    return pd.DataFrame(rows, columns=[
        "month", "platform", "spend", "clicks", "platform_conversions",
        "first_touch", "last_touch", "linear_touch"])


def customers() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = 4000
    signup = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 900, size=n), unit="D")
    orders = rng.poisson(3.2, size=n) + 1
    aov = np.round(rng.lognormal(4.9, 0.45, size=n), 2)
    revenue = np.round(orders * aov, 2)
    margin = np.round(revenue * rng.uniform(0.22, 0.38, size=n), 2)
    last = pd.Series(signup) + pd.to_timedelta(
        rng.integers(0, 700, size=n), unit="D")
    last = last.clip(upper=pd.Timestamp("2025-12-31"))
    return pd.DataFrame({
        "customer_id": [f"C-{i:05d}" for i in range(1, n + 1)],
        "signup_date": signup.strftime("%Y-%m-%d"),
        "segment": rng.choice(["Basic", "Plus", "Premium"], size=n,
                              p=[0.5, 0.32, 0.18]),
        "region": rng.choice(["Midwest", "Northeast", "South", "West"], size=n),
        "orders": orders,
        "revenue": revenue,
        "margin": margin,
        "last_order_date": last.dt.strftime("%Y-%m-%d"),
        "nps": rng.integers(0, 11, size=n),
        "support_tickets": rng.poisson(0.8, size=n),
    })


def events() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n_users = 2000
    rows = []
    names = ["login", "view_product", "add_to_cart", "checkout", "purchase",
             "share", "use_filter"]
    probs = [0.95, 0.80, 0.52, 0.34, 0.26, 0.12, 0.31]
    for u in range(n_users):
        depth = int(np.argmax(rng.random(len(probs)) > probs) + 1) if (
            rng.random(len(probs)) > probs).any() else len(probs)
        for step in names[:max(1, min(depth, len(names)))]:
            for _ in range(int(rng.integers(1, 4))):
                rows.append((f"U-{u:05d}", step,
                             pd.Timestamp("2025-06-01")
                             + pd.DateOffset(days=int(rng.integers(0, 180))),
                             rng.choice(["ios", "android", "web"])))
    df = pd.DataFrame(rows, columns=["user_id", "event", "event_date", "platform"])
    df["event_date"] = df["event_date"].dt.strftime("%Y-%m-%d")
    return df


def operations() -> pd.DataFrame:
    steps = [
        ("order_received", 4, 8), ("payment_check", 6, 5), ("fraud_screen", 9, 12),
        ("pick_items", 18, 22), ("quality_check", 7, 9), ("pack", 11, 14),
        ("label", 5, 6), ("carrier_pickup", 26, 31), ("transit", 240, 8),
        ("customs", 35, 18), ("last_mile", 62, 15), ("delivered", 6, 4),
    ]
    return pd.DataFrame([
        (i + 1, name, mins, defect, round(mins * 0.42, 2))
        for i, (name, mins, defect) in enumerate(steps)
    ], columns=["step", "step_name", "avg_minutes", "defects_per_1000",
                "cost_per_order"])


def finance(m: pd.DataFrame) -> pd.DataFrame:
    agg = m.groupby("month").agg(revenue=("revenue", "sum"),
                                 orders=("orders", "sum"),
                                 cogs=("cogs", "sum"),
                                 margin=("margin", "sum")).reset_index()
    rng = np.random.default_rng(SEED)
    agg["budget_revenue"] = (agg["revenue"] * rng.uniform(0.94, 1.06,
                                                          size=len(agg))).round(2)
    agg["budget_orders"] = (agg["orders"] * rng.uniform(0.95, 1.05,
                                                        size=len(agg))).round(0).astype(int)
    agg["opex"] = (agg["revenue"] * 0.19).round(2)
    agg["gross_margin_pct"] = ((agg["revenue"] - agg["cogs"]) / agg["revenue"]).round(4)
    return agg


def cohort_activity() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    months = pd.date_range("2023-01-01", periods=24, freq="MS").strftime("%Y-%m")
    rows = []
    for i, cohort in enumerate(months):
        size = int(rng.integers(80, 180))
        for k in range(12):
            base = 0.96 * np.exp(-0.16 * k)
            active = int(round(size * base * float(rng.normal(1, 0.05))))
            rows.append((cohort, size, k, active))
    return pd.DataFrame(rows, columns=[
        "cohort", "cohort_size", "months_since_signup", "active_customers"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("generating Chapter 15 business fixtures ...")
    m = monthly()
    m.to_csv(OUT / "biz_monthly.csv", index=False)
    cohort_activity().to_csv(OUT / "biz_cohort_activity.csv", index=False)
    products(m).to_csv(OUT / "biz_products.csv", index=False)
    campaigns().to_csv(OUT / "biz_campaigns.csv", index=False)
    customers().to_csv(OUT / "biz_customers.csv", index=False)
    events().to_csv(OUT / "biz_events.csv", index=False)
    operations().to_csv(OUT / "biz_operations.csv", index=False)
    finance(m).to_csv(OUT / "biz_finance.csv", index=False)
    for p in sorted(OUT.glob("biz_*.csv")):
        print(f"  {p.name}: {len(pd.read_csv(p))} rows")
    print("done.")


if __name__ == "__main__":
    main()
