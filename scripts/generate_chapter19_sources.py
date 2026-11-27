#!/usr/bin/env python3
"""Generate the Chapter 19 project data and figures.

Run from the project root:

    python scripts/generate_chapter19_sources.py

Outputs (under data/raw/project19/ except figures):

    kestrel_project.db       six project tables in SQLite
    marketing_spend.csv      monthly spend by channel
    finance_revenue.csv      finance's reported net revenue by quarter

    images/ch19_*.png        eight figures used in the chapter

Everything is deterministic. The project is a two-quarter comparison with a
deliberate 12-week extraction gap in the prior year, a late finance adjustment,
and a few planted data-quality defects.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "data" / "raw" / "project19"
IMAGES = ROOT / "images"
SEED = 1907

REGIONS = ["Midwest", "Northeast", "South", "West"]
CHANNELS = ["Web", "App", "Phone"]
CATS = ["Camping", "Footwear", "Apparel", "Accessories"]
BASE_PRICE = {"Camping": 178.0, "Footwear": 122.0,
              "Apparel": 86.0, "Accessories": 46.0}


def products(rng) -> pd.DataFrame:
    rows = []
    pid = 0
    for cat in CATS:
        for j in range(10):
            pid += 1
            price = round(BASE_PRICE[cat] * float(rng.uniform(0.85, 1.18)), 2)
            cost = round(price * float(rng.uniform(0.50, 0.66)), 2)
            rows.append((f"P-{pid:04d}", f"{cat} item {j + 1}",
                         cat, price, cost))
    return pd.DataFrame(rows, columns=[
        "product_id", "product_name", "category", "list_price", "unit_cost"])


def customers(rng, n=8000) -> pd.DataFrame:
    signup = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 1000, n), unit="D")
    return pd.DataFrame({
        "customer_id": [f"C-{i + 1:05d}" for i in range(n)],
        "signup_date": signup.strftime("%Y-%m-%d"),
        "loyalty_tier": rng.choice(["Basic", "Plus", "Premium"], n,
                                   p=[0.50, 0.32, 0.18]),
        "home_region": rng.choice(REGIONS, n),
    })


def stores() -> pd.DataFrame:
    cities = ["Chicago", "Denver", "Boston", "Atlanta", "Seattle", "Austin",
              "Portland", "Phoenix", "Detroit", "Miami", "Minneapolis", "Reno"]
    return pd.DataFrame({
        "store_id": [f"S-{i + 1:02d}" for i in range(12)],
        "city": cities,
        "region": REGIONS * 3,
        "opened": ["2019-01-01"] * 12,
    })


def make_orders(rng, products_df, customers_df, period, start, weeks,
                order_count, price_mult, disc_shift, return_shift) -> tuple:
    prod = products_df.to_dict("records")
    cust_ids = customers_df["customer_id"].to_numpy()
    order_rows, line_rows, return_rows = [], [], []
    oid = 0
    lid = 0
    rid = 0
    for w in range(weeks):
        wd = pd.Timestamp(start) + pd.Timedelta(days=7 * int(w))
        season = 1 + 0.10 * np.sin(2 * np.pi * (wd.dayofyear - 20) / 365)
        promo = 1.22 if w in (3, 9) else 1.0
        for _ in range(int(order_count / 13 * season * promo)):
            oid += 1
            oid_str = f"O-{period[:4]}-{oid:06d}"
            cid = str(rng.choice(cust_ids))
            ch = str(rng.choice(CHANNELS, p=[0.55, 0.30, 0.15]))
            reg = str(rng.choice(REGIONS))
            store = f"S-{REGIONS.index(reg) + 1:02d}"
            date = (wd + pd.Timedelta(days=int(rng.integers(0, 7))))
            nlines = int(rng.integers(1, 5))
            gross = 0.0
            disc_total = 0.0
            ship = round(float(rng.uniform(3.5, 7.5)), 2)
            for _ in range(nlines):
                p = prod[int(rng.integers(0, len(prod)))]
                qty = int(rng.integers(1, 4))
                price = round(p["list_price"] * price_mult
                              * float(rng.normal(1, 0.04)), 2)
                disc = float(np.clip(rng.normal(8 + disc_shift, 9), 0, 60))
                lr = round(qty * price * (1 - disc / 100), 2)
                gross += lr
                disc_total += qty * price * disc / 100
                lid += 1
                line_rows.append((f"L-{period[:4]}-{lid:07d}", oid_str,
                                  p["product_id"], qty, price,
                                  round(disc, 2), lr, p["unit_cost"] * price_mult))
            ret_flag = int(rng.random() < (0.055 + return_shift))
            order_rows.append((oid_str, date.strftime("%Y-%m-%d"),
                               cid, ch, reg, store, round(gross, 2),
                               round(disc_total, 2), ship, ret_flag, period))
            if ret_flag:
                frac = float(rng.choice([0.35, 0.65, 1.0],
                                        p=[0.25, 0.25, 0.50]))
                rid += 1
                return_rows.append((f"R-{rid:06d}", oid_str,
                                    round(gross * frac, 2),
                                    round(frac, 2),
                                    (date + pd.Timedelta(days=12)).strftime("%Y-%m-%d"),
                                    str(rng.choice(["size", "damage", "changed_mind"])),
                                    period))
    return (pd.DataFrame(order_rows, columns=[
                "order_id", "order_date", "customer_id", "channel", "region",
                "store_id", "gross_revenue", "discount_amount", "shipping_cost",
                "returned", "period"]),
            pd.DataFrame(line_rows, columns=[
                "line_id", "order_id", "product_id", "quantity", "unit_price",
                "discount_pct", "line_revenue", "unit_cost"]),
            pd.DataFrame(return_rows, columns=[
                "return_id", "order_id", "refund_amount", "return_fraction",
                "return_date", "reason", "period"]))


def main() -> None:
    PROJ.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("generating Chapter 19 project data ...")

    p = products(rng)
    cu = customers(rng)
    st = stores()

    # prior year has a 12-week extract, current year a full 13 weeks
    o0, l0, r0 = make_orders(rng, p, cu, "2025Q1", "2025-01-06", 12,
                             12400, 1.00, 0.0, 0.0)
    o1, l1, r1 = make_orders(rng, p, cu, "2026Q1", "2026-01-05", 13,
                             12000, 0.947, 5.0, 0.008)

    orders = pd.concat([o0, o1], ignore_index=True)
    lines = pd.concat([l0, l1], ignore_index=True)
    returns = pd.concat([r0, r1], ignore_index=True)

    # scale prices so prior-year net revenue is close to the case-study figure
    net0 = (o0["gross_revenue"].sum() - r0["refund_amount"].sum()
            - o0["shipping_cost"].sum())
    scale = 4392118.0 / net0
    for frame, col in [(lines, "unit_price"), (lines, "unit_cost")]:
        frame[col] = np.round(frame[col] * scale, 2)
    for frame, col in [(orders, "gross_revenue"), (returns, "refund_amount")]:
        frame[col] = np.round(frame[col] * scale, 2)
    lines["line_revenue"] = np.round(
        lines["quantity"] * lines["unit_price"]
        * (1 - lines["discount_pct"] / 100), 2)

    # planted defects: a few duplicate order keys and missing costs
    dup = orders.sample(60, random_state=SEED).copy()
    dup["order_id"] = dup["order_id"] + "-D"
    orders = pd.concat([orders, dup], ignore_index=True)
    miss = lines.sample(120, random_state=SEED).copy()
    miss["unit_cost"] = np.nan
    lines = pd.concat([lines, miss], ignore_index=True)
    lines.loc[lines.index[-120:], "line_id"] = [
        f"L-X-{i:07d}" for i in range(120)]

    db = PROJ / "kestrel_project.db"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    for name, frame in [
        ("kestrel_orders", orders), ("kestrel_order_lines", lines),
        ("kestrel_products", p), ("kestrel_customers", cu),
        ("kestrel_stores", st), ("kestrel_returns", returns),
    ]:
        frame.to_sql(name, con, index=False)
    con.close()

    spend_rows = []
    for period, months in [("2025Q1", ["2025-01", "2025-02", "2025-03"]),
                           ("2026Q1", ["2026-01", "2026-02", "2026-03"])]:
        for m in months:
            for ch, base in [("Search", 42000), ("Social", 31000),
                             ("Display", 22000), ("Email", 9000)]:
                mult = 1.06 if period == "2026Q1" else 1.0
                spend_rows.append((m, ch, round(base * mult
                                                 * float(rng.uniform(0.94, 1.06)), 2)))
    spend = pd.DataFrame(spend_rows, columns=["month", "channel", "spend"])
    spend.to_csv(PROJ / "marketing_spend.csv", index=False)

    net0 = round(float(o0["gross_revenue"].sum() * scale
                       - r0["refund_amount"].sum() * scale
                       - o0["shipping_cost"].sum()), 2)
    net1 = round(float(o1["gross_revenue"].sum() * scale
                       - r1["refund_amount"].sum() * scale
                       - o1["shipping_cost"].sum()), 2)
    fin = pd.DataFrame([
        ("2025Q1", net0, "finance ledger"),
        ("2026Q1", round(net1 + 12450.0, 2),
         "finance ledger; +$12,450 post-close adjustment"),
    ], columns=["period", "reported_net_revenue", "note"])
    fin.to_csv(PROJ / "finance_revenue.csv", index=False)

    print(f"  kestrel_project.db: {len(orders)} orders, "
          f"{len(lines)} lines, {len(returns)} returns")
    print(f"  Q1 2025 net {net0:,.2f}; Q1 2026 net {net1:,.2f}; "
          f"change {net1 / net0 - 1:.4f}")
    figures(orders, lines, returns, spend)
    for fp in sorted(IMAGES.glob("ch19_*.png")):
        print(f"  {fp.name}: {fp.stat().st_size:,} bytes")
    print("done.")


def figures(orders, lines, returns, spend) -> None:
    net = orders.groupby("period").apply(
        lambda g: g["gross_revenue"].sum() - g["shipping_cost"].sum()
        - returns.loc[returns["order_id"].isin(g["order_id"]),
                      "refund_amount"].sum(), include_groups=False)
    periods = list(net.index)

    # 1. monthly net revenue
    mm = orders.copy()
    mm["month"] = mm["order_date"].str[:7]
    rev = mm.groupby("month")["gross_revenue"].sum() / 1e6
    f, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(rev.index, rev.values, marker="o", lw=2, color="#c0392b")
    ax.set(title="Monthly gross revenue", ylabel="$m")
    ax.tick_params(axis="x", rotation=45)
    f.tight_layout(); f.savefig(IMAGES / "ch19_monthly.png", dpi=130); plt.close(f)

    # 2. bridge
    vals = [float(net.iloc[0]), -180000, 95000, 42000, float(net.iloc[1])]
    labels = ["2025Q1", "Volume", "Price", "Mix", "2026Q1"]
    f, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(labels, vals, color=["#2c3e50", "#c0392b", "#27ae60", "#f39c12", "#2c3e50"])
    ax.set(title="Price–volume–mix bridge (illustrative)", ylabel="$")
    f.tight_layout(); f.savefig(IMAGES / "ch19_bridge.png", dpi=130); plt.close(f)

    # 3. region contribution
    reg = orders.groupby(["period", "region"])["gross_revenue"].sum().unstack()
    delta = reg.loc["2026Q1"] - reg.loc["2025Q1"]
    f, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(delta.index, delta.values / 1000, color="#8e44ad")
    ax.axhline(0, color="k", lw=1)
    ax.set(title="Revenue change by region ($k)", ylabel="$k")
    f.tight_layout(); f.savefig(IMAGES / "ch19_regions.png", dpi=130); plt.close(f)

    # 4. discount margin curve
    d = lines.copy()
    d["band"] = pd.cut(d["discount_pct"], [-1, 5, 15, 30, 60],
                       labels=["0-5", "6-15", "16-30", "31-60"])
    margin = d.groupby("band", observed=True).apply(
        lambda g: (g["line_revenue"].sum()
                   - (g["unit_price"] - g["unit_cost"])
                   * g["quantity"]).sum() / g["line_revenue"].sum(),
        include_groups=False)
    f, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot([str(x) for x in margin.index], margin.values, marker="o", lw=2)
    ax.set(title="Margin rate by discount band", ylabel="margin")
    f.tight_layout(); f.savefig(IMAGES / "ch19_discount.png", dpi=130); plt.close(f)

    # 5. promo lift and pull-forward
    wk = orders.copy()
    wk["week"] = pd.to_datetime(wk["order_date"]).dt.isocalendar().week
    daily = wk.groupby("order_date")["gross_revenue"].sum()
    f, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(daily.index, daily.values, lw=1)
    ax.set(title="Daily revenue with two promotion weeks", ylabel="$")
    ax.tick_params(axis="x", rotation=45)
    f.tight_layout(); f.savefig(IMAGES / "ch19_promo.png", dpi=130); plt.close(f)

    # 6. returns contribution
    rr = returns.groupby("period")["refund_amount"].sum()
    f, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.bar(rr.index, rr.values / 1000, color="#c0392b")
    ax.set(title="Returns by quarter ($k)", ylabel="$k")
    f.tight_layout(); f.savefig(IMAGES / "ch19_returns.png", dpi=130); plt.close(f)

    # 7. product pareto
    prod = lines.groupby("product_id")["line_revenue"].sum().sort_values(ascending=False)
    cum = prod.cumsum() / prod.sum()
    f, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(np.arange(1, len(cum) + 1) / len(cum), cum.values, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(title="Product revenue concentration", xlabel="share of products",
           ylabel="share of revenue")
    f.tight_layout(); f.savefig(IMAGES / "ch19_pareto.png", dpi=130); plt.close(f)

    # 8. four-chart executive pack
    f, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(rev.index, rev.values, color="#c0392b"); axes[0, 0].set_title("Revenue trend")
    axes[0, 1].bar(delta.index, delta.values / 1000); axes[0, 1].set_title("Regional change")
    axes[1, 0].plot([str(x) for x in margin.index], margin.values, marker="o"); axes[1, 0].set_title("Discount vs margin")
    axes[1, 1].plot(np.arange(1, len(cum) + 1) / len(cum), cum.values); axes[1, 1].set_title("Product concentration")
    for a in axes.ravel():
        a.tick_params(axis="x", rotation=45, labelsize=7)
    f.suptitle("Executive visual pack")
    f.tight_layout(); f.savefig(IMAGES / "ch19_pack.png", dpi=130); plt.close(f)


if __name__ == "__main__":
    main()
