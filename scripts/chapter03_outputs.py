#!/usr/bin/env python3
"""Print every output block used in Chapter 3, from the current files.

Run from the project root:

    python scripts/chapter03_outputs.py

The script is deliberately linear so that a reader can match each labelled
block to a block in the chapter.
"""

from __future__ import annotations

import timeit
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 12)
pd.set_option("display.max_rows", 30)


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


ORDERS = RAW / "kestrel_orders.csv"


def load_clean() -> pd.DataFrame:
    """Load, de-duplicate, and parse the orders file the way Chapter 1 did."""
    df = pd.read_csv(
        ORDERS,
        dtype={"customer_id": "string", "region": "category"},
    )
    df = df.drop_duplicates().copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"]).copy()
    df["return_value"] = df["revenue"] * df["returned"]
    df["net_revenue"] = df["revenue"] - df["return_value"] - df["shipping_cost"]
    return df


# ---------------------------------------------------------------- 3.1 NumPy
def section_31() -> None:
    a = np.array([10.0, 20.0, 30.0, 40.0])
    show("array from list", a)
    show("arange", np.arange(0, 10, 2))
    show("linspace", np.linspace(0, 1, 5))
    show("zeros_ones_full", (np.zeros(3), np.ones(3), np.full(3, 7.0)))
    show("shape ndim size dtype", (a.shape, a.ndim, a.size, a.dtype))

    m = np.arange(1, 13).reshape(3, 4)
    show("matrix", m)
    show("m[1, 2]", m[1, 2])
    show("m[:, 0]", m[:, 0])
    show("m[0, :]", m[0, :])
    show("m[::-1, ::2]", m[::-1, ::2])
    show("m[-1, -1]", m[-1, -1])

    prices = np.array([59.0, 89.0, 1299.0, 45.0, 349.0])
    mask = prices > 100
    show("boolean mask", mask)
    show("prices[mask]", prices[mask])
    show("fancy", prices[[0, 4, 2]])
    show("np.where", np.where(prices > 100, "high", "low"))
    show(
        "np.select",
        np.select([prices > 500, prices > 100], ["premium", "mid"], "budget"),
    )

    base = np.arange(5)
    view = base[1:4]
    view[0] = 99
    show("view changes base", base)
    copy = base[1:4].copy()
    copy[0] = -1
    show("copy leaves base", base)


# ------------------------------------------------- 3.2 Vectorization
def section_32() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    show("elementwise", (a + b, a * b, b / a, a ** 2))
    show("comparisons", (a > 1.5, int((a > 1.5).sum())))

    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    row = np.array([10.0, 20.0, 30.0])
    col = np.array([[100.0], [200.0]])
    show("m shape", m.shape)
    show("row broadcast", m + row)
    show("col broadcast", m + col)

    x = np.array([[59.0, 89.0, 1299.0], [45.0, 349.0, 129.0]])
    centered = x - x.mean(axis=0)
    standardized = centered / x.std(axis=0)
    show("mean axis0", x.mean(axis=0).round(2))
    show("centered", centered.round(2))
    show("standardized", standardized.round(2))

    values = np.arange(1_000_000, dtype=float)
    loop_time = timeit.timeit(lambda: [v * 2 for v in values.tolist()], number=5)
    vec_time = timeit.timeit(lambda: values * 2, number=5)
    show(
        "loop vs vectorized (seconds, varies)",
        f"loop: {loop_time:.3f}s  vectorized: {vec_time:.4f}s",
    )


# ------------------------------------------- 3.3 NumPy statistics
def section_33() -> None:
    net = np.array([120.50, 89.00, 245.75, 0.00, 310.00])
    show("sum mean", (float(net.sum()), round(float(net.mean()), 2)))
    show("std var", (round(float(net.std()), 2), round(float(net.var()), 2)))
    show("median percentile", (float(np.median(net)), float(np.percentile(net, 90))))
    show("argmin argmax", (int(net.argmin()), int(net.argmax())))
    show("cumsum", net.cumsum())
    show("diff", np.diff(net))
    show("sort", np.sort(net))

    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    show("axis0", m.sum(axis=0))
    show("axis1", m.sum(axis=1))
    show("total", m.sum())

    with_nan = np.array([1.0, np.nan, 3.0])
    show("nan sum", with_nan.sum())
    show("nan-safe", (float(np.nansum(with_nan)), float(np.nanmean(with_nan))))

    rng = np.random.default_rng(42)
    draw = rng.integers(0, 100, size=5)
    show("seeded draw", draw)
    show("seeded draw again", np.random.default_rng(42).integers(0, 100, size=5))

    price = np.array([100.0, 120.0, 150.0, 180.0])
    units = np.array([10.0, 12.0, 15.0, 18.0])
    show("dot", np.dot(price, units))
    show("matmul", np.array([[1, 2], [3, 4]]) @ np.array([[5, 6], [7, 8]]))

    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.1, 3.9, 6.2, 7.8])
    slope, intercept = np.polyfit(x, y, 1)
    show("least squares", (round(float(slope), 3), round(float(intercept), 3)))


# ----------------------------------------------- 3.4 Series / DataFrame
def section_34() -> None:
    s = pd.Series([185.78, 179.99, 190.93], index=["2026-01", "2026-02", "2026-03"])
    show("series", s)
    show("series.values", s.values)
    show("series.index", s.index)

    df = pd.DataFrame(
        {
            "channel": ["Web", "App", "Phone"],
            "net": [2702000.0, 1318000.0, 372118.0],
        }
    )
    show("frame", df)
    show("frame.dtypes", df.dtypes)
    show("df[channel]", df["channel"])
    show("df.net", df.net)

    records = [{"channel": "Web", "net": 100.0}, {"channel": "App", "net": 200.0}]
    show("from records", pd.DataFrame(records))
    show("from arrays", pd.DataFrame(np.arange(6).reshape(3, 2), columns=["a", "b"]))

    left = pd.Series([1.0, 2.0], index=["Web", "App"])
    right = pd.Series([10.0, 20.0], index=["App", "Phone"])
    show("aligned add", left + right)

    a = pd.DataFrame({"net": [100.0, 200.0]})
    b = a
    c = a.copy()
    b.loc[0, "net"] = 999.0
    show("assignment shares", (a["net"].tolist(), c["net"].tolist()))


# --------------------------------------------- 3.5 Loading and saving
def section_35() -> None:
    head = pd.read_csv(ORDERS, nrows=3, usecols=["order_id", "channel", "revenue"])
    show("usecols nrows", head)

    typed = pd.read_csv(
        ORDERS,
        nrows=5,
        usecols=["customer_id", "region", "order_date", "quantity"],
        dtype={"customer_id": "string", "region": "category", "quantity": "int8"},
        parse_dates=["order_date"],
    )
    show("typed dtypes", typed.dtypes)

    weird = pd.read_csv(
        ORDERS, nrows=5, na_values=["", "none", "n/a"], keep_default_na=True
    )
    show("na_values shape", weird.shape)

    small = load_clean().head(200).copy()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    csv_path = PROCESSED / "_ch03_sample.csv"
    parquet_path = PROCESSED / "_ch03_sample.parquet"
    small.to_csv(csv_path, index=False)
    small.to_parquet(parquet_path, index=False)
    show(
        "sizes (bytes)",
        {
            "csv": csv_path.stat().st_size,
            "parquet": parquet_path.stat().st_size,
        },
    )
    back = pd.read_parquet(parquet_path, columns=["order_id", "net_revenue"])
    show("parquet columns", list(back.columns))
    csv_path.unlink()
    parquet_path.unlink()

    xlsx_path = PROCESSED / "_ch03_sample.xlsx"
    small.head(100).to_excel(xlsx_path, index=False, sheet_name="orders")
    with pd.ExcelFile(xlsx_path) as workbook:
        show("excel sheets", workbook.sheet_names)
        show(
            "excel read",
            pd.read_excel(workbook, usecols=["order_id", "net_revenue"]).head(2),
        )
    xlsx_path.unlink()

    json_path = PROCESSED / "_ch03_sample.json"
    small.head(2)[["order_id", "channel", "net_revenue"]].to_json(
        json_path, orient="records", lines=True
    )
    show("json lines", json_path.read_text(encoding="utf-8").strip().splitlines())
    show(
        "read_json",
        pd.read_json(json_path, lines=True),
    )
    json_path.unlink()


# ----------------------------------------------- 3.6 Selection / filtering
def section_36() -> None:
    df = load_clean()

    show("clean row count", len(df))
    show("loc single", df.loc[0, ["order_id", "channel", "revenue"]])
    show("iloc single", df.iloc[0, [0, 3, 12]])
    show("at", df.at[0, "channel"])
    show("iat", df.iat[0, 12])
    show("loc slice", df.loc[0:2, ["order_id", "net_revenue"]])
    show("iloc slice", df.iloc[0:2, 0:2])

    q1 = df[(df["order_date"] >= "2026-01-01") & (df["order_date"] <= "2026-03-31")]
    show("q1 row count via mask", len(q1))

    web = df[df["channel"] == "Web"]
    show("isin", int(df["region"].isin(["West", "South"]).sum()))
    show("between", int(df["revenue"].between(100, 200).sum()))
    show("query", len(df.query("channel == 'App' and quantity >= 3")))
    show("negation", int((~df["channel"].isin(["Web", "App"])).sum()))

    # filter waterfall for Q1 2026
    steps = {}
    cur = df.copy()
    steps["all clean rows"] = len(cur)
    cur = cur[cur["order_date"] >= "2026-01-01"]
    steps["from 2026-01"] = len(cur)
    cur = cur[cur["order_date"] <= "2026-03-31"]
    steps["to 2026-03-31"] = len(cur)
    cur = cur[cur["revenue"] > 0]
    steps["positive revenue"] = len(cur)
    cur = cur[cur["channel"] == "Web"]
    steps["web only"] = len(cur)
    show("filter waterfall", steps)


# ------------------------------------------- 3.7 Inspection / profiling
def section_37() -> None:
    df = load_clean()
    show("head", df.head(3)[["order_id", "order_date", "channel", "category", "revenue"]])
    show("tail", df.tail(3)[["order_id", "order_date", "channel", "revenue"]])
    show("shape", df.shape)
    show("nunique", df.nunique().sort_values().head(6))
    show("value_counts channel", df["channel"].value_counts())
    show(
        "describe revenue",
        df["revenue"].describe().round(2),
    )
    show("grain check", (df["order_id"].nunique(), len(df)))
    dup = df.duplicated(subset=["order_id", "product_id"]).sum()
    show("dup order+product", int(dup))
    show(
        "missing table",
        (df.isna().mean().mul(100).round(2).sort_values(ascending=False).head(5)),
    )


# ------------------------------------------- 3.8 Cleaning primitives
def section_38() -> None:
    s = pd.Series(["$1,250.00", "n/a", "(45.00)", "78.5", None])
    cleaned = (
        s.str.replace(r"[$,]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        .replace({"n/a": np.nan})
        .astype("float64")
    )
    show("parsed money", cleaned)

    df = pd.DataFrame(
        {"region": ["West", "west", " West ", None, "South"], "net": [1, 2, 3, 4, 5]}
    )
    normalized = df["region"].str.strip().str.title()
    show("normalized region", normalized)

    show("isna", df["region"].isna().tolist())
    show("fillna", df["region"].fillna("Unknown").tolist())
    show("dropna", len(df.dropna(subset=["region"])))

    dup = pd.DataFrame({"k": [1, 1, 2], "v": ["a", "a", "b"]})
    show("duplicated", dup.duplicated().tolist())
    show("drop_duplicates", dup.drop_duplicates())

    dates = pd.to_datetime(pd.Series(["2026-01-05", "no-date", "2026-03-31"]), errors="coerce")
    show("to_datetime coerced", dates)
    show("valid dates", int(dates.notna().sum()))

    show(
        "map",
        pd.Series(["Web", "App", "Phone"]).map(
            {"Web": "Online", "App": "Online", "Phone": "Assisted"}
        ),
    )
    show("rename", pd.DataFrame({"net": [1.0]}).rename(columns={"net": "net_revenue"}))

    spends = pd.Series([12.0, 45.0, 89.0, 120.0, 340.0, 1299.0])
    show("cut", pd.cut(spends, bins=[0, 50, 200, 2000], labels=["small", "mid", "large"]).tolist())
    show("qcut", pd.qcut(spends, q=3, labels=["low", "mid", "high"]).tolist())


# ------------------------------------------- 3.9 Grouping and aggregation
def section_39() -> None:
    df = load_clean()
    q1 = df[df["order_date"].between("2026-01-01", "2026-03-31")]

    by_channel = q1.groupby("channel", observed=True)["net_revenue"].sum().round(0)
    show("sum by channel", by_channel)

    named = q1.groupby("channel", observed=True).agg(
        net_revenue=("net_revenue", "sum"),
        orders=("order_id", "nunique"),
        lines=("net_revenue", "size"),
        return_rate=("returned", "mean"),
    )
    show("named agg", named.round(2))

    by_region_channel = q1.groupby(
        ["region", "channel"], observed=True
    )["net_revenue"].sum().round(0)
    show("two keys (head)", by_region_channel.head(6))

    q1 = q1.copy()
    q1["channel_mean"] = q1.groupby("channel", observed=True)["net_revenue"].transform("mean")
    show("transform channel mean (head)", q1["channel_mean"].head(3).round(2).tolist())
    q1["share"] = (q1["net_revenue"] / q1["channel_mean"]).round(3)
    show("transform share (head)", q1["share"].head(3).tolist())

    big_groups = q1.groupby("category", observed=True).filter(lambda g: len(g) > 1000)
    show("filter groups", big_groups["category"].nunique())

    recon = q1.groupby("channel", observed=True)["net_revenue"].size().sum()
    show("reconcile", (int(recon), len(q1)))

    share = q1.groupby("channel", observed=True)["net_revenue"].sum()
    show("share of revenue", (share / share.sum()).round(4))
    show(
        "largest lines",
        q1.nlargest(3, "net_revenue")[["order_id", "channel", "net_revenue"]],
    )


# ------------------------------------------- 3.10 Combining / reshaping
def section_310() -> None:
    jan = pd.DataFrame({"channel": ["Web", "App"], "net": [100.0, 200.0]})
    feb = pd.DataFrame({"channel": ["Web", "Phone"], "net": [150.0, 50.0]})
    show("concat rows", pd.concat([jan, feb], ignore_index=True))

    orders = pd.DataFrame(
        {"order_id": ["O-1", "O-2", "O-3"], "customer_id": ["C-1", "C-2", "C-1"]}
    )
    customers = pd.DataFrame(
        {"customer_id": ["C-1", "C-2"], "region": ["West", "South"]}
    )
    merged = orders.merge(customers, on="customer_id", how="left", validate="many_to_one")
    show("merge many_to_one", merged)

    show(
        "merge indicator",
        orders.merge(customers, on="customer_id", how="left", indicator=True),
    )

    left = pd.DataFrame({"k": ["a", "b"], "x": [1, 2]})
    right = pd.DataFrame({"k": ["a", "a", "b"], "y": [10, 20, 30]})
    show("fan-out rows", len(left.merge(right, on="k", validate="one_to_many")))

    wide = pd.DataFrame(
        {
            "channel": ["Web", "App"],
            "2025Q1": [2702000.0, 1318000.0],
            "2026Q1": [2500000.0, 1400000.0],
        }
    )
    show("wide", wide)
    tidy = wide.melt(id_vars="channel", var_name="quarter", value_name="net_revenue")
    show("melt", tidy)
    show("pivot back", tidy.pivot(index="channel", columns="quarter", values="net_revenue"))

    df = load_clean()
    q1 = df[df["order_date"].between("2026-01-01", "2026-03-31")]
    ct = pd.crosstab(q1["channel"], q1["category"])
    show("crosstab", ct)
    pt = q1.pivot_table(index="channel", columns="category", values="net_revenue", aggfunc="sum")
    show("pivot_table", pt.round(0))

    grouped = q1.groupby(["channel", "category"], observed=True)["net_revenue"].sum()
    wide_cat = grouped.unstack("category")
    show("unstack wide", wide_cat.round(0))
    show("stack back", wide_cat.stack().head(4).round(0))


# ------------------------------------------- 3.11 Time basics
def section_311() -> None:
    df = load_clean()
    df["order_month"] = df["order_date"].dt.to_period("M")
    monthly = df.groupby("order_month")["net_revenue"].sum().round(0)
    show("monthly tail", monthly.tail(4))
    show("dt accessors", (df["order_date"].dt.year.head(1).tolist(), df["order_date"].dt.day_name().head(1).tolist()))

    ts = df.set_index("order_date")["net_revenue"].sort_index()
    weekly = ts.resample("W").sum().round(0)
    show("resample weekly head", weekly.head(3))
    rolling = ts.resample("D").sum().rolling(28).mean().dropna()
    show("rolling 28d tail", rolling.tail(2).round(0))

    df = df.sort_values("order_date")
    df["prev_day_net"] = df["net_revenue"].shift(1)
    show("shift head", df[["order_date", "net_revenue", "prev_day_net"]].head(3))

    q1 = df[df["order_date"].between("2026-01-01", "2026-03-31")]
    q1_months = q1.groupby(q1["order_date"].dt.month_name())["net_revenue"].sum().round(0)
    show("q1 by month name", q1_months)

    q1_25 = df[df["order_date"].between("2025-01-01", "2025-03-31")]["net_revenue"].sum()
    q1_26 = df[df["order_date"].between("2026-01-01", "2026-03-31")]["net_revenue"].sum()
    show(
        "q1 yoy",
        {
            "2025Q1": round(float(q1_25), 2),
            "2026Q1": round(float(q1_26), 2),
            "pct_change": round(float(q1_26 / q1_25 - 1) * 100, 1),
        },
    )


# ------------------------------------------- 3.12 Performance / correctness
def section_312() -> None:
    df = load_clean()
    tip = df["net_revenue"].sum()
    show("sum net (total)", round(tip, 2))

    small = df.head(1000).copy()
    method = small.groupby("channel", observed=True)["net_revenue"].sum()
    show("series money", method.round(2))

    mem = pd.DataFrame(
        {
            "channel_str": small["channel"].astype("object"),
            "channel_cat": small["channel"].astype("category"),
        }
    )
    show("memory bytes", mem.memory_usage(deep=True).to_dict())

    # apply vs vectorized
    row_time = timeit.timeit(lambda: small["net_revenue"].apply(lambda v: v * 0.08), number=20)
    vec_time = timeit.timeit(lambda: small["net_revenue"] * 0.08, number=20)
    show("apply vs vector", f"apply: {row_time:.4f}s  vector: {vec_time:.4f}s")


# ------------------------------------------- 3.13 example bank
def section_313() -> None:
    df = load_clean()
    show("shape", df.shape)
    show("channels", df["channel"].nunique())
    show("date range", (str(df["order_date"].min().date()), str(df["order_date"].max().date())))
    show("median revenue", round(float(df["revenue"].median()), 2))
    show(
        "by channel sum/mean",
        df.groupby("channel", observed=True)["net_revenue"].agg(["sum", "mean"]).round(2),
    )
    show("category share", df["category"].value_counts(normalize=True).round(3))
    show(
        "priciest products",
        df.nlargest(3, "unit_price")[["product_id", "category", "unit_price"]],
    )
    show("shipping mix", df["shipping_cost"].value_counts().sort_index())
    returned_gross = df.loc[df["returned"] == 1, "revenue"].sum()
    show("gross revenue on returned lines", round(float(returned_gross), 2))
    show("return share of gross", round(float(returned_gross / df["revenue"].sum()), 4))
    show("discount share", round(float(df["discount_pct"].fillna(0).gt(0).mean()), 4))
    show(
        "net by year",
        df.groupby(df["order_date"].dt.year)["net_revenue"].sum().round(0),
    )
    show(
        "mean net by region",
        df.groupby("region", observed=True)["net_revenue"].mean().round(2),
    )
    show("channel vs returned", pd.crosstab(df["channel"], df["returned"]))
    show("distinct customers", int(df["customer_id"].nunique()))
    show(
        "distinct orders by channel",
        df.groupby("channel", observed=True)["order_id"].nunique(),
    )
    show(
        "top lines",
        df.nlargest(3, "net_revenue")[["order_id", "channel", "net_revenue"]],
    )
    show(
        "quarterly line counts",
        df["order_date"].dt.to_period("Q").value_counts().sort_index().tail(4),
    )
    prod_rev = (
        df.groupby("product_id")["net_revenue"].sum().nlargest(3).round(0)
    )
    show("top products by revenue", prod_rev)
    show(
        "top stores",
        df.groupby("store_id")["net_revenue"].sum().nlargest(3).round(0),
    )
    lines_per_order = df.groupby("order_id").size()
    show("avg lines per order", round(float(lines_per_order.mean()), 3))
    show("net quantiles", df["net_revenue"].quantile([0.9, 0.99]).round(2))
    show(
        "weekday pattern",
        df["order_date"].dt.day_name().value_counts().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ),
    )
    show(
        "region value counts (keep NaN)",
        df["region"].value_counts(dropna=False),
    )
    show(
        "channel x category share (index)",
        pd.crosstab(df["channel"], df["category"], normalize="index").round(3),
    )
    show("stores", df["store_id"].nunique())
    big_orders = (lines_per_order > 2).sum()
    show("orders with more than 2 lines", int(big_orders))


# ------------------------------------------- 3.14 capstone
def section_314() -> None:
    orders = load_clean()
    products = pd.read_csv(RAW / "kestrel_products.csv")
    customers = pd.read_csv(
        RAW / "kestrel_customers.csv", dtype={"customer_id": "string"}
    )

    with_product = orders.merge(
        products, on="product_id", how="left", validate="many_to_one"
    )
    show("orders vs after product merge", (len(orders), len(with_product)))

    customer_table = (
        with_product[with_product["customer_id"].notna()]
        .groupby("customer_id")
        .agg(
            orders=("order_id", "nunique"),
            lines=("order_id", "size"),
            net_revenue=("net_revenue", "sum"),
            first_order=("order_date", "min"),
            last_order=("order_date", "max"),
        )
        .reset_index()
    )
    joined = customer_table.merge(
        customers, on="customer_id", how="left", indicator=True
    )
    show("indicator", joined["_merge"].value_counts().to_dict())
    show(
        "customer table head",
        joined.head(4)[["customer_id", "orders", "net_revenue", "loyalty_tier", "home_region"]],
    )
    show("table shape", joined.shape)
    guest_lines = int(orders["customer_id"].isna().sum())
    show(
        "reconcile lines",
        {
            "all clean": len(orders),
            "guest lines": guest_lines,
            "customer-table lines": int(joined["lines"].sum()),
            "check": len(orders) - guest_lines,
        },
    )
    show(
        "reconcile orders",
        {
            "non-guest distinct orders": int(
                orders[orders["customer_id"].notna()]["order_id"].nunique()
            ),
            "customer-table orders": int(joined["orders"].sum()),
        },
    )


# ------------------------------------------- 3.13.6+ example bank II
def section_313b() -> None:
    df = load_clean()

    # string operations on identifiers
    print("### order_id length counts")
    print(df["order_id"].str.len().value_counts())
    print()
    print("### order_id all start with O-")
    print(bool(df["order_id"].str.startswith("O-").all()))
    print()
    print("### order_id numeric part")
    print(df["order_id"].str.slice(2).head(3).tolist())
    print()
    print("### product_id matches P-dddd")
    print(round(float(df["product_id"].str.match(r"P-\d{4}").mean()), 4))
    print()

    # concat round trip
    by_year = {year: g for year, g in df.groupby(df["order_date"].dt.year)}
    annual = pd.concat(
        [g.assign(year=y) for y, g in by_year.items()], ignore_index=True
    )
    print("### concat round trip length")
    print(len(annual))
    print()

    # grouped transform: share and rank
    q1 = df[df["order_date"].between("2026-01-01", "2026-03-31")].copy()
    q1["channel_total"] = q1.groupby("channel", observed=True)["net_revenue"].transform(
        "sum"
    )
    q1["share_of_channel"] = q1["net_revenue"] / q1["channel_total"]
    print("### share of channel (head)")
    print(q1["share_of_channel"].head(3).round(4).tolist())
    print()
    q1["rank_in_channel"] = q1.groupby("channel", observed=True)["net_revenue"].rank(
        ascending=False
    )
    print("### rank within channel (head)")
    print(q1["rank_in_channel"].head(3).tolist())
    print()

    # method chaining
    chain = (
        df.assign(month=df["order_date"].dt.to_period("M"))
        .query("channel == 'Web'")
        .groupby("month", observed=True)["net_revenue"]
        .sum()
        .rename("web_net")
    )
    print("### method chain tail")
    print(chain.tail(3).round(0))
    print()

    # pipe with a named step
    def top_n(frame: pd.DataFrame, n: int) -> pd.DataFrame:
        return frame.nlargest(n, "net_revenue")

    top = (
        df[df["order_date"].between("2026-01-01", "2026-03-31")]
        .pipe(top_n, 2)[["order_id", "channel", "net_revenue"]]
    )
    print("### pipe top 2")
    print(top)
    print()

    # describe by group
    print("### describe net by channel")
    print(df.groupby("channel", observed=True)["net_revenue"].describe().round(2))
    print()

    # where and mask
    s = pd.Series([-1.0, 2.0, -3.0, 4.0])
    print("### where / mask")
    print(s.where(s > 0, 0.0).tolist(), s.mask(s > 0, 0.0).tolist())
    print()

    # explode
    e = pd.DataFrame(
        {"order_id": ["O-1", "O-2"], "products": [["P-1", "P-2"], ["P-3"]]}
    )
    print("### explode")
    print(e.explode("products"))
    print()

    # join to loyalty tier and summarize
    customers = pd.read_csv(
        RAW / "kestrel_customers.csv", dtype={"customer_id": "string"}
    )
    joined = df[df["customer_id"].notna()].merge(
        customers, on="customer_id", how="left"
    )
    print("### net by loyalty tier")
    print(
        joined.groupby("loyalty_tier")["net_revenue"]
        .sum()
        .round(0)
        .sort_values(ascending=False)
    )
    print()


def section_extras() -> None:
    df = load_clean()

    # 3.1.9 twelve months in a four-column array
    sales = np.array(
        [
            [120, 135, 128, 141], [110, 122, 131, 119], [145, 150, 139, 160],
            [160, 158, 170, 165], [150, 149, 155, 158], [170, 181, 176, 190],
            [185, 190, 188, 195], [180, 178, 186, 184], [195, 210, 201, 214],
            [205, 198, 215, 209], [230, 240, 228, 246], [260, 255, 270, 268],
        ],
        dtype=float,
    )
    show("sales array shape", sales.shape)
    show("store totals (axis=0)", sales.sum(axis=0))
    show("month totals (axis=1)", sales.sum(axis=1))
    show("grand total", float(sales.sum()))

    # 3.3.7 year-to-date with cumsum
    monthly = df.groupby(df["order_date"].dt.to_period("M"))["net_revenue"].sum()
    ytd = monthly.loc["2025-01":"2025-12"].cumsum().round(0)
    show("ytd 2025 head", ytd.head(3))
    show("ytd 2025 full year", float(ytd.iloc[-1]))

    # 3.6.6 ten business questions
    q1 = df[df["order_date"].between("2026-01-01", "2026-03-31")]
    answers = {
        "Q1 2026 net revenue": round(float(q1["net_revenue"].sum()), 2),
        "top channel": q1.groupby("channel", observed=True)["net_revenue"].sum().idxmax(),
        "Q1 return rate": round(float(q1["returned"].mean()), 4),
        "Q1 average order value": round(float(q1["net_revenue"].sum() / q1["order_id"].nunique()), 2),
        "Q4 2025 camping lines": int(((df["order_date"].between("2025-10-01", "2025-12-31")) & (df["category"] == "Camping")).sum()),
        "customers with 20+ orders": int((df.groupby("customer_id")["order_id"].nunique() >= 20).sum()),
        "lines at $49.95 shipping": int((df["shipping_cost"] == 49.95).sum()),
        "mean discount when discounted": round(float(df.loc[df["discount_pct"].fillna(0) > 0, "discount_pct"].mean()), 2),
        "West app net revenue": round(float(df.loc[(df["region"] == "West") & (df["channel"] == "App"), "net_revenue"].sum()), 2),
        "distinct products sold": int(df["product_id"].nunique()),
    }
    print("### ten questions")
    for key, value in answers.items():
        print(f"{key}: {value}")
    print()

    # 3.9.8 one agg, several metrics
    q1_roll = q1.groupby(["region", "channel"], observed=True).agg(
        net_revenue=("net_revenue", "sum"),
        orders=("order_id", "nunique"),
        lines=("order_id", "size"),
        return_rate=("returned", "mean"),
    ).round(2)
    show("region x channel agg head", q1_roll.head(4))
    show(
        "AOV from the same table",
        (q1_roll["net_revenue"] / q1_roll["orders"]).round(2).head(4),
    )

    # 3.10.7 merge lab
    orders = df
    products = pd.read_csv(RAW / "kestrel_products.csv")
    customers = pd.read_csv(RAW / "kestrel_customers.csv", dtype={"customer_id": "string"})
    step1 = orders.merge(products, on="product_id", how="left", validate="many_to_one")
    show("merge lab step1 counts", (len(orders), len(step1)))
    step2 = step1.merge(customers, on="customer_id", how="left", indicator=True)
    show("merge lab step2 counts", (len(step1), len(step2)))
    show("merge lab unmapped products", int(step1["product_name"].isna().sum()))
    show("merge lab unmatched customers", int((step2["_merge"] == "left_only").sum()))


def main() -> None:
    print("pandas", pd.__version__, "numpy", np.__version__)
    print()
    for section in (
        section_31,
        section_32,
        section_33,
        section_34,
        section_35,
        section_36,
        section_37,
        section_38,
        section_39,
        section_310,
        section_311,
        section_312,
        section_313,
        section_314,
        section_extras,
        section_313b,
    ):
        print("=" * 70)
        section()


if __name__ == "__main__":
    main()
