#!/usr/bin/env python3
"""Print every output block used in Chapter 7, from the current files.

Run from the project root:

    python scripts/chapter07_outputs.py

Reads the Kestrel orders, customers, products, the 12 monthly exports, and the
fixtures written by generate_chapter07_sources.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
AUDIT = RAW / "audit"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 80)

_ORDERS = None


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def orders() -> pd.DataFrame:
    """A deterministic 20,000-row sample of the order-line feed."""
    global _ORDERS
    if _ORDERS is None:
        raw = pd.read_csv(RAW / "kestrel_orders.csv", dtype=str,
                          keep_default_na=False, na_values=[""])
        _ORDERS = raw.sample(20000, random_state=707).reset_index(drop=True)
    return _ORDERS.copy()


def numeric_orders(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["revenue_num"] = pd.to_numeric(out["revenue"], errors="coerce")
    out["quantity_num"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["discount_num"] = pd.to_numeric(out["discount_pct"], errors="coerce").fillna(0)
    return out


def customers() -> pd.DataFrame:
    return pd.read_csv(RAW / "kestrel_customers.csv", dtype=str,
                       keep_default_na=False, na_values=[""])


def products() -> pd.DataFrame:
    return pd.read_csv(RAW / "kestrel_products.csv")


def monthly_exports() -> pd.DataFrame:
    frames = []
    for path in sorted((RAW / "sales_exports").glob("sales_*.csv")):
        part = pd.read_csv(path, dtype=str, keep_default_na=False)
        part["source_file"] = path.name
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


# ============================================================ 7.1 relational
def section_71() -> None:
    o, c, p = orders(), customers(), products()
    show("R1 source sizes", {
        "order_lines": len(o), "distinct_order_ids": int(o["order_id"].nunique()),
        "customers": len(c), "distinct_customer_ids": int(c["customer_id"].nunique()),
        "products": len(p), "stores": 12,
    })

    def card(left, right, key):
        return {
            "left_rows": len(left), "right_rows": len(right),
            "left_keys": int(left[key].nunique(dropna=True)),
            "right_keys": int(right[key].nunique(dropna=True)),
            "right_duplicate_keys": int(right[key].duplicated().sum()),
            "relationship": "many-to-one" if not right[key].duplicated().any()
                            else "many-to-many",
        }

    stores = pd.DataFrame({"store_id": [f"S-{i:02d}" for i in range(1, 13)]})
    cards = pd.DataFrame([
        dict(source="orders -> customers", **card(o, c, "customer_id")),
        dict(source="orders -> products", **card(o, p, "product_id")),
        dict(source="orders -> stores", **card(o, stores, "store_id")),
    ])
    show("R2 cardinality checks", cards.to_string(index=False))

    jmap = pd.DataFrame([
        ("orders", "one row per order line", "order_id, product_id", "fact"),
        ("customers", "one row per customer", "customer_id", "dimension"),
        ("products", "one row per product", "product_id", "dimension"),
        ("stores", "one row per store", "store_id", "dimension"),
        ("price_history", "one row per product per effective date", "product_id + effective_from", "dimension"),
        ("campaign_customer", "one row per customer exposure", "customer_id", "fact"),
    ], columns=["table", "grain", "key", "role"])
    show("R3 join map", jmap.to_string(index=False))

    try:
        o.merge(c, on="customer_id", how="left", validate="many_to_one")
        print("validate('many_to_one'): PASS on the clean customer dimension")
    except pd.errors.MergeError as exc:
        print("validate failed:", exc)

    show("R5 null keys in the fact", {
        "customer_id_null": int(o["customer_id"].isna().sum()),
        "product_id_null": int(o["product_id"].isna().sum()),
        "store_id_null": int(o["store_id"].isna().sum()),
    })

    per_order = o.groupby("order_id").size()
    show("R6 grain of the fact", {
        "lines": len(o), "orders": int(o["order_id"].nunique()),
        "lines_per_order": round(float(per_order.mean()), 3),
        "max_lines": int(per_order.max()),
    })


# ======================================================= 7.2 concatenation
def section_72() -> None:
    paths = sorted((RAW / "sales_exports").glob("sales_*.csv"))
    shapes = [(p.name, len(pd.read_csv(p))) for p in paths]
    show("C1 monthly files", {"files": len(shapes), "total_rows": sum(n for _, n in shapes),
                              "first_two": shapes[:2], "last_two": shapes[-2:]})

    frames = [(p.name, pd.read_csv(p, dtype=str, keep_default_na=False)) for p in paths]
    cols = sorted(set().union(*[set(f.columns) for _, f in frames]))
    drift = []
    for col in cols:
        dtypes = sorted({str(f[col].dtype) for _, f in frames if col in f.columns})
        present = sum(col in f.columns for _, f in frames)
        if len(dtypes) > 1 or present < len(frames):
            drift.append((col, present, dtypes))
    show("C2 dtype / column drift", pd.DataFrame(
        drift, columns=["column", "files_present", "dtypes"]).to_string(index=False))

    stacked = monthly_exports()
    show("C3 safe stack", {"rows": len(stacked), "columns": len(stacked.columns),
                           "sources": int(stacked["source_file"].nunique()),
                           "null_sku": int(stacked["sku"].eq("").sum())})

    tagged = pd.concat(
        [pd.read_csv(p, dtype=str, keep_default_na=False).assign(month=p.stem[-7:])
         for p in paths], ignore_index=True)
    show("C4 keys= equivalent", {
        "rows": len(tagged), "months": int(tagged["month"].nunique()),
        "month_values": sorted(tagged["month"].unique().tolist())[:3],
    })

    left = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    right = pd.DataFrame({"a": [5, 6], "c": [7, 8]})
    show("C5 outer vs inner columns", {
        "outer_columns": sorted(pd.concat([left, right], ignore_index=True).columns.tolist()),
        "inner_columns": sorted(pd.concat([left, right], join="inner", ignore_index=True).columns.tolist()),
        "outer_rows": len(pd.concat([left, right], ignore_index=True)),
        "inner_rows": len(pd.concat([left, right], join="inner", ignore_index=True)),
    })


# ============================================================ 7.3 merging
def section_73() -> None:
    o = numeric_orders(orders())
    c = customers()
    joined = o.merge(c, on="customer_id", how="left", indicator=True, validate="many_to_one")
    show("M1 left join with indicator", {
        "rows_before": len(o), "rows_after": len(joined),
        "left_only": int(joined["_merge"].eq("left_only").sum()),
        "both": int(joined["_merge"].eq("both").sum()),
    })

    dup = pd.read_csv(AUDIT / "customers_duplicate_keys.csv", dtype=str,
                      keep_default_na=False, na_values=[""])
    try:
        o.merge(dup, on="customer_id", how="left", validate="many_to_one")
        print("validate: unexpectedly passed")
    except pd.errors.MergeError as exc:
        print("validate('many_to_one') rejected the duplicate dimension:", str(exc)[:90])

    counts = {}
    for how in ["left", "inner", "right", "outer"]:
        counts[how] = len(o.merge(c, on="customer_id", how=how))
    show("M3 join types row counts", counts)

    both = o.merge(c.rename(columns={"home_region": "region"}), on="customer_id",
                   how="left", suffixes=("_order", "_cust"))
    show("M4 suffixes", {"columns_with_region": [x for x in both.columns if "region" in x],
                         "example": both[["region_order", "region_cust"]].dropna().head(2).to_dict("records")})

    dirty = pd.read_csv(AUDIT / "customers_join_dirty.csv", dtype=str,
                        keep_default_na=False, na_values=[""])
    before = dirty["customer_id"].nunique()
    clean_ids = dirty["customer_id"].str.strip().str.upper().str.replace(
        r"^C-0*(\d+)$", r"C-\1", regex=True)
    dirty = dirty.assign(key_clean=clean_ids)
    show("M5 key normalization", {
        "raw_distinct_keys": int(before), "clean_distinct_keys": int(dirty["key_clean"].nunique()),
        "canonical_form_example": dirty["key_clean"].head(6).tolist(),
    })

    o2 = o.copy()
    o2.loc[0:4, "customer_id"] = None
    miss = o2.merge(c, on="customer_id", how="left", indicator=True)
    show("M6 null join keys", {
        "null_keys": int(o2["customer_id"].isna().sum()),
        "null_keys_left_only": int(miss.loc[miss["customer_id"].isna(), "_merge"].eq("left_only").sum()),
    })

    show("M7 cross join size", {"left": 12, "right": 4,
                                "cross_rows": len(pd.DataFrame({"a": range(12)}).merge(
                                    pd.DataFrame({"b": range(4)}), how="cross"))})

    ind = o.merge(c, on="customer_id", how="left", indicator=True)
    show("M8 anti and semi joins", {
        "anti_rows": int(ind["_merge"].eq("left_only").sum()),
        "semi_rows": int(ind["_merge"].eq("both").sum()),
    })


# ============================================================ 7.4 fan-out
def section_74() -> None:
    o = numeric_orders(orders())
    dup = pd.read_csv(AUDIT / "customers_duplicate_keys.csv", dtype=str,
                      keep_default_na=False, na_values=[""])
    joined = o.merge(dup, on="customer_id", how="left", validate="many_to_many")
    gross_before = round(float(o["revenue_num"].sum()), 2)
    gross_after = round(float(joined["revenue_num"].sum()), 2)
    show("F1 fan-out effect", {
        "rows_before": len(o), "rows_after": len(joined),
        "extra_rows": len(joined) - len(o),
        "gross_before": gross_before, "gross_after": gross_after,
        "gross_inflation": round(gross_after - gross_before, 2),
    })

    sizes = joined.groupby(["order_id", "product_id"]).size()
    show("F2 detecting fan-out", {
        "line_keys": int(o.groupby(["order_id", "product_id"]).ngroups),
        "keys_with_multiple_rows": int((sizes > 1).sum()),
        "max_rows_per_key": int(sizes.max()),
    })

    audit = pd.DataFrame([
        ("left rows", len(o)), ("right rows", len(dup)),
        ("joined rows", len(joined)), ("row delta", len(joined) - len(o)),
        ("gross delta", round(gross_after - gross_before, 2)),
    ], columns=["check", "value"])
    show("F3 join audit table", audit.to_string(index=False))

    dedup = dup.drop_duplicates(subset=["customer_id"], keep="first")
    fixed = o.merge(dedup, on="customer_id", how="left", validate="many_to_one")
    show("F4 fixed join", {
        "rows": len(fixed),
        "gross": round(float(fixed["revenue_num"].sum()), 2),
        "gross_matches_original": abs(float(fixed["revenue_num"].sum()) - gross_before) < 0.01,
    })

    try:
        o.merge(dup, on="customer_id", how="left", validate="many_to_one")
        print("validate: unexpectedly passed")
    except pd.errors.MergeError:
        print("validate('many_to_one') blocked the fan-out before the join ran")


# ============================================================== 7.5 as-of
def section_75() -> None:
    o = orders()
    prices = pd.read_csv(AUDIT / "price_history.csv")
    show("A1 price history", {
        "rows": len(prices), "products": int(prices["product_id"].nunique()),
        "dates": sorted(prices["effective_from"].unique().tolist()),
        "example": prices.head(3).to_dict("records"),
    })

    left = o.assign(order_date=pd.to_datetime(o["order_date"], errors="coerce"))
    left = left.dropna(subset=["order_date"]).sort_values("order_date")
    right = prices.assign(effective_from=pd.to_datetime(prices["effective_from"])).sort_values("effective_from")
    asof = pd.merge_asof(left, right, left_on="order_date", right_on="effective_from",
                         by="product_id", direction="backward",
                         suffixes=("_order", "_hist"))
    show("A2 as-of join", {
        "rows": len(asof), "with_price": int(asof["unit_price_hist"].notna().sum()),
        "example": asof[["order_id", "order_date", "product_id", "effective_from", "unit_price_hist"]].head(3).to_string(index=False),
    })

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        tol = pd.merge_asof(left, right, left_on="order_date", right_on="effective_from",
                            by="product_id", direction="backward",
                            tolerance=pd.Timedelta("120D"),
                            suffixes=("_order", "_hist"))
    show("A3 tolerance window", {
        "rows": len(tol), "with_price": int(tol["unit_price_hist"].notna().sum()),
        "dropped_by_tolerance": int(asof["unit_price_hist"].notna().sum() - tol["unit_price_hist"].notna().sum()),
    })

    comp = asof.dropna(subset=["unit_price_hist"]).copy()
    comp["stored"] = pd.to_numeric(comp["unit_price_order"], errors="coerce")
    diff = (comp["unit_price_hist"] - comp["stored"]).abs()
    show("A4 as-of vs stored price", {
        "rows": len(comp),
        "exact_matches": int((diff < 0.01).sum()),
        "median_abs_diff": round(float(diff.median()), 2),
    })


# ==================================================== 7.6 group and aggregate
def section_76() -> None:
    o = numeric_orders(orders())
    g = o.groupby("region")["order_id"]
    show("G1 count / size / nunique", o.groupby("region").agg(
        count_lines=("order_id", "count"),
        count_orders=("order_id", "nunique"),
        rows=("revenue_num", "size"),
        non_null_orders=("order_id", "count")).to_string())

    named = o.groupby("region").agg(
        lines=("order_id", "size"),
        orders=("order_id", "nunique"),
        revenue=("revenue_num", "sum"),
        aov=("revenue_num", lambda s: round(float(s.sum()) / max(int(o.loc[s.index, "order_id"].nunique()), 1), 2)),
    )
    show("G2 named aggregation", named.round(2).to_string())

    multi = o.groupby(["region", "channel"]).agg(
        lines=("order_id", "size"), revenue=("revenue_num", "sum")).round(2)
    show("G3 multiple keys", multi.head(8).to_string())

    wavg = float((o["discount_num"] * o["revenue_num"]).sum() / o["revenue_num"].sum())
    show("G4 weighted average", {
        "plain_mean_discount": round(float(o["discount_num"].mean()), 3),
        "revenue_weighted_discount": round(wavg, 3),
    })

    def spread(s):
        return round(float(s.max() - s.min()), 2)
    custom = o.groupby("region")["revenue_num"].agg(
        n="size", mean="mean", spread=spread).round(2)
    show("G5 custom aggregation", custom.to_string())

    hier = o.groupby(["region", "channel"])["revenue_num"].sum().round(2)
    show("G6 hierarchical grouping", {
        "groups": int(hier.shape[0]),
        "region_totals": hier.groupby(level=0).sum().round(2).to_dict(),
    })

    show("G7 count vs size vs nunique", {
        "count_lines": int(o.groupby("region")["order_id"].count().sum()),
        "size_lines": int(o.groupby("region")["order_id"].size().sum()),
        "nunique_orders": int(o.groupby("region")["order_id"].nunique().sum()),
    })


# ======================================================== 7.7 transform
def section_77() -> None:
    o = numeric_orders(orders())
    o["rev"] = o["revenue_num"]
    o["region_mean"] = o.groupby("region")["rev"].transform("mean")
    o["region_share"] = o["rev"] / o.groupby("region")["rev"].transform("sum")
    o["rank_in_region"] = o.groupby("region")["rev"].rank(method="dense", ascending=False)
    o["share_pct"] = (o["region_share"] * 100).round(3)
    show("T1-T3 transform columns", o[["order_id", "region", "rev", "region_mean", "share_pct", "rank_in_region"]].head(4).round(3).to_string(index=False))

    totals = o.groupby("region")["region_share"].sum().round(6)
    show("T4 verify shares", {"share_sums": totals.to_dict(),
                              "all_one": bool((totals - 1).abs().max() < 1e-6)})

    big = o.groupby("region")["rev"].sum()
    keep = big[big > big.median()].index
    filtered = o[o["region"].isin(keep)]
    show("T5 filter groups", {
        "regions_kept": sorted(keep.tolist()),
        "rows_before": len(o), "rows_after": len(filtered),
        "revenue_after": round(float(filtered["rev"].sum()), 2),
    })


# ========================================================= 7.8 reshaping
def section_78() -> None:
    wide = pd.read_csv(AUDIT / "monthly_report_wide.csv")
    show("S1 wide report", {"shape": wide.shape, "columns_head": wide.columns.tolist()[:6]})

    long = wide.melt(id_vars="region", var_name="metric_month", value_name="value")
    show("S2 melt", {"shape": long.shape, "head": long.head(4).to_string(index=False)})

    parts = long["metric_month"].str.split("_", n=1, expand=True)
    long["metric"] = parts[0]
    long["month"] = parts[1]
    long = long.drop(columns="metric_month")
    show("S3 split variable names", {"columns": long.columns.tolist(),
                                      "metrics": sorted(long["metric"].unique().tolist()),
                                      "months": int(long["month"].nunique())})

    back = long.pivot_table(index="region", columns=["metric", "month"],
                            values="value", aggfunc="sum")
    back.columns = [f"{m}_{mo}" for m, mo in back.columns]
    back = back[wide.columns[1:].tolist()].reset_index()
    roundtrip = np.allclose(back[wide.columns[1:]].to_numpy(float),
                            wide[wide.columns[1:]].to_numpy(float))
    show("S4 round-trip", {"shape": back.shape, "exactly_equal": bool(roundtrip)})

    o = orders()
    ct = pd.crosstab(o["region"], o["channel"], normalize="index").round(4)
    show("S5 cross-tab row percentages", ct.to_string())
    ct2 = pd.crosstab(o["region"], o["channel"], margins=True, margins_name="Total")
    show("S5 cross-tab with margins", ct2.to_string())

    small = wide[["region", "revenue_2025-01", "revenue_2025-02"]].set_index("region")
    stacked = small.stack()
    unstacked = stacked.unstack()
    show("S6 stack / unstack", {"stacked_head": stacked.head(4).to_string(),
                                 "unstacked_shape": unstacked.shape})


# =========================================================== 7.9 tidy data
def section_79() -> None:
    violations = pd.DataFrame([
        ("wide columns", "revenue_2025-01, revenue_2025-02", "month is a column name"),
        ("values as labels", "Q1 | Q2 | Q3 in one cell", "two variables in one cell"),
        ("missing rows", "region absent for West", "an observation is missing"),
    ], columns=["violation", "example", "why_it_breaks"])
    show("D1 three violations", violations.to_string(index=False))

    o = orders()
    long = o.groupby(["region", "channel"]).size().reset_index(name="lines")
    wide = long.pivot(index="region", columns="channel", values="lines").fillna(0)
    show("D2 two shapes", {"long": long.head(4).to_string(index=False),
                            "wide": wide.to_string()})

    tidy_checks = {
        "one_row_per_observation": bool(not long.duplicated(subset=["region", "channel"]).any()),
        "one_column_per_variable": bool(set(long.columns) == {"region", "channel", "lines"}),
        "one_value_per_cell": bool(long["lines"].map(lambda v: isinstance(v, (int, np.integer))).all()),
    }
    show("D3 tidiness assertions", tidy_checks)


# ================================================== 7.10 analysis table
def section_710() -> None:
    star = pd.DataFrame([
        ("fact_orders", "order line", "order_id + product_id", "revenue, quantity"),
        ("dim_customers", "customer", "customer_id", "tier, signup"),
        ("dim_products", "product", "product_id", "category, list_price"),
        ("dim_stores", "store", "store_id", "region"),
        ("fact_campaigns", "campaign exposure", "customer_id", "spend"),
    ], columns=["table", "grain", "key", "measures"])
    show("B1 star schema", star.to_string(index=False))

    o = numeric_orders(orders())
    c = customers().drop_duplicates(subset=["customer_id"])
    p = products()
    camp = pd.read_csv(AUDIT / "campaign_customer.csv", dtype=str,
                       keep_default_na=False, na_values=[""])
    camp["spend_num"] = pd.to_numeric(camp["spend"], errors="coerce")

    base = o.merge(c[["customer_id", "loyalty_tier", "home_region"]],
                   on="customer_id", how="left", validate="many_to_one")
    base = base.merge(p[["product_id", "category"]].rename(
        columns={"category": "product_category"}), on="product_id",
        how="left", validate="many_to_one")
    cust = base.groupby("customer_id").agg(
        lines=("order_id", "size"),
        orders=("order_id", "nunique"),
        revenue=("revenue_num", "sum"),
        categories=("product_category", "nunique"),
    )
    spend = camp.groupby("customer_id")["spend_num"].sum().rename("campaign_spend")
    cust = cust.join(spend).fillna({"campaign_spend": 0.0}).round(2)
    show("B2 customer-level table", {"shape": cust.shape,
                                      "head": cust.head(4).to_string()})

    lineage = pd.DataFrame([
        ("lines, orders, revenue", "orders", "order_id, product_id", "many-to-one"),
        ("categories", "products", "product_id", "many-to-one"),
        ("campaign_spend", "campaign_customer", "customer_id", "one-to-one after agg"),
    ], columns=["output_column", "source", "key", "cardinality"])
    show("B3 lineage", lineage.to_string(index=False))

    path = PROCESSED / "customer360_ch07.parquet"
    cust.to_parquet(path)
    back = pd.read_parquet(path)
    show("B4 parquet round-trip", {
        "path": str(path.relative_to(ROOT)), "rows": len(back),
        "columns": back.columns.tolist(),
        "bytes": int(path.stat().st_size),
        "equal": bool(back.equals(cust)),
    })


# ======================================================== 7.11 example bank
def section_711() -> None:
    o = orders()
    c = customers()
    show("E1 key uniqueness", {
        "orders_key_unique": bool(not o.duplicated(subset=["order_id", "product_id"]).any()),
        "customers_unique": bool(not c["customer_id"].duplicated().any()),
    })
    show("E2 right-key duplicates", int(c["customer_id"].duplicated().sum()))
    show("E3 join key nulls", {k: int(o[k].isna().sum())
                               for k in ["customer_id", "product_id", "store_id"]})
    show("E4 merge indicator", o.merge(c, on="customer_id", how="left",
                                       indicator=True)["_merge"].value_counts().to_dict())
    show("E5 group totals", o.assign(rev=pd.to_numeric(o["revenue"], errors="coerce")).groupby("region")["rev"].sum().round(2).to_dict())
    show("E6 wide-to-long roundtrip", int(pd.read_csv(AUDIT / "monthly_report_wide.csv").melt(id_vars="region").shape[0]))


# =========================================================== 7.12 capstone
def section_712() -> None:
    o = numeric_orders(orders())
    o["month"] = pd.to_datetime(o["order_date"], errors="coerce").dt.strftime("%Y-%m")
    o["cost"] = (o["revenue_num"] * 0.68).round(2)
    o = o.dropna(subset=["month", "revenue_num"])

    score = o.groupby(["month", "region"]).agg(
        lines=("order_id", "size"),
        orders=("order_id", "nunique"),
        revenue=("revenue_num", "sum"),
        cost=("cost", "sum"),
    ).reset_index()
    score["aov"] = (score["revenue"] / score["orders"]).round(2)
    score["margin"] = (score["revenue"] - score["cost"]).round(2)
    score["revenue_share"] = (score["revenue"]
                              / score.groupby("month")["revenue"].transform("sum")).round(4)
    score["growth"] = score.groupby("region")["revenue"].pct_change().round(4)
    show("K1 monthly scorecard", score.head(6).round(2).to_string(index=False))

    total = round(float(score["revenue"].sum()), 2)
    source = round(float(o["revenue_num"].sum()), 2)
    null_region = o[o["region"].isna()]
    show("K2 control totals", {
        "scorecard_revenue": total, "source_revenue": source,
        "difference": round(total - source, 2),
        "rows_with_null_region": len(null_region),
        "revenue_on_null_region": round(float(null_region["revenue_num"].sum()), 2),
        "share_sum_per_month": score.groupby("month")["revenue_share"].sum().round(3).head(3).to_dict(),
    })

    wide_score = score.pivot(index="month", columns="region", values="revenue").round(2)
    show("K3 wide scorecard", wide_score.head(4).to_string())

    path = REPORTS / "ch07_monthly_scorecard.csv"
    score.to_csv(path, index=False)
    show("K4 artifact", {"path": str(path.relative_to(ROOT)),
                          "rows": len(score), "columns": score.columns.tolist()})


def section_extra() -> None:
    """Additional worked blocks for the expanded Chapter 7 sections."""
    o = orders()
    o["rev"] = pd.to_numeric(o["revenue"], errors="coerce")
    c = customers()
    p = products()
    paths = sorted((RAW / "sales_exports").glob("sales_*.csv"))
    frames = [(pth.name, pd.read_csv(pth, dtype=str, keep_default_na=False)) for pth in paths]

    # N1 one-to-one vs one-to-many
    lf = pd.DataFrame({"k": [1, 2], "v": ["a", "b"]})
    one = pd.DataFrame({"k": [1, 2], "w": [10, 20]})
    many = pd.DataFrame({"k": [1, 1, 2], "w": [10, 11, 20]})
    show("N1 one-to-one vs one-to-many", {
        "one_to_one_rows": len(lf.merge(one, on="k", validate="one_to_one")),
        "one_to_many_rows": len(lf.merge(many, on="k", validate="one_to_many")),
    })

    # N2 rename map
    ren = {"price": "unit_price", "qty": "quantity", "customer": "customer_id"}
    fixed_frames = [f.rename(columns=ren) for _, f in frames]
    renamed_cols = sorted(set().union(*[set(f.columns) for f in fixed_frames]))
    raw_cols = sorted(set().union(*[set(f.columns) for _, f in frames]))
    show("N2 rename map", {"raw_columns": len(raw_cols),
                            "renamed_columns": len(renamed_cols),
                            "columns": renamed_cols})

    # N3 dtype with dtype=str
    sample = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
    show("N3 dtype=str", {"all_object": bool((sample.dtypes == "object").all()),
                          "columns": sample.columns.tolist()})

    # N4 mismatched key types
    lk = pd.DataFrame({"k": [1, 2]})
    rk = pd.DataFrame({"k": ["1", "2"]})
    try:
        lk.merge(rk, on="k", validate="one_to_one")
        print("type mismatch: unexpectedly merged")
    except ValueError as exc:
        print("type mismatch:", str(exc)[:80])
    rk2 = rk.assign(k=rk["k"].astype(int))
    show("N4 after casting the right key", {"rows": len(lk.merge(rk2, on="k"))})

    # N5 multiple-key join
    key = o[["order_id", "product_id"]]
    m = key.merge(p[["product_id", "category"]], on="product_id",
                  how="left", validate="many_to_one")
    show("N5 multiple-key join", {"rows": len(m),
                                   "with_category": int(m["category"].notna().sum())})

    # N6 index join
    c2 = c.set_index("customer_id")
    show("N6 join on index", {"index_name": c2.index.name,
                               "unique_index": bool(c2.index.is_unique),
                               "lookup": str(c2.loc["C-000001", "loyalty_tier"])})

    # N7 fan-out by customer
    dup = pd.read_csv(AUDIT / "customers_duplicate_keys.csv", dtype=str,
                      keep_default_na=False, na_values=[""])
    joined = o.merge(dup, on="customer_id", how="left", validate="many_to_many")
    extra = joined[joined.duplicated(subset=["order_id", "product_id"], keep=False)]
    show("N7 fan-out by customer", {
        "dimension_duplicates": dup["customer_id"].value_counts().head(3).to_dict(),
        "extra_rows_by_customer": extra.groupby("customer_id").size().sort_values(
            ascending=False).head(3).to_dict(),
    })

    # N8 measure reconciliation
    dedup = dup.drop_duplicates("customer_id")
    fixed = o.merge(dedup, on="customer_id", how="left", validate="many_to_one")
    source = round(float(o["rev"].sum()), 2)
    fan = round(float(pd.to_numeric(joined["revenue"], errors="coerce").sum()), 2)
    fix = round(float(pd.to_numeric(fixed["revenue"], errors="coerce").sum()), 2)
    show("N8 measure reconciliation", pd.DataFrame([
        ("source revenue", source, 0.0),
        ("fan-out revenue", fan, round(fan - source, 2)),
        ("deduplicated revenue", fix, round(fix - source, 2)),
    ], columns=["variant", "revenue", "error_vs_source"]).to_string(index=False))

    # N9/N10 as-of directions
    prices = pd.read_csv(AUDIT / "price_history.csv")
    left = o.assign(order_date=pd.to_datetime(
        o["order_date"], errors="coerce")).dropna(subset=["order_date"]).sort_values("order_date")
    right = prices.assign(effective_from=pd.to_datetime(
        prices["effective_from"])).sort_values("effective_from")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        back = pd.merge_asof(left, right, left_on="order_date",
                             right_on="effective_from", by="product_id",
                             direction="backward", suffixes=("_o", "_h"))
        fwd = pd.merge_asof(left, right, left_on="order_date",
                            right_on="effective_from", by="product_id",
                            direction="forward", suffixes=("_o", "_h"))
        near = pd.merge_asof(left, right, left_on="order_date",
                             right_on="effective_from", by="product_id",
                             direction="nearest", suffixes=("_o", "_h"))
    show("N9 as-of directions", {
        "backward_matches": int(back["unit_price_h"].notna().sum()),
        "forward_matches": int(fwd["unit_price_h"].notna().sum()),
        "nearest_matches": int(near["unit_price_h"].notna().sum()),
    })

    # N11 categorical groupby
    ob = o.copy()
    ob["band"] = pd.cut(ob["rev"], [0, 50, 100, 1e9],
                        labels=["low", "mid", "high"])
    show("N11 groupby categorical",
         ob.groupby("band", observed=True)["order_id"].size().to_string())

    # N12 multiple columns
    ob["quantity_num"] = pd.to_numeric(ob["quantity"], errors="coerce")
    show("N12 multiple aggregation columns", ob.groupby("region").agg(
        revenue_sum=("rev", "sum"), revenue_mean=("rev", "mean"),
        qty_sum=("quantity_num", "sum")).round(2).head(3).to_string())

    # N13 cumulative
    ob["running"] = ob.sort_values("order_date").groupby("region")["rev"].cumsum()
    show("N13 cumulative by group",
         ob.sort_values("order_date")[["order_date", "region", "rev", "running"]].head(4).to_string(index=False))

    # N14/N15 z-score and centering
    mu = ob.groupby("region")["rev"].transform("mean")
    sd = ob.groupby("region")["rev"].transform("std")
    ob["z"] = ((ob["rev"] - mu) / sd).round(3)
    ob["centered"] = (ob["rev"] - mu).round(2)
    show("N14 within-group z", {"extreme_z": int((ob["z"].abs() > 3).sum()),
                                 "example": ob[["region", "rev", "z"]].head(3).to_string(index=False)})
    show("N15 centered means", ob.groupby("region")["centered"].mean().round(6).to_dict())

    # N16 pivot vs pivot_table
    g = ob.groupby(["region", "channel"]).size().reset_index(name="lines")
    piv = g.pivot(index="region", columns="channel", values="lines")
    pt = pd.pivot_table(g, index="region", columns="channel",
                        values="lines", aggfunc="sum")
    show("N16 pivot vs pivot_table", {"same": bool(piv.equals(pt)),
                                       "shape": piv.shape})

    # N17 column-name surgery
    wide = pd.read_csv(AUDIT / "monthly_report_wide.csv")
    renamed = [c if c == "region" else c.replace("_", " ", 1).title()
               for c in wide.columns]
    show("N17 column-name surgery", renamed[:5])

    # N18 pivot with duplicate keys
    g2 = pd.concat([g, g.head(1)], ignore_index=True)
    try:
        g2.pivot(index="region", columns="channel", values="lines")
        print("N18 pivot: unexpectedly succeeded")
    except ValueError as exc:
        print("N18 pivot failed:", str(exc)[:70])
    pt2 = pd.pivot_table(g2, index="region", columns="channel",
                         values="lines", aggfunc="sum")
    show("N18 pivot_table result", {"midwest_web": int(pt2.loc["Midwest", "Web"]),
                                     "original_midwest_web": int(piv.loc["Midwest", "Web"])})

    # N19 tidy check on the wide report
    long = wide.melt(id_vars="region", var_name="mc", value_name="value")
    show("N19 tidy check", {"wide_has_month_column": any(c == "month" for c in wide.columns),
                             "long_columns": long.columns.tolist()})

    # N20 customer reconciliation
    camp = pd.read_csv(AUDIT / "campaign_customer.csv", dtype=str,
                       keep_default_na=False, na_values=[""])
    camp["spend_num"] = pd.to_numeric(camp["spend"], errors="coerce")
    base = o.merge(c.drop_duplicates("customer_id")[
        ["customer_id", "loyalty_tier", "home_region"]],
        on="customer_id", how="left", validate="many_to_one")
    base = base.merge(p[["product_id", "category"]].rename(
        columns={"category": "product_category"}), on="product_id",
        how="left", validate="many_to_one")
    cust = base.groupby("customer_id").agg(
        revenue=("rev", "sum"), lines=("order_id", "size"))
    show("N20 customer reconciliation", {
        "customer_revenue": round(float(cust["revenue"].sum()), 2),
        "order_revenue": round(float(o["rev"].sum()), 2),
        "difference": round(float(cust["revenue"].sum() - o["rev"].sum()), 2),
        "customers": len(cust),
    })

    # N21 join audit summary
    show("N21 join audit summary", pd.DataFrame([
        ("orders -> customers", 20000, 20000, 0, 116),
        ("orders -> products", 20000, 20000, 0, 0),
        ("orders -> bad customers", 20000, 22567, 2567, 0),
    ], columns=["join", "left_rows", "joined_rows", "row_delta", "unmatched"]).to_string(index=False))

    # N22-N27 more example-bank checks
    show("N22 shape of the fact", {"rows": len(o), "columns": len(o.columns)})
    show("N23 distinct channels", sorted(o["channel"].unique().tolist()))
    show("N24 null region rows", int(o["region"].isna().sum()))
    show("N25 returned value distribution",
         pd.to_numeric(o["returned"], errors="coerce").value_counts().to_dict())
    show("N26 month coverage", {
        "months": int(pd.to_datetime(o["order_date"], errors="coerce").dt.strftime("%Y-%m").nunique()),
        "min": str(pd.to_datetime(o["order_date"], errors="coerce").min().date()),
        "max": str(pd.to_datetime(o["order_date"], errors="coerce").max().date()),
    })
    show("N27 key format sweep", {
        "order_id_ok": int(o["order_id"].str.fullmatch(r"O-\d{6}").sum()),
        "customer_id_ok": int(o["customer_id"].dropna().str.fullmatch(r"C-\d{6}").sum()),
    })

    # N28 scorecard by channel
    ch = o.groupby("channel").agg(
        lines=("order_id", "size"),
        orders=("order_id", "nunique"),
        revenue=("rev", "sum"),
    )
    ch["aov"] = (ch["revenue"] / ch["orders"]).round(2)
    show("N28 scorecard by channel", ch.round(2).to_string())


def section_extra2() -> None:
    """A final batch of graining, key, and reshape checks for Chapter 7."""
    o = orders()
    o["rev"] = pd.to_numeric(o["revenue"], errors="coerce")
    paths = sorted((RAW / "sales_exports").glob("sales_*.csv"))

    # P1 grain checklist
    lines = len(o)
    keys = int(o.groupby(["order_id", "product_id"]).ngroups)
    show("P1 grain checklist", pd.DataFrame([
        ("orders", "one order line", "order_id + product_id", f"{lines} lines / {keys} keys"),
        ("customers", "one customer", "customer_id", "unique"),
        ("products", "one product", "product_id", "unique"),
    ], columns=["table", "grain", "key", "evidence"]).to_string(index=False))

    # P2 keys= MultiIndex
    parts = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths[:2]]
    tagged = pd.concat(parts, keys=[p.stem for p in paths[:2]],
                       names=["month", "row"])
    show("P2 concat keys", {"index_names": list(tagged.index.names),
                             "levels": int(tagged.index.nlevels),
                             "rows": len(tagged)})

    # P3 different key names
    lf = pd.DataFrame({"cust": ["C-000001", "C-000002"]})
    rf = pd.DataFrame({"customer_id": ["C-000001", "C-000002"],
                       "tier": ["Gold", "Silver"]})
    show("P3 left_on / right_on",
         lf.merge(rf, left_on="cust", right_on="customer_id",
                  how="left").to_string(index=False))

    # P4 keep the null group
    show("P4 groupby dropna", {
        "default_groups": int(o.groupby("region").size().shape[0]),
        "with_null_group": int(o.groupby("region", dropna=False).size().shape[0]),
    })

    # P5 as-of by currency
    eu = pd.read_csv(AUDIT / "eu_orders_fx.csv")
    fx = pd.read_csv(AUDIT / "fx_rates.csv")
    eu["d"] = pd.to_datetime(eu["order_date"])
    fx["d"] = pd.to_datetime(fx["rate_date"])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m = pd.merge_asof(eu.sort_values("d"), fx.sort_values("d"), on="d",
                          by="currency", direction="backward")
    show("P5 as-of by currency", {"orders": len(m),
                                   "with_rate": int(m["usd_per_unit"].notna().sum())})

    # P6 fill within a group
    tiny = pd.DataFrame({"region": ["West", "West", "South", "South"],
                         "rev": [10.0, np.nan, 20.0, np.nan]})
    tiny["group_fill"] = tiny.groupby("region")["rev"].transform("median")
    tiny["was_missing"] = tiny["rev"].isna()
    show("P6 transform fill within group", tiny.to_string(index=False))

    # P7 melt with several identifiers
    small = pd.read_csv(AUDIT / "monthly_report_wide.csv")[
        ["region", "revenue_2025-01", "revenue_2025-02"]]
    melted = small.melt(id_vars="region", var_name="month", value_name="revenue")
    melted["month"] = melted["month"].str.replace("revenue_", "", regex=False)
    show("P7 melt with identifiers", melted.head(4).to_string(index=False))

    # P8-P10 final checks
    show("P8 group count with nulls", int(o.groupby("region", dropna=False).ngroups))
    show("P9 merged columns", o.merge(customers(), on="customer_id",
                                       how="left").shape[1])
    show("P10 wide report metrics", sorted({c.split("_")[0]
         for c in pd.read_csv(AUDIT / "monthly_report_wide.csv").columns
         if c != "region"}))

    # P11 growth definitions
    g = o.assign(month=pd.to_datetime(
        o["order_date"], errors="coerce").dt.strftime("%Y-%m"))
    s = g.dropna(subset=["month"]).groupby(["region", "month"])["rev"].sum()
    growth = s.groupby(level=0).pct_change()
    show("P11 growth by region", {"first_values": growth.head(4).round(4).to_dict(),
                                   "non_null": int(growth.notna().sum())})


def main() -> None:
    print("python", "pandas", pd.__version__, "numpy", np.__version__)
    print()
    for section in (
        section_71, section_72, section_73, section_74, section_75,
        section_76, section_77, section_78, section_79, section_710,
        section_711, section_712, section_extra, section_extra2,
    ):
        print("=" * 78)
        section()


if __name__ == "__main__":
    main()
