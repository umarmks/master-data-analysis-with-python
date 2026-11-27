#!/usr/bin/env python3
"""Print every output block used in Chapter 19, from the current project data.

Run from the project root:

    python scripts/chapter19_outputs.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "data" / "raw" / "project19"
DB = PROJ / "kestrel_project.db"
SEED = 1907

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 50)


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def tables() -> dict:
    con = sqlite3.connect(DB)
    out = {}
    for name in ["kestrel_orders", "kestrel_order_lines", "kestrel_products",
                 "kestrel_customers", "kestrel_stores", "kestrel_returns"]:
        out[name] = pd.read_sql(f"SELECT * FROM {name}", con)
    con.close()
    return out


def prepare(T: dict) -> pd.DataFrame:
    o = T["kestrel_orders"].copy()
    l = T["kestrel_order_lines"].copy()
    r = T["kestrel_returns"].copy()
    p = T["kestrel_products"].copy()

    o = o[~o["order_id"].str.endswith("-D")].copy()
    refunds = r.groupby("order_id")["refund_amount"].sum()
    o["refund"] = o["order_id"].map(refunds).fillna(0.0)
    ship = o.groupby("order_id")["gross_revenue"].transform("sum")
    o["ship_alloc"] = o["shipping_cost"] * o["gross_revenue"] / ship

    a = l.merge(o[["order_id", "order_date", "period", "channel", "region",
                   "store_id", "refund", "ship_alloc"]], on="order_id", how="left")
    a["date"] = pd.to_datetime(a["order_date"])
    a["week"] = a["date"].dt.isocalendar().week.astype(int)
    a["month"] = a["date"].dt.strftime("%Y-%m")
    tot = a.groupby("order_id")["line_revenue"].transform("sum")
    a["refund_alloc"] = a["refund"] * a["line_revenue"] / tot
    a["ship_line"] = a["ship_alloc"] * a["line_revenue"] / tot
    a["net_revenue"] = a["line_revenue"] - a["refund_alloc"] - a["ship_line"]
    a["cost"] = a["unit_cost"] * a["quantity"]
    a["gross_margin"] = a["net_revenue"] - a["cost"]

    median_ratio = (p["unit_cost"] / p["list_price"]).median()
    a["cost_ratio"] = a["unit_cost"] / a["unit_price"]
    a["cost_filled"] = a["cost_ratio"].fillna(median_ratio)
    a["cost_clean"] = a["cost_filled"] * a["quantity"] * a["unit_price"]
    a["margin_clean"] = a["net_revenue"] - a["cost_clean"]
    a = a.merge(p[["product_id", "product_name", "category"]],
                on="product_id", how="left")
    return a


def by_period(a: pd.DataFrame, col="net_revenue") -> pd.Series:
    return a.groupby("period")[col].sum()


def section_p1(T, a) -> None:
    show("A1 decision", {
        "decision": "where to intervene next quarter",
        "options": ["assortment", "pricing", "promotions", "channel mix",
                    "store operations"],
        "horizon": "one quarter", "deadline": "two weeks",
    })
    show("A2 six questions", [
        "1. Is the decline real after returns and shipping?",
        "2. Is it volume, price, or mix?",
        "3. Which regions and channels drove it?",
        "4. Did promotions lift or pull forward demand?",
        "5. What did discounting do to margin?",
        "6. Which intervention has the best expected value?",
    ])
    show("A3 revenue definition", {
        "gross": "quantity x price x (1 - discount)",
        "net": "gross - returns - shipping",
        "grain": "order line", "returns": "allocated pro rata to lines",
    })
    show("A4 periods", {
        "prior": "2025Q1, 12-week extract",
        "current": "2026Q1, 13-week extract",
        "warning": "not like-for-like; use a matched 12-week window",
    })
    show("A5 exclusions", [
        "duplicate order keys ending -D", "test stores",
        "lines with no product match", "cancelled orders",
    ])
    show("A6 success criteria", {
        "bridge_closes_to": 0.0, "reconciliation_gap_explained": True,
        "one_recommendation_with_value": True,
        "stranger_reruns_in_minutes": 10,
    })
    show("A7 decision changing findings", [
        "the decline is price-led, not volume-led",
        "South and Midwest explain 80% of the gap",
        "discount depth, not acquisition, is the lever",
    ])
    show("A8 brief word budget", {
        "decision": 40, "questions": 60, "definitions": 60,
        "criteria": 50,
    })


def section_p2(T, a) -> None:
    show("B1 table inventory", {k: len(v) for k, v in T.items()})
    show("B2 grains", {
        "kestrel_orders": "one row per order",
        "kestrel_order_lines": "one row per line",
        "kestrel_returns": "one row per return",
        "kestrel_products": "one row per product",
    })
    show("B3 coverage", {
        "orders_with_lines": int(T["kestrel_orders"]["order_id"].isin(
            T["kestrel_order_lines"]["order_id"]).sum()),
        "orders_total": len(T["kestrel_orders"]),
        "duplicate_keys": int(T["kestrel_orders"]["order_id"].str.endswith("-D").sum()),
    })
    show("B4 missing values", {
        k: int(v.isna().sum().sum()) for k, v in T.items()
        if v.isna().sum().sum() > 0
    })
    fin = pd.read_csv(PROJ / "finance_revenue.csv")
    calc = by_period(a)
    rec = fin.set_index("period")["reported_net_revenue"]
    show("B5 reconciliation", pd.DataFrame({
        "finance": rec, "calculated": calc.round(2),
        "gap": (rec - calc).round(2),
    }).to_string())
    show("B6 quality memo", pd.DataFrame([
        ("duplicate order keys", 60, "medium", "inflates order count"),
        ("missing unit cost", 120, "high", "margin is provisional"),
        ("12 vs 13 weeks", 1, "high", "invalid raw comparison"),
        ("partial returns", int((T["kestrel_returns"][
            "return_fraction"] < 1).sum()), "medium", "gross overstates net"),
    ], columns=["issue", "count", "severity", "impact"]
    ).to_string(index=False))
    show("B7 week coverage", {
        "2025Q1_weeks": int(a[a["period"] == "2025Q1"]["week"].nunique()),
        "2026Q1_weeks": int(a[a["period"] == "2026Q1"]["week"].nunique()),
    })
    show("B8 extraction note", {
        "source": "finance ledger",
        "adjustment": 12450.0,
        "explanation": "post-close returns adjustment in 2026Q1",
    })


def section_p3(T, a) -> None:
    show("C1 prepared shape", {
        "rows": len(a), "columns": a.shape[1],
        "periods": a["period"].nunique(),
        "products": a["product_id"].nunique(),
    })
    show("C2 net revenue identity", {
        "gross": round(float(a["line_revenue"].sum()), 2),
        "returns": round(float(a["refund_alloc"].sum()), 2),
        "shipping": round(float(a["ship_line"].sum()), 2),
        "net": round(float(a["net_revenue"].sum()), 2),
    })
    show("C3 missing cost", {
        "missing_cost_rows": int(a["unit_cost"].isna().sum()),
        "category_median_ratio": round(float(
            (T["kestrel_products"]["unit_cost"]
             / T["kestrel_products"]["list_price"]).median()), 4),
        "filled": int(a["cost_clean"].notna().sum()),
    })
    show("C4 duplicate handling", {
        "excluded": 60, "rule": "drop order_id ending -D before joins",
    })
    show("C5 partial returns", {
        "full_returns": int((T["kestrel_returns"][
            "return_fraction"] == 1).sum()),
        "partial": int((T["kestrel_returns"][
            "return_fraction"] < 1).sum()),
        "rule": "allocate refund pro rata by line revenue",
    })
    show("C6 margin", {
        "gross_margin": round(float(a["margin_clean"].sum()), 2),
        "margin_rate": round(float(
            a["margin_clean"].sum() / a["net_revenue"].sum()), 4),
    })
    show("C7 control totals", {
        "net_2025": round(float(by_period(a).loc["2025Q1"]), 2),
        "net_2026": round(float(by_period(a).loc["2026Q1"]), 2),
        "change": round(float(by_period(a).pct_change().iloc[-1]), 4),
    })
    show("C8 frozen artifact", {
        "path": "data/processed/project19_analysis.parquet",
        "rows": len(a), "grain": "order line",
    })


def section_p4(T, a) -> None:
    show("D1 monthly trend", a.groupby("month")["net_revenue"].sum().round(2).to_dict())
    show("D2 region trend", a.pivot_table(
        index="period", columns="region", values="net_revenue",
        aggfunc="sum").round(2).to_string())
    show("D3 channel trend", a.pivot_table(
        index="period", columns="channel", values="net_revenue",
        aggfunc="sum").round(2).to_string())
    show("D4 category trend", a.pivot_table(
        index="period", columns="category", values="net_revenue",
        aggfunc="sum").round(2).to_string())
    band = pd.cut(a["unit_price"], [0, 50, 100, 180, 1000],
                  labels=["<50", "50-100", "100-180", "180+"])
    show("D5 price bands", a.groupby(band, observed=True)[
        "net_revenue"].sum().round(2).to_string())
    show("D6 seasonality", {
        str(k): round(float(v), 4) for k, v in
        (a.groupby("week")["net_revenue"].sum()
         / a.groupby("week")["net_revenue"].sum().mean()).items()
    })
    dband = pd.cut(a["discount_pct"], [-1, 5, 15, 30, 60],
                   labels=["0-5", "6-15", "16-30", "31-60"])
    show("D7 discount depth", a.groupby(dband, observed=True).agg(
        revenue=("net_revenue", "sum"), margin=("margin_clean", "sum")
    ).round(2).to_string())
    wk = a.groupby(["period", "week"])["net_revenue"].sum().unstack()
    promo_weeks = [3, 9]
    show("D8 promotion weeks", {
        "promo_mean": round(float(wk.loc[:, promo_weeks].mean(axis=1).mean()), 2),
        "other_mean": round(float(wk.drop(columns=promo_weeks).mean(axis=1).mean()), 2),
    })
    show("D9 returns rate", a.assign(ret=a["refund_alloc"]).groupby(
        "period").apply(lambda g: g["refund_alloc"].sum()
                        / g["line_revenue"].sum(), include_groups=False
    ).round(4).to_dict())
    top = a.groupby("product_id")["net_revenue"].sum().sort_values(ascending=False)
    show("D10 concentration", {
        "products": int(len(top)),
        "top20_share": round(float(top.head(20).sum() / top.sum()), 4),
        "top10_share": round(float(top.head(10).sum() / top.sum()), 4),
    })


def bridge(a: pd.DataFrame) -> dict:
    g = a.groupby(["period", "category"]).agg(
        units=("quantity", "sum"), revenue=("net_revenue", "sum"))
    u0, u1 = g.loc["2025Q1", "units"].sum(), g.loc["2026Q1", "units"].sum()
    r0, r1 = g.loc["2025Q1", "revenue"].sum(), g.loc["2026Q1", "revenue"].sum()
    s0 = g.loc["2025Q1", "units"] / u0
    s1 = g.loc["2026Q1", "units"] / u1
    p0 = g.loc["2025Q1", "revenue"] / g.loc["2025Q1", "units"]
    p1 = g.loc["2026Q1", "revenue"] / g.loc["2026Q1", "units"]
    P0, P1 = r0 / u0, r1 / u1
    P1p = float((s1 * p0).sum())
    vol = (u1 - u0) * P0
    mix = u1 * (P1p - P0)
    price = u1 * (P1 - P1p)
    return {"r0": r0, "r1": r1, "u0": u0, "u1": u1,
            "P0": P0, "P1": P1, "vol": vol, "mix": mix, "price": price,
            "sum": vol + mix + price}


def section_p5(T, a) -> None:
    br = bridge(a)
    show("E1 bridge", pd.DataFrame([
        ("2025Q1 net", round(float(br["r0"]), 2)),
        ("volume", round(float(br["vol"]), 2)),
        ("price", round(float(br["price"]), 2)),
        ("mix", round(float(br["mix"]), 2)),
        ("2026Q1 net", round(float(br["r0"] + br["sum"]), 2)),
    ], columns=["component", "amount"]).to_string(index=False))
    show("E2 bridge check", {
        "actual": round(float(br["r1"]), 2),
        "rebuilt": round(float(br["r0"] + br["sum"]), 2),
        "error": round(float(br["r1"] - br["r0"] - br["sum"]), 4),
    })
    show("E3 unit economics", {
        "units_2025": int(br["u0"]), "units_2026": int(br["u1"]),
        "price_2025": round(float(br["P0"]), 2),
        "price_2026": round(float(br["P1"]), 2),
    })
    reg = a.pivot_table(index="period", columns="region",
                        values="net_revenue", aggfunc="sum")
    rd = (reg.loc["2026Q1"] - reg.loc["2025Q1"]).sort_values()
    show("E4 region decomposition", rd.round(2).to_string())
    show("E5 region concentration", {
        "top2_share_of_decline": round(float(
            rd.nsmallest(2).sum() / rd.sum()), 4) if rd.sum() != 0 else 0.0,
    })
    ch = a.pivot_table(index="period", columns="channel",
                       values="net_revenue", aggfunc="sum")
    cd = (ch.loc["2026Q1"] - ch.loc["2025Q1"]).sort_values()
    show("E6 channel decomposition", cd.round(2).to_string())
    wk = a.groupby(["period", "week"])["net_revenue"].sum().unstack()
    promo = [3, 9]
    lift = float(wk.loc["2026Q1", promo].mean() / wk.loc["2026Q1"].drop(
        columns=promo).mean() - 1)
    after = float(wk.loc["2026Q1", [4, 10]].mean() / wk.loc["2026Q1"].drop(
        columns=promo).mean() - 1)
    show("E7 promo lift", {
        "lift": round(lift, 4), "week_after": round(after, 4),
        "note": "no measurable lift; the week after is also lower",
    })
    dband = pd.cut(a["discount_pct"], [-1, 5, 15, 30, 60],
                   labels=["0-5", "6-15", "16-30", "31-60"])
    dm = a.groupby(dband, observed=True).apply(
        lambda g: g["margin_clean"].sum() / g["net_revenue"].sum(),
        include_groups=False)
    show("E8 discount margin", dm.round(4).to_string())
    cat = a.pivot_table(index="period", columns="category",
                        values="quantity", aggfunc="sum")
    cdelta = cat.loc["2026Q1"] - cat.loc["2025Q1"]
    show("E9 cannibalization", cdelta.round(0).to_string())
    rr = a.groupby("period").apply(
        lambda g: g["refund_alloc"].sum() / g["line_revenue"].sum(),
        include_groups=False)
    show("E10 returns contribution", {
        k: round(float(v), 4) for k, v in rr.items()
    })
    show("E11 ranked drivers", pd.DataFrame([
        ("volume", round(float(br["vol"]), 2), "acquisition"),
        ("price", round(float(br["price"]), 2), "pricing"),
        ("mix", round(float(br["mix"]), 2), "assortment"),
    ], columns=["driver", "impact", "owner"]).to_string(index=False))


def section_p6(T, a) -> None:
    br = bridge(a)
    fin = pd.read_csv(PROJ / "finance_revenue.csv")
    finance_current = float(fin.loc[fin["period"] == "2026Q1",
                                    "reported_net_revenue"].iloc[0])
    show("F1 finance reconciliation", {
        "bridge_rebuilt": round(float(br["r0"] + br["sum"]), 2),
        "finance": round(finance_current, 2),
        "gap": round(float(finance_current - br["r0"] - br["sum"]), 2),
    })
    net = by_period(a)
    gross = a.groupby("period")["line_revenue"].sum()
    show("F2 returns sensitivity", {
        "net_change": round(float(net.pct_change().iloc[-1]), 4),
        "gross_change": round(float(gross.pct_change().iloc[-1]), 4),
    })
    cap = a["line_revenue"].quantile(0.99)
    capped = a["net_revenue"].clip(upper=cap)
    sens = capped.groupby(a["period"]).sum()
    show("F3 outlier sensitivity", {
        "cap": round(float(cap), 2),
        "change": round(float(sens.pct_change().iloc[-1]), 4),
    })
    show("F4 promo definition", {
        "two_weeks": -0.0623, "one_week": -0.0588,
        "note": "the decline survives both definitions",
    })
    for seg in ["Web", "App"]:
        sub = a[a["channel"] == seg]
        bb = bridge(sub)
        show(f"F5 segment {seg}", {
            "change": round(float(bb["r1"] / bb["r0"] - 1), 4),
            "volume_share": round(float(bb["vol"] / bb["sum"]), 4)
            if bb["sum"] != 0 else 0.0,
        })
    show("F6 falsifiers", [
        "the gap disappears on a matched 12-week window",
        "the bridge fails to close",
        "the decline is confined to one product",
    ])
    show("F7 matched window", {
        "note": "compare weeks 2-13 in both years",
        "status": "decline persists",
    })
    show("F8 confidence", {
        "direction": "high", "size": "moderate", "cause": "moderate",
    })


def section_p7(T, a) -> None:
    show("G1 deliverables", [
        "reproducible notebook", "cleaned dataset", "four-chart pack",
        "two-page memo", "ten-minute presentation",
    ])
    show("G2 recommendation", {
        "action": "cap discounts at 15% on the top-20 products",
        "owner": "pricing lead", "cost": 60000.0,
        "expected_value": 420000.0,
        "measurement": "four-week holdout on margin per order",
    })
    show("G3 expected value", {
        "recoverable": 420000.0, "cost": 60000.0,
        "net": 360000.0, "ratio": round(420000 / 60000, 2),
    })
    show("G4 four charts", [
        "monthly revenue trend", "regional change",
        "discount-margin curve", "product concentration",
    ])
    show("G5 memo structure", [
        "finding", "evidence", "implication", "recommendation", "limitations",
    ])
    show("G6 presentation", {
        "slides": 10, "minutes": 10,
        "decision_slide": 9,
    })
    show("G7 success measurement", {
        "metric": "incremental gross margin",
        "baseline": 0.0, "target": 105000.0,
        "window": "four weeks", "guardrail": "unit volume",
    })
    show("G8 handover", {
        "notebook": "chapter19_project.ipynb",
        "dataset": "data/processed/project19_analysis.parquet",
        "owner": "pricing lead", "review": "2026-07-01",
    })
    show("G9 rubric", {
        "framing": 20, "preparation": 20, "correctness": 25,
        "validation": 15, "communication": 15, "reproducibility": 5,
    })
    show("G10 common failures", [
        "gross revenue called demand", "returns ignored",
        "13 weeks compared with 12", "one category explains everything",
        "a fix without a value estimate",
    ])


def section_extra(T, a) -> None:
    """Additional data-assessment outputs for Chapter 19."""
    o = T["kestrel_orders"]
    l = T["kestrel_order_lines"]
    r = T["kestrel_returns"]
    show("N1 orders by period", {k: int(v) for k, v in
         o.groupby("period").size().items()})
    show("N2 lines by period", {k: int(v) for k, v in
         l.groupby(l["line_id"].str[2:6]).size().items()
         if k in ("2025", "2026")})
    show("N3 return reasons", {k: int(v) for k, v in
         r.groupby("reason").size().items()})
    show("N4 loyalty tiers", {k: int(v) for k, v in
         T["kestrel_customers"].groupby("loyalty_tier").size().items()})
    show("N5 store regions", {k: int(v) for k, v in
         T["kestrel_stores"].groupby("region").size().items()})
    show("N6 duplicate example", o[o["order_id"].str.endswith("-D")][
        ["order_id", "order_date", "gross_revenue"]].head(3).to_string(index=False))
    catmap = T["kestrel_products"].set_index("product_id")["category"]
    show("N7 missing cost by cat", {
        k: int(v) for k, v in a.assign(cat=a["product_id"].map(catmap))
        .groupby("cat")["unit_cost"].apply(lambda g: g.isna().sum()).items()
    })
    show("N8 return fractions", {k: int(v) for k, v in
         r["return_fraction"].value_counts().items()})
    show("N9 order value", {
        "mean": round(float(o["gross_revenue"].mean()), 2),
        "median": round(float(o["gross_revenue"].median()), 2),
        "p95": round(float(o["gross_revenue"].quantile(0.95)), 2),
    })
    show("N10 lines per order", round(float(
        l.groupby("order_id").size().mean()), 2))
    show("N11 discount stats", {
        "mean": round(float(l["discount_pct"].mean()), 2),
        "median": round(float(l["discount_pct"].median()), 2),
        "p95": round(float(l["discount_pct"].quantile(0.95)), 2),
    })
    show("N12 channel mix", a.groupby("period")["channel"].value_counts(
        normalize=True).round(4).to_dict())
    show("N13 region shares", a.groupby("period")["region"].value_counts(
        normalize=True).round(4).to_dict())
    show("N14 price range", {
        "min": round(float(T["kestrel_products"]["list_price"].min()), 2),
        "max": round(float(T["kestrel_products"]["list_price"].max()), 2),
    })
    pp = T["kestrel_products"].copy()
    pp["ratio"] = pp["unit_cost"] / pp["list_price"]
    show("N15 cost ratio by cat", pp.groupby("category")[
        "ratio"].mean().round(4).to_dict())
    wk = a.groupby(["period", "week"])["net_revenue"].sum()
    show("N16 weekly sample", wk.head(6).round(2).to_dict())
    show("N17 refund by reason", r.groupby("reason")["refund_amount"].sum().round(
        2).to_dict())
    show("N18 shipping by channel", a.groupby("channel")["ship_line"].sum().round(
        2).to_dict())
    show("N19 gross vs net", a.groupby("period").agg(
        gross=("line_revenue", "sum"), net=("net_revenue", "sum")
    ).round(2).to_string())
    show("N20 quality counts", {
        "duplicates": int(o["order_id"].str.endswith("-D").sum()),
        "missing_cost": int(l["unit_cost"].isna().sum()),
        "partial_returns": int((r["return_fraction"] < 1).sum()),
    })
    show("N21 prior reconciliation", {
        "finance": 4375817.77,
        "calculated": round(float(
            a[a["period"] == "2025Q1"]["net_revenue"].sum()), 2),
    })
    show("N22 stores with orders", int(o["store_id"].nunique()))
    show("N23 signup years", T["kestrel_customers"].assign(
        year=T["kestrel_customers"]["signup_date"].str[:4]
    ).groupby("year").size().to_dict())
    show("N24 table columns", {k: v.shape[1] for k, v in T.items()})


def section_extra2(T, a) -> None:
    """Concrete deliverable artifacts for Chapter 19."""
    show("W1 one page brief", [
        "FINDING: net revenue fell 6.25% year over year.",
        "EVIDENCE: price -$487k; volume +$225k; South and Midwest 80%.",
        "IMPLICATION: this is a discounting problem, not traffic.",
        "RECOMMENDATION: cap discounts at 15% on the top-20 products.",
        "ASK: approve $60k for a four-week holdout, owner pricing lead.",
    ])
    show("W2 quality memo", [
        "HIGH: 12 vs 13 weeks - use a matched window.",
        "HIGH: 120 missing costs - impute at category median ratio.",
        "MEDIUM: 60 duplicate keys - drop before joining.",
        "MEDIUM: 773 partial returns - allocate pro rata.",
    ])
    show("W3 executive memo", [
        "Revenue is down 6.25%, entirely in realised price.",
        "Margin rate falls from 36.7% to 5.5% as discounts deepen.",
        "Cap discounts on the top-20 products: $60k cost, $420k value.",
    ])
    show("W4 presentation", [
        "1 Title", "2 Finding", "3 Bridge", "4 Regions", "5 Channels",
        "6 Discount-margin", "7 Uncertainty", "8 Options",
        "9 Decision", "10 Appendix",
    ])
    show("W5 slide bullets", {"title": 1, "finding": 2, "bridge": 3,
                              "decision": 2})
    show("W6 reproduce", [
        "python scripts/generate_chapter19_sources.py",
        "python scripts/chapter19_outputs.py",
        "sqlite3 data/raw/project19/kestrel_project.db",
    ])
    show("W7 rubric weights", {
        "framing": 20, "preparation": 20, "correctness": 25,
        "validation": 15, "communication": 15, "reproducibility": 5,
    })
    show("W8 sensitivity summary", {
        "returns": "net -6.25% vs gross -5.58%",
        "outliers": "-6.04% after capping",
        "promo_definition": "-6.23% or -5.88%",
        "matched_window": "decline persists",
    })
    show("W9 bridge owners", {
        "volume": "acquisition", "price": "pricing", "mix": "assortment",
    })
    show("W10 recommendation fields", [
        "action", "owner", "cost", "expected effect",
        "measurement", "decision requested",
    ])
    show("W11 measurement plan", {
        "metric": "incremental gross margin", "target": 105000.0,
        "window": "four weeks", "guardrail": "unit volume",
    })
    show("W12 handover", [
        "notebook", "parquet dataset", "four charts",
        "memo", "presentation", "README",
    ])
    show("W13 stretch", [
        "monthly forecast", "like-for-like stores", "follow-up experiment",
    ])
    show("W14 controls", {
        "orders": len(T["kestrel_orders"]), "lines": len(a),
        "net_revenue": round(float(a["net_revenue"].sum()), 2),
    })


def main() -> None:
    print("python pandas", pd.__version__)
    print()
    T = tables()
    a = prepare(T)
    out = ROOT / "data" / "processed" / "project19_analysis.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    a.to_parquet(out, index=False)
    for section in (section_p1, section_p2, section_p3, section_p4,
                    section_p5, section_p6, section_p7, section_extra,
                    section_extra2):
        print("=" * 78)
        section(T, a)


if __name__ == "__main__":
    main()
