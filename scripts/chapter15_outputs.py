#!/usr/bin/env python3
"""Print every output block used in Chapter 15, from the current files.

Run from the project root:

    python scripts/chapter15_outputs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BIZ = ROOT / "data" / "raw" / "business"
SEED = 1507

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 80)


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(BIZ / f"biz_{name}.csv")


# =============================================== 15.1 metrics and KPIs
def section_151() -> None:
    m = load("monthly")
    c = load("campaigns")
    cu = load("customers")
    total_rev = float(m["revenue"].sum())
    total_ord = int(m["orders"].sum())

    rows = [
        ("revenue", "sum of order revenue", total_rev, "USD", "order line"),
        ("orders", "count of orders", total_ord, "count", "order"),
        ("sessions", "count of sessions", int(m["sessions"].sum()), "count", "session"),
        ("conversion", "orders / sessions", total_ord / m["sessions"].sum(), "share", "session"),
        ("aov", "revenue / orders", total_rev / total_ord, "USD", "order"),
        ("gross_margin", "(revenue - cogs) / revenue",
         (total_rev - m["cogs"].sum()) / total_rev, "share", "order"),
        ("return_rate", "returns / revenue", m["returns"].sum() / total_rev,
         "share", "order"),
        ("new_customers", "count of first-time buyers", int(cu.shape[0]), "count", "customer"),
        ("spend", "sum of platform spend", float(c["spend"].sum()), "USD", "campaign"),
        ("roas", "attributed revenue / spend",
         float(m.loc[m["month"].str.startswith("2025"), "revenue"].sum()
               / c["spend"].sum()), "ratio", "campaign"),
    ]
    cat = pd.DataFrame(rows, columns=["metric", "definition", "value",
                                       "unit", "grain"])
    cat["value"] = [
        f"{float(v):,.2f}" if u in ("USD", "count") else f"{float(v):.4f}"
        for v, u in zip(cat["value"], cat["unit"])
    ]
    show("M1 metrics catalog", cat.to_string(index=False))

    show("M2 count vs rate", {
        "orders": total_ord,
        "sessions": int(m["sessions"].sum()),
        "conversion_rate": round(float(total_ord / m["sessions"].sum()), 4),
        "warning": "a rate without its numerator and denominator is ambiguous",
    })

    show("M3 leading vs lagging", {
        "leading": ["sessions", "add_to_cart rate", "support tickets"],
        "lagging": ["revenue", "churn", "margin"],
    })

    show("M4 metric tree", {
        "revenue": "sessions x conversion x AOV",
        "sessions": "traffic sources x visit rate",
        "conversion": "product view x cart x checkout",
        "aov": "units per order x price x (1 - discount)",
    })

    show("M5 guardrails", [
        "return rate must not rise with revenue",
        "margin per order must not fall with discount",
        "support tickets per 100 orders is a counter-metric to speed",
    ])

    show("M6 goodharts law", [
        "a metric used as a target stops being a measure",
        "publish the definition and a counter-metric together",
        "review the metric when behaviour changes, not only the number",
    ])

    show("M7 benchmark vs target", {
        "benchmark": "the comparison group or prior period",
        "target": "a commitment chosen by the business",
        "note": "a target can be missed while beating the benchmark",
    })

    show("M8 versioning", {
        "metric": "conversion rate",
        "v1_denominator": "all sessions",
        "v2_denominator": "sessions with a product view",
        "effect": "the same week reports a different rate after the change",
    })


# ================================================= 15.2 sales and revenue
def section_152() -> None:
    m = load("monthly")
    p = load("products")
    q3 = m[m["month"].between("2024-07", "2024-09")]
    q4 = m[m["month"].between("2024-10", "2024-12")]
    a, b = q3["revenue"].sum(), q4["revenue"].sum()
    show("R1 quarterly revenue", {"Q3": round(float(a), 2),
                                   "Q4": round(float(b), 2),
                                   "change": round(float(b - a), 2),
                                   "pct": round(float(b / a - 1), 4)})

    def agg(frame):
        sess = float(frame["sessions"].sum())
        orders = int(frame["orders"].sum())
        rev = float(frame["revenue"].sum())
        return sess, sess and orders / sess, rev / orders

    s3, c3, v3 = agg(q3)
    s4, c4, v4 = agg(q4)
    identity = s4 * c4 * v4
    show("R2 revenue identity", {
        "Q3": {"sessions": int(s3), "conversion": round(c3, 4), "aov": round(v3, 2)},
        "Q4": {"sessions": int(s4), "conversion": round(c4, 4), "aov": round(v4, 2)},
        "rebuilt_Q4": round(identity, 2), "actual_Q4": round(float(b), 2),
    })

    # a simple sequential bridge on the identity
    step1 = s4 * c3 * v3
    step2 = s4 * c4 * v3
    show("R3 sequential bridge", pd.DataFrame([
        ("Q3 revenue", round(float(a), 2)),
        ("volume effect", round(step1 - a, 2)),
        ("conversion effect", round(step2 - step1, 2)),
        ("price/mix effect", round(identity - step2, 2)),
        ("Q4 revenue", round(identity, 2)),
    ], columns=["component", "amount"]).to_string(index=False))

    p["share"] = p["revenue_q4"] / p["revenue_q4"].sum()
    p = p.sort_values("revenue_q4", ascending=False)
    p["cum_share"] = p["share"].cumsum()
    abc = pd.cut(p["cum_share"], [0, 0.8, 0.95, 1.0],
                 labels=["A", "B", "C"]).astype(str)
    show("R4 ABC classification", pd.DataFrame({
        "class": abc.value_counts().reindex(["A", "B", "C"]).to_dict()
    }).to_string())
    show("R5 pareto", {
        "top_3_share": round(float(p["share"].head(3).sum()), 4),
        "top_5_share": round(float(p["share"].head(5).sum()), 4),
        "products": int(len(p)),
    })

    show("R6 margin and returns", {
        "gross_margin_pct": round(float(
            (q4["revenue"].sum() - q4["cogs"].sum()) / q4["revenue"].sum()), 4),
        "return_rate_pct": round(float(
            q4["returns"].sum() / q4["revenue"].sum()), 4),
        "shipping_per_order": round(float(
            q4["shipping"].sum() / q4["orders"].sum()), 2),
    })

    by_channel = q4.groupby("channel").agg(
        revenue=("revenue", "sum"), orders=("orders", "sum"))
    by_channel["aov"] = (by_channel["revenue"] / by_channel["orders"]).round(2)
    show("R7 channel mix", by_channel.round(2).to_string())

    show("R8 promotion lift", {
        "december_revenue": round(float(m[m["month"] == "2024-12"]["revenue"].sum()), 2),
        "november_revenue": round(float(m[m["month"] == "2024-11"]["revenue"].sum()), 2),
        "lift": round(float(m[m["month"] == "2024-12"]["revenue"].sum()
                            / m[m["month"] == "2024-11"]["revenue"].sum() - 1), 4),
        "caveat": "seasonality and pull-forward are not removed",
    })


# ======================================================== 15.3 marketing
def section_153() -> None:
    c = load("campaigns")
    ev = load("events")
    g = c.groupby("platform").agg(spend=("spend", "sum"),
                                   clicks=("clicks", "sum"),
                                   platform=("platform_conversions", "sum"),
                                   first=("first_touch", "sum"),
                                   last=("last_touch", "sum"),
                                   linear=("linear_touch", "sum"))
    g["ctr"] = g["platform"] / g["clicks"]
    g["cpc"] = g["spend"] / g["clicks"]
    g["cpa_platform"] = g["spend"] / g["platform"]
    g["cpa_last"] = g["spend"] / g["last"]
    show("K1 platform metrics", g.round(4).to_string())

    aov = float(load("monthly")["revenue"].sum() / load("monthly")["orders"].sum())
    revenue_by_model = pd.DataFrame({
        "first_touch": g["first"] * aov,
        "last_touch": g["last"] * aov,
        "linear": g["linear"] * aov,
        "platform": g["platform"] * aov,
    })
    show("K2 attributed revenue", {k: round(float(v), 2)
                                    for k, v in revenue_by_model.sum().items()})
    show("K3 ROAS by model", (revenue_by_model.sum() / g["spend"].sum()).round(3).to_dict())

    show("K4 platform vs warehouse", {
        "platform_conversions": int(g["platform"].sum()),
        "warehouse_last_touch": int(g["last"].sum()),
        "gap": int(g["platform"].sum() - g["last"].sum()),
        "note": "different counting windows and attribution, not necessarily fraud",
    })

    funnel = ev.groupby("event")["user_id"].nunique().reindex(
        ["login", "view_product", "add_to_cart", "checkout", "purchase"])
    show("K5 product funnel", funnel.to_string())
    show("K6 step conversion", (funnel / funnel.shift(1)).round(3).to_dict())

    spend = float(c["spend"].sum())
    ltv = aov * 3.2 * 0.38
    show("K7 unit economics", {
        "spend": round(spend, 2), "aov": round(aov, 2),
        "assumed_ltv_margin": round(ltv, 2),
        "ltv_to_cac": round(ltv / (spend / int(g["last"].sum())), 2),
        "note": "LTV uses margin, not revenue; CAC uses last-touch conversions",
    })

    show("K8 saturation", {
        "rule": "marginal CAC rises as spend grows on a fixed audience",
        "test": "hold out a region or week and compare incremental orders",
    })


# ================================================== 15.4 customer analytics
def section_154() -> None:
    cu = load("customers")
    ref = pd.Timestamp("2026-01-01")
    cu["recency"] = (ref - pd.to_datetime(cu["last_order_date"])).dt.days
    cu["tenure"] = (ref - pd.to_datetime(cu["signup_date"])).dt.days // 30
    cu["cohort"] = pd.to_datetime(cu["signup_date"]).dt.strftime("%Y-%m")

    act = load("cohort_activity")
    sizes = act.groupby("cohort")["cohort_size"].first()
    show("C1 cohort sizes", {k: int(v) for k, v in sizes.head(5).items()})

    triangle = act.pivot(index="cohort", columns="months_since_signup",
                         values="active_customers")
    tri_pct = triangle.div(sizes, axis=0).round(3)
    show("C2 cohort triangle (head)", tri_pct.head(4).iloc[:, :6].to_string())
    show("C2 month-12 retention", {
        "mean": round(float(tri_pct[11].mean()), 4),
        "min": round(float(tri_pct[11].min()), 4),
        "max": round(float(tri_pct[11].max()), 4),
    })

    mature = cu[cu["tenure"] >= 12]
    m12 = mature.groupby("segment")["margin"].mean()
    show("C3 12-month margin CLV", {k: round(float(v), 2) for k, v in m12.items()})

    r = pd.qcut(cu["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    f = pd.qcut(cu["orders"].rank(method="first"), 5,
                labels=[1, 2, 3, 4, 5]).astype(int)
    mo = pd.qcut(cu["revenue"].rank(method="first"), 5,
                 labels=[1, 2, 3, 4, 5]).astype(int)
    rfm = pd.DataFrame({"R": r, "F": f, "M": mo})
    rfm["segment"] = np.where(
        (rfm["R"] >= 4) & (rfm["M"] >= 4), "champions",
        np.where(rfm["R"] >= 4, "loyal",
                 np.where(rfm["F"] >= 4, "frequent",
                          np.where(rfm["R"] <= 2, "at_risk", "low_value"))))
    show("C4 RFM segments", rfm["segment"].value_counts().to_dict())
    show("C5 RFM profile", cu.groupby(rfm["segment"]).agg(
        customers=("customer_id", "size"),
        revenue=("revenue", "mean"), recency=("recency", "mean")
    ).round(2).to_string())

    show("C6 NPS", {
        "promoters": int((cu["nps"] >= 9).sum()),
        "detractors": int((cu["nps"] <= 6).sum()),
        "nps": round(float((cu["nps"] >= 9).mean() - (cu["nps"] <= 6).mean()), 4),
        "caution": "a single number hides why customers scored as they did",
    })

    show("C7 retention definition", {
        "definition_A": "bought again within 30 days of first order",
        "definition_B": "active in month 12",
        "warning": "two teams can report different retention from the same table",
    })

    show("C8 identity resolution", {
        "records": len(cu),
        "unique_ids": int(cu["customer_id"].nunique()),
        "note": "one row per customer here; real data needs matching rules",
    })


# ================================================= 15.5 product analytics
def section_155() -> None:
    ev = load("events")
    ev["date"] = pd.to_datetime(ev["event_date"])
    dau = ev[ev["date"] == "2025-09-15"]["user_id"].nunique()
    wau = ev[ev["date"].between("2025-09-08", "2025-09-15")]["user_id"].nunique()
    mau = ev[ev["date"].between("2025-08-16", "2025-09-15")]["user_id"].nunique()
    show("P1 DAU WAU MAU", {"dau": int(dau), "wau": int(wau), "mau": int(mau),
                             "stickiness_dau_mau": round(dau / mau, 4),
                             "stickiness_wau_mau": round(wau / mau, 4)})

    adoption = ev.groupby("event")["user_id"].nunique()
    show("P2 feature adoption", adoption.to_dict())
    show("P3 selection effect", {
        "note": "users who reached purchase see more features by construction",
        "test": "compare adoption among users with equal session counts",
    })

    order = ["login", "view_product", "add_to_cart", "checkout", "purchase"]
    fc = adoption.reindex(order)
    show("P4 product funnel", fc.to_string())
    show("P5 activation", {
        "activation_event": "add_to_cart",
        "reached": int(fc["add_to_cart"]),
        "share_of_login": round(float(fc["add_to_cart"] / fc["login"]), 4),
    })

    show("P6 experimentation", [
        "randomise at the user level, not the event level",
        "define the metric before the test",
        "check novelty effects with a second week",
    ])


# ====================================================== 15.6 operations
def section_156() -> None:
    op = load("operations")
    total = op["avg_minutes"].sum()
    show("O1 cycle time", {
        "steps": len(op), "total_minutes": round(float(total), 2),
        "total_hours": round(float(total) / 60, 2),
        "bottleneck": op.loc[op["avg_minutes"].idxmax(), "step_name"],
    })

    throughput = 120
    wip = throughput * total / 60
    show("O2 little's law", {
        "throughput_per_hour": throughput,
        "cycle_time_hours": round(float(total / 60), 3),
        "expected_wip": round(float(wip), 2),
    })

    show("O3 defect rate", {
        "total_defects_per_1000": int(op["defects_per_1000"].sum()),
        "worst_step": op.loc[op["defects_per_1000"].idxmax(), "step_name"],
    })

    show("O4 cost to serve", pd.DataFrame({
        "step": op["step_name"],
        "cost": op["cost_per_order"],
        "share": (op["cost_per_order"] / op["cost_per_order"].sum()).round(4),
    }).sort_values("cost", ascending=False).head(5).to_string(index=False))

    show("O5 improvement value", {
        "step": "customs",
        "minutes_saved": 15,
        "orders_per_month": 6000,
        "hours_saved": round(15 * 6000 / 60, 2),
        "note": "time saved is not cash; attach a rate before claiming value",
    })

    show("O6 exception review", [
        "steps above 30 minutes",
        "defect rates above 15 per 1000",
        "cost per order above 20",
        "any step whose distribution is bimodal",
    ])


# ================================================== 15.7 finance for analysts
def section_157() -> None:
    f = load("finance")
    rev, cogs = f["revenue"].sum(), f["cogs"].sum()
    show("F1 margins", {
        "revenue": round(float(rev), 2),
        "gross_margin": round(float(rev - cogs), 2),
        "gross_margin_pct": round(float((rev - cogs) / rev), 4),
        "operating_margin_pct": round(float((rev - cogs - f["opex"].sum()) / rev), 4),
    })

    variance = f["revenue"] - f["budget_revenue"]
    show("F2 budget variance", {
        "budget": round(float(f["budget_revenue"].sum()), 2),
        "actual": round(float(rev), 2),
        "variance": round(float(variance.sum()), 2),
        "months_below": int((variance < 0).sum()),
    })

    # volume / price / mix on the monthly table
    b_ord, a_ord = f["budget_orders"].sum(), f["orders"].sum()
    b_aov = f["budget_revenue"].sum() / b_ord
    a_aov = rev / a_ord
    vol = (a_ord - b_ord) * b_aov
    price = (a_aov - b_aov) * b_ord
    cross = (a_ord - b_ord) * (a_aov - b_aov)
    show("F3 variance decomposition", pd.DataFrame([
        ("volume", round(float(vol), 2)),
        ("price", round(float(price), 2)),
        ("interaction", round(float(cross), 2)),
        ("total", round(float(variance.sum()), 2)),
    ], columns=["component", "amount"]).to_string(index=False))

    show("F4 unit economics", {
        "orders": int(a_ord),
        "margin_per_order": round(float((rev - cogs) / a_ord), 2),
        "opex_per_order": round(float(f["opex"].sum() / a_ord), 2),
        "contribution_per_order": round(float((rev - cogs - f["opex"].sum()) / a_ord), 2),
    })

    m = load("monthly")
    monthly_rev = m.groupby("month")["revenue"].sum()
    show("F5 cash seasonality", {
        "peak_month": monthly_rev.idxmax(), "peak": round(float(monthly_rev.max()), 2),
        "trough_month": monthly_rev.idxmin(), "trough": round(float(monthly_rev.min()), 2),
    })

    show("F6 accrual vs cash", [
        "revenue is recognised when earned, not when paid",
        "a cash bridge needs receivables and payables movements",
        "involve a controller before publishing a cash figure",
    ])


# ============================================ 15.8 analysis to recommendation
def section_158() -> None:
    show("B1 value of a fix", {
        "issue": "customs step adds 35 minutes per order",
        "minutes_saved": 15,
        "orders_per_month": 6000,
        "hours_saved": 1500.0,
        "loaded_rate": 45.0,
        "monthly_value": 67500.0,
        "assumption": "time saved converts to capacity or cost",
    })

    show("B2 business case fields", [
        "problem with a number", "proposed change", "owner",
        "cost", "expected effect", "measurement plan", "risk",
    ])

    show("B3 counterfactual", {
        "question": "what happens if we do nothing?",
        "rule": "compare against the baseline, not against the target",
    })

    show("B4 measurement plan", [
        "metric: average cycle time at the customs step",
        "window: four weeks before and after",
        "guardrail: defect rate at quality check",
    ])

    show("B5 one-page memo", {
        "headline": "Reducing customs handling by 15 minutes is worth $67.5k a month",
        "evidence": "35 minutes average, 15 minutes saved, 6,000 orders, $45 loaded rate",
        "ask": "fund a pre-clearance pilot for one region",
    })

    show("B6 limitations", [
        "time saved may not become cash without headcount action",
        "one region may not generalise",
        "the estimate assumes volume stays constant",
    ])


# ================================================ 15.9 example bank
def section_159() -> None:
    m = load("monthly")
    cu = load("customers")
    show("E1 months", int(m["month"].nunique()))
    show("E2 channels", m["channel"].value_counts().to_dict())
    show("E3 revenue", round(float(m["revenue"].sum()), 2))
    show("E4 orders", int(m["orders"].sum()))
    show("E5 conversion", round(float(m["orders"].sum() / m["sessions"].sum()), 4))
    show("E6 margin pct", round(float(
        (m["revenue"].sum() - m["cogs"].sum()) / m["revenue"].sum()), 4))
    show("E7 customers", len(cu))
    show("E8 revenue per customer", round(float(cu["revenue"].mean()), 2))


# =================================================== 15.10 capstone
def section_1510() -> None:
    m = load("monthly")
    cu = load("customers")
    rev = float(m["revenue"].sum())
    orders = int(m["orders"].sum())
    show("K1 scorecard", {
        "revenue": round(rev, 2),
        "orders": orders,
        "conversion": round(float(orders / m["sessions"].sum()), 4),
        "aov": round(float(rev / orders), 2),
        "gross_margin_pct": round(float((rev - m["cogs"].sum()) / rev), 4),
        "customers": len(cu),
    })

    q4 = m[m["month"].between("2024-10", "2024-12")]
    show("K2 quarterly change", {
        "revenue": round(float(q4["revenue"].sum()), 2),
        "orders": int(q4["orders"].sum()),
        "aov": round(float(q4["revenue"].sum() / q4["orders"].sum()), 2),
    })

    show("K3 control totals", {
        "monthly_rows": len(m), "customer_rows": len(cu),
        "event_rows": len(load("events")), "product_rows": len(load("products")),
    })

    show("K4 artifacts", ["metrics_catalog.csv", "revenue_bridge.csv",
                           "cohort_clv.csv", "business_case.md"])


def section_extra() -> None:
    """Additional business queries for the expanded Chapter 15 sections."""
    m = load("monthly")
    c = load("campaigns")
    cu = load("customers")
    ev = load("events")
    p = load("products")
    op = load("operations")
    f = load("finance")

    agg = m.groupby("month", as_index=False).agg(
        revenue=("revenue", "sum"), orders=("orders", "sum"),
        sessions=("sessions", "sum"), cogs=("cogs", "sum"))
    agg["mom"] = agg["revenue"].pct_change()
    agg["yoy"] = agg["revenue"].pct_change(12)
    agg["ytd"] = agg.groupby(agg["month"].str[:4])["revenue"].cumsum()
    show("X1 monthly scorecard", agg.head(5).round(4).to_string(index=False))

    q = m.assign(q=m["month"].str[:4] + "-Q" + (
        (m["month"].str[5:7].astype(int) - 1) // 3 + 1).astype(str))
    show("X2 quarterly revenue", q.groupby("q")["revenue"].sum().round(2).to_dict())

    cha = m.groupby("channel").agg(
        revenue=("revenue", "sum"), margin=("margin", "sum"))
    cha["margin_pct"] = (cha["margin"] / cha["revenue"]).round(4)
    show("X3 channel margin", cha.round(2).to_string())

    reg = m.groupby("region").agg(
        revenue=("revenue", "sum"), returns=("returns", "sum"))
    reg["return_pct"] = (reg["returns"] / reg["revenue"]).round(4)
    show("X4 return rate by region", reg.round(4).to_string())

    cm = c.groupby("month", as_index=False).agg(
        spend=("spend", "sum"), conversions=("last_touch", "sum"))
    cm["cpa"] = (cm["spend"] / cm["conversions"]).round(2)
    show("X5 monthly campaigns", cm.head(4).round(2).to_string(index=False))

    show("X6 spend share", (c.groupby("platform")["spend"].sum()
                            / c["spend"].sum()).round(4).to_dict())

    clk = c.groupby("platform").agg(clicks=("clicks", "sum"),
                                     conv=("platform_conversions", "sum"))
    show("X7 click to conversion", (clk["conv"] / clk["clicks"]).round(4).to_dict())

    ev2 = ev.copy()
    ev2["platform"] = ev2["platform"].astype(str)
    fun = ev2.pivot_table(index="event", columns="platform",
                          values="user_id", aggfunc="nunique", fill_value=0)
    show("X8 funnel by platform", fun.to_string())

    coh = load("cohort_activity")
    coh["retained"] = coh["active_customers"] / coh["cohort_size"]
    seg_map = cu.set_index("customer_id")["segment"]
    show("X9 cohort retention spread", {
        "m1_mean": round(float(coh[coh["months_since_signup"] == 1]["retained"].mean()), 4),
        "m6_mean": round(float(coh[coh["months_since_signup"] == 6]["retained"].mean()), 4),
        "m11_mean": round(float(coh[coh["months_since_signup"] == 11]["retained"].mean()), 4),
    })

    show("X10 revenue concentration", {
        "top_decile_share": round(float(
            cu["revenue"].nlargest(400).sum() / cu["revenue"].sum()), 4),
        "top_half_share": round(float(
            cu["revenue"].nlargest(2000).sum() / cu["revenue"].sum()), 4),
    })

    show("X11 promoter share by segment", {
        k: round(float(v), 4) for k, v in cu.assign(
            prom=cu["nps"] >= 9).groupby("segment")["prom"].mean().items()
    })

    show("X12 events by platform", ev2["platform"].value_counts().to_dict())
    show("X13 DAU by platform", {
        k: int(v) for k, v in ev2[ev2["event_date"] == "2025-09-15"][
            "platform"].value_counts().items()
    })

    show("X14 cost ranking", op.sort_values("cost_per_order", ascending=False)[
        ["step_name", "cost_per_order"]].head(4).to_string(index=False))

    show("X15 defect ranking", op.sort_values("defects_per_1000", ascending=False)[
        ["step_name", "defects_per_1000"]].head(4).to_string(index=False))

    show("X16 throughput scenarios", {
        t: round(float(t * op["avg_minutes"].sum() / 60), 2)
        for t in [60, 120, 180]
    })

    f["q"] = f["month"].str[:4] + "-Q" + (
        (f["month"].str[5:7].astype(int) - 1) // 3 + 1).astype(str)
    show("X17 quarterly margins", f.groupby("q").agg(
        revenue=("revenue", "sum"), margin=("margin", "sum")).round(2).to_string())

    f["var"] = f["revenue"] - f["budget_revenue"]
    show("X18 variance by month", f[["month", "var"]].head(6).round(2).to_string(index=False))

    chan = m.groupby("channel").agg(revenue=("revenue", "sum"),
                                     margin=("margin", "sum"))
    show("X19 contribution by channel", {
        k: round(float(v), 2) for k, v in chan["margin"].items()
    })

    show("X20 final dashboard", {
        "revenue": round(float(m["revenue"].sum()), 2),
        "orders": int(m["orders"].sum()),
        "customers": len(cu),
        "campaign_spend": round(float(c["spend"].sum()), 2),
        "gross_margin_pct": round(float(
            (m["revenue"].sum() - m["cogs"].sum()) / m["revenue"].sum()), 4),
    })


def section_extra2() -> None:
    """A second batch of verified business queries for Chapter 15."""
    m = load("monthly")
    c = load("campaigns")
    cu = load("customers")
    ev = load("events")
    p = load("products")
    op = load("operations")
    f = load("finance")

    # sales detail
    top = p.sort_values("revenue_q4", ascending=False).head(5)
    show("Y1 top products", top[["product_id", "category", "revenue_q4"]]
         .round(2).to_string(index=False))

    cat = p.groupby("category").agg(q3=("units_q3", "sum"),
                                    q4=("units_q4", "sum"))
    cat["growth"] = ((cat["q4"] / cat["q3"] - 1)).round(4)
    show("Y2 category units", cat.to_string())

    p["margin_rate"] = (p["price"] - p["unit_cost"]) / p["price"]
    show("Y3 product margin range", {
        "min": round(float(p["margin_rate"].min()), 4),
        "max": round(float(p["margin_rate"].max()), 4),
        "mean": round(float(p["margin_rate"].mean()), 4),
    })

    share = m.groupby("region")["revenue"].sum()
    show("Y4 regional share", {
        k: round(float(v / share.sum()), 4) for k, v in share.items()
    })

    pivot = m.pivot_table(index="month", columns="channel",
                          values="revenue", aggfunc="sum")
    show("Y5 channel monthly", pivot.tail(4).round(2).to_string())

    reg = m.groupby("region").agg(
        revenue=("revenue", "sum"), orders=("orders", "sum"))
    show("Y6 orders per session by region", {
        k: round(float(v), 4) for k, v in (
            m.groupby("region")["orders"].sum() / m.groupby("region")["sessions"].sum()
        ).items()
    })

    bych = m.groupby("channel").agg(
        revenue=("revenue", "sum"), returns=("returns", "sum"))
    bych["return_pct"] = (bych["returns"] / bych["revenue"]).round(4)
    show("Y7 returns by channel", bych.round(4).to_string())

    # marketing detail
    cpc = c.groupby("month", as_index=False).agg(
        spend=("spend", "sum"), clicks=("clicks", "sum"))
    cpc["cpc"] = (cpc["spend"] / cpc["clicks"]).round(2)
    show("Y8 monthly cpc", cpc.head(4).round(2).to_string(index=False))

    g = c.groupby("platform").agg(spend=("spend", "sum"),
                                   last=("last_touch", "sum"),
                                   first=("first_touch", "sum"))
    aov_val = float(m["revenue"].sum() / m["orders"].sum())
    show("Y9 roas by platform", {
        k: round(float(g.loc[k, "last"] * aov_val / g.loc[k, "spend"]), 3)
        for k in g.index
    })
    show("Y10 cac by platform", {
        k: round(float(g.loc[k, "spend"] / g.loc[k, "last"]), 2) for k in g.index
    })
    show("Y11 first to last ratio", {
        k: round(float(g.loc[k, "first"] / g.loc[k, "last"]), 3) for k in g.index
    })
    show("Y12 platform conversion share", {
        k: round(float(v / g["last"].sum()), 4) for k, v in g["last"].items()
    })

    ev2 = ev.copy()
    ev2["platform"] = ev2["platform"].astype(str)
    adoption = ev2.groupby("event")["user_id"].nunique()
    order = ["login", "view_product", "add_to_cart", "checkout", "purchase"]
    fc = adoption.reindex(order)
    show("Y13 funnel absolute drop", {
        a + "_to_" + b: int(fc[a] - fc[b])
        for a, b in zip(order[:-1], order[1:])
    })

    # customer detail
    show("Y14 segment revenue share", {
        k: round(float(v / cu["revenue"].sum()), 4)
        for k, v in cu.groupby("segment")["revenue"].sum().items()
    })
    show("Y15 order frequency", {
        k: int(v) for k, v in cu["orders"].value_counts().head(6).items()
    })
    show("Y16 clv by region", {
        k: round(float(v), 2) for k, v in cu.groupby("region")["margin"].mean().items()
    })
    ref = pd.Timestamp("2026-01-01")
    cu["recency"] = (ref - pd.to_datetime(cu["last_order_date"])).dt.days
    buckets = pd.cut(cu["recency"], [-1, 90, 180, 365, 10000],
                     labels=["0-90", "91-180", "181-365", "365+"])
    show("Y17 recency buckets", {
        k: int(v) for k, v in buckets.value_counts().sort_index().items()
    })
    show("Y18 top customers", cu.nlargest(3, "revenue")[
        ["customer_id", "segment", "orders", "revenue"]].to_string(index=False))
    show("Y19 nps distribution", {
        k: int(v) for k, v in cu["nps"].value_counts().head(5).items()
    })

    # product detail
    show("Y20 events by type", {
        k: int(v) for k, v in ev2["event"].value_counts().items()
    })
    ev2["date"] = pd.to_datetime(ev2["event_date"])
    ev2["week"] = ev2["date"].dt.to_period("W").astype(str)
    weekly = ev2.groupby("week")["user_id"].nunique()
    show("Y21 weekly active", weekly.head(4).to_dict())
    show("Y22 most active users", {
        k: int(v) for k, v in ev2["user_id"].value_counts().head(4).items()
    })
    show("Y23 filter users by platform", {
        k: int(v) for k, v in ev2[ev2["event"] == "use_filter"][
            "platform"].value_counts().items()
    })

    # operations detail
    show("Y24 step durations", op.sort_values("avg_minutes", ascending=False)[
        ["step_name", "avg_minutes"]].head(4).to_string(index=False))
    serve = op.sort_values("cost_per_order", ascending=False)
    cum = (serve["cost_per_order"].cumsum()
           / serve["cost_per_order"].sum()).round(4)
    show("Y25 cumulative cost share", {
        serve.iloc[i]["step_name"]: float(cum.iloc[i]) for i in [0, 1, 2, 3]
    })
    work = op[op["avg_minutes"] < 60]["avg_minutes"].sum()
    wait = op[op["avg_minutes"] >= 60]["avg_minutes"].sum()
    show("Y26 work versus wait", {
        "work_minutes": round(float(work), 2),
        "wait_minutes": round(float(wait), 2),
        "wait_share": round(float(wait / (work + wait)), 4),
    })

    # finance detail
    f["opex_share"] = (f["opex"] / f["revenue"]).round(4)
    show("Y27 opex share", {
        "min": round(float(f["opex_share"].min()), 4),
        "max": round(float(f["opex_share"].max()), 4),
    })
    f["q"] = f["month"].str[:4] + "-Q" + (
        (f["month"].str[5:7].astype(int) - 1) // 3 + 1).astype(str)
    f["per_order"] = ((f["revenue"] - f["cogs"]) / f["orders"]).round(2)
    show("Y28 margin per order by quarter", {
        k: round(float(v), 2) for k, v in f.groupby("q")["per_order"].mean().items()
    })
    show("Y29 budget versus actual", f[["month", "budget_revenue", "revenue"]]
         .head(3).round(2).to_string(index=False))
    show("Y30 annual totals", {
        "revenue": round(float(f["revenue"].sum()), 2),
        "cogs": round(float(f["cogs"].sum()), 2),
        "opex": round(float(f["opex"].sum()), 2),
        "margin": round(float(f["margin"].sum()), 2),
    })


def section_extra3() -> None:
    """A third batch of verified business queries for Chapter 15."""
    m = load("monthly")
    c = load("campaigns")
    cu = load("customers")
    ev = load("events")
    op = load("operations")
    f = load("finance")

    spec = pd.DataFrame([{
        "name": "conversion rate", "formula": "orders / sessions",
        "numerator": "orders", "denominator": "sessions", "unit": "share",
        "grain": "session", "window": "trailing 30 days",
        "filters": "no internal traffic", "source": "warehouse.orders",
        "owner": "growth analytics",
    }])
    show("Z1 metric specification", spec.to_string(index=False))

    show("Z2 counts and rates", {
        "orders": int(m["orders"].sum()),
        "sessions": int(m["sessions"].sum()),
        "revenue": round(float(m["revenue"].sum()), 2),
        "orders_per_session": round(float(
            m["orders"].sum() / m["sessions"].sum()), 4),
    })

    q3 = m[m["month"].between("2024-07", "2024-09")]
    q4 = m[m["month"].between("2024-10", "2024-12")]
    show("Z3 quarterly orders", {
        "Q3": int(q3["orders"].sum()), "Q4": int(q4["orders"].sum()),
        "change_pct": round(float(
            q4["orders"].sum() / q3["orders"].sum() - 1), 4),
    })

    revm = m.groupby("month")["revenue"].sum()
    show("Z4 monthly volatility", {
        "mean": round(float(revm.mean()), 2),
        "std": round(float(revm.std()), 2),
        "cv": round(float(revm.std() / revm.mean()), 4),
    })
    show("Z5 best and worst month", {
        "best": revm.idxmax(), "best_value": round(float(revm.max()), 2),
        "worst": revm.idxmin(), "worst_value": round(float(revm.min()), 2),
    })

    yr = m.assign(year=m["month"].str[:4]).groupby(["year", "channel"])[
        "revenue"].sum().unstack()
    show("Z6 channel by year", yr.round(2).to_string())

    cm = m.groupby("month", as_index=False).agg(rev=("revenue", "sum"))
    cm["cum"] = cm["rev"].cumsum().round(2)
    show("Z7 cumulative revenue", cm[["month", "cum"]].head(4)
         .to_string(index=False))

    campm = c.groupby("month", as_index=False).agg(
        spend=("spend", "sum"), last=("last_touch", "sum"))
    aov_val = float(m["revenue"].sum() / m["orders"].sum())
    campm["roas"] = (campm["last"] * aov_val / campm["spend"]).round(3)
    show("Z8 monthly roas", campm.head(4).to_string(index=False))

    show("Z9 monthly conversions", {
        k: int(v) for k, v in c.groupby("month")["last_touch"].sum().head(4).items()
    })

    g = c.groupby("platform").agg(spend=("spend", "sum"),
                                   last=("last_touch", "sum"))
    g["cpa"] = g["spend"] / g["last"]
    show("Z10 efficiency rank", {
        k: round(float(v), 2) for k, v in g["cpa"].sort_values().items()
    })

    ref = pd.Timestamp("2026-01-01")
    cu["tenure"] = (ref - pd.to_datetime(cu["signup_date"])).dt.days // 30
    tb = pd.cut(cu["tenure"], [-1, 12, 24, 36, 1000],
                labels=["0-12", "13-24", "25-36", "36+"])
    show("Z11 tenure buckets", {
        k: int(v) for k, v in tb.value_counts().sort_index().items()
    })
    show("Z12 tickets by segment", {
        k: round(float(v), 3) for k, v in cu.groupby("segment")[
            "support_tickets"].mean().items()
    })
    top = cu.nlargest(400, "revenue")
    show("Z13 margin concentration", {
        "top_decile_margin_share": round(float(
            top["margin"].sum() / cu["margin"].sum()), 4)
    })

    r = pd.qcut(cu["recency"] if "recency" in cu else (ref - pd.to_datetime(
        cu["last_order_date"])).dt.days, 5, labels=[5, 4, 3, 2, 1]).astype(int)
    fr = pd.qcut(cu["orders"].rank(method="first"), 5,
                 labels=[1, 2, 3, 4, 5]).astype(int)
    mo = pd.qcut(cu["revenue"].rank(method="first"), 5,
                 labels=[1, 2, 3, 4, 5]).astype(int)
    rfm = pd.DataFrame({"R": r, "F": fr, "M": mo})
    show("Z14 rfm mean scores", {
        k: round(float(v), 2) for k, v in rfm.mean().items()
    })

    sy = pd.to_datetime(cu["signup_date"]).dt.year
    show("Z15 signup years", {k: int(v) for k, v in sy.value_counts().items()})

    epm = ev.groupby("user_id").size()
    eb = pd.cut(epm, [0, 3, 6, 10, 1000],
                labels=["1-3", "4-6", "7-10", "10+"])
    show("Z16 events per user", {
        k: int(v) for k, v in eb.value_counts().sort_index().items()
    })

    ev2 = ev.copy()
    ev2["platform"] = ev2["platform"].astype(str)
    show("Z17 platform share by event", {
        k: round(float(v), 4) for k, v in (
            ev2["platform"].value_counts() / len(ev2)).items()
    })

    ev2["date"] = pd.to_datetime(ev2["event_date"])
    ev2["dow"] = ev2["date"].dt.dayofweek
    show("Z18 weekend events", {
        "weekday": int((ev2["dow"] < 5).sum()),
        "weekend": int((ev2["dow"] >= 5).sum()),
    })

    op2 = op.copy()
    op2["risk"] = op2["defects_per_1000"] * op2["cost_per_order"]
    show("Z19 defect cost proxy", op2.sort_values("risk", ascending=False)[
        ["step_name", "risk"]].round(2).head(4).to_string(index=False))

    ds = op.sort_values("avg_minutes", ascending=False)
    show("Z20 top three duration share", round(float(
        ds["avg_minutes"].head(3).sum() / ds["avg_minutes"].sum()), 4))

    f2 = f.copy().sort_values("month")
    f2["cum_rev"] = f2["revenue"].cumsum().round(2)
    f2["cum_bud"] = f2["budget_revenue"].cumsum().round(2)
    show("Z21 cumulative budget", f2[["month", "cum_rev", "cum_bud"]]
         .head(3).to_string(index=False))

    f2["year"] = f2["month"].str[:4]
    f2["gm"] = (f2["revenue"] - f2["cogs"]) / f2["revenue"]
    show("Z22 gross margin by year", {
        k: round(float(v), 4) for k, v in f2.groupby("year")["gm"].mean().items()
    })

    show("Z23 conversion by channel", {
        k: round(float(v), 4) for k, v in (
            m.groupby("channel")["orders"].sum()
            / m.groupby("channel")["sessions"].sum()).items()
    })

    show("Z24 final control totals", {
        "monthly_rows": len(m), "campaign_rows": len(c),
        "customer_rows": len(cu), "event_rows": len(ev),
        "operation_rows": len(op), "finance_rows": len(f),
    })


def main() -> None:
    print("python", "pandas", pd.__version__)
    print()
    for section in (
        section_151, section_152, section_153, section_154, section_155,
        section_156, section_157, section_158, section_159, section_1510,
        section_extra, section_extra2, section_extra3,
    ):
        print("=" * 78)
        section()


if __name__ == "__main__":
    main()
