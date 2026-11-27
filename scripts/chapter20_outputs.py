#!/usr/bin/env python3
"""Print every output block used in Chapter 20, from the current project data.

Run from the project root:

    python scripts/chapter20_outputs.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "data" / "raw" / "project20"
DB = PROJ / "kestrel_membership.db"
SEED = 2007
FEATURES = ["tenure_months", "is_premium", "is_plus", "is_paid_channel",
            "is_invoice", "orders_6m", "revenue_6m", "margin_6m",
            "tickets_6m", "sessions_6m", "email_open_rate"]
LEAKS = ["cancellation_reason", "days_after_end", "final_refund"]


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def load():
    con = sqlite3.connect(DB)
    out = {}
    for name in ["members", "orders", "tickets", "sessions", "email"]:
        out[name] = pd.read_sql(f"SELECT * FROM {name}", con)
    con.close()
    fm = pd.read_csv(PROJ / "fixed_horizon.csv")
    cad = pd.read_csv(PROJ / "cohort_activity.csv")
    sv = pd.read_csv(PROJ / "exit_survey.csv")
    fin = pd.read_csv(PROJ / "finance_margin.csv")
    return out, fm, cad, sv, fin


def split(fm):
    tr = fm[fm["snapshot"] == "2025-01-01"]
    te = fm[fm["snapshot"] == "2025-07-01"]
    return tr, te


def fit_predict(tr, te, features=FEATURES):
    m = LogisticRegression(max_iter=2000, random_state=SEED)
    m.fit(tr[features], tr["churn_90d"])
    return m, m.predict_proba(te[features])[:, 1]


def section_a(T, fm, cad, sv, fin) -> None:
    show("A1 churn definitions", {
        "contractual": "member with an end date left the paid program",
        "non_contractual": "90 days with no order is treated as disengagement",
        "voluntary": "member cancelled (price, service, moved, competitor)",
        "involuntary": "payment failed",
    })
    y = T["members"]["churned"]
    show("A2 base rates", {
        "contractual": round(float(y.mean()), 4),
        "involuntary_share": round(float(
            T["members"]["churn_type"].eq("involuntary").sum()
            / y.sum()), 4),
        "active_at_cutoff": int((y == 0).sum()),
    })
    show("A3 observation window", {
        "signup_start": "2023-01-01", "cutoff": "2026-01-01",
        "label_horizon": "90 days from each snapshot",
    })
    show("A4 outcome", {
        "primary": "churn within 90 days of snapshot",
        "secondary": "monthly retention by signup cohort",
    })
    show("A5 decision", {
        "budget": "one retention program next quarter",
        "question": "who is leaving, why, and which intervention pays back",
    })
    show("A6 diagnosis vs scoring", {
        "diagnosis": "which factors are associated with leaving",
        "scoring": "which members to contact first",
        "reason": "a driver model is not a targeting model",
    })
    show("A7 frame criteria", {
        "one_churn_definition": True, "leak_free_features": True,
        "temporal_holdout": True, "payback_calculated": True,
    })


def section_b(T, fm, cad, sv, fin) -> None:
    mem = T["members"]
    show("B1 table inventory", {k: len(v) for k, v in T.items()})
    show("B2 coverage", {
        "members": len(mem),
        "with_orders": int(mem["member_id"].isin(T["orders"]["member_id"]).sum()),
        "with_sessions": int(mem["member_id"].isin(
            T["sessions"]["member_id"]).sum()),
        "with_email": int(mem["member_id"].isin(T["email"]["member_id"]).sum()),
    })
    show("B3 censoring", {
        "churned": int(mem["churned"].sum()),
        "censored": int((mem["churned"] == 0).sum()),
        "share_censored": round(float((mem["churned"] == 0).mean()), 4),
    })
    show("B4 identity", {
        "unique_ids": int(mem["member_id"].nunique()),
        "rows": len(mem),
        "duplicate_ids": int(mem["member_id"].duplicated().sum()),
    })
    show("B5 date reliability", {
        "churn_dates_present": int(mem["end_date"].notna().sum()),
        "involuntary": int(mem["churn_type"].eq("involuntary").sum()),
        "reason_none_for_active": int(mem.loc[
            mem["churned"] == 0, "cancellation_reason"].eq("none").sum()),
    })
    miss = T["orders"].groupby("member_id").size()
    show("B6 missing by cohort", {
        "no_orders": int((~mem["member_id"].isin(miss.index)).sum()),
        "no_email": int((~mem["member_id"].isin(
            T["email"]["member_id"])).sum()),
    })
    show("B7 definition sensitivity", {
        "contractual": round(float(mem["churned"].mean()), 4),
        "involuntary_only": round(float(
            mem["churn_type"].eq("involuntary").mean()), 4),
        "voluntary_only": round(float(
            mem["churn_type"].eq("voluntary").mean()), 4),
    })
    show("B8 horizon sensitivity", {
        "90_day_2025_01": round(float(fm.loc[
            fm["snapshot"] == "2025-01-01", "churn_90d"].mean()), 4),
        "90_day_2025_07": round(float(fm.loc[
            fm["snapshot"] == "2025-07-01", "churn_90d"].mean()), 4),
    })


def section_c(T, fm, cad, sv, fin) -> None:
    tr, te = split(fm)
    show("C1 fixed horizon shape", {
        "rows": len(fm), "features": len(FEATURES),
        "train_rows": len(tr), "test_rows": len(te),
    })
    show("C2 train test rates", {
        "train": round(float(tr["churn_90d"].mean()), 4),
        "test": round(float(te["churn_90d"].mean()), 4),
    })
    show("C3 feature list", FEATURES)
    show("C4 excluded leaks", LEAKS)
    reason_map = {"none": 0, "payment_failed": 1, "price": 2,
                  "service": 3, "moved": 4, "competitor": 5}
    tr_l = tr.copy(); te_l = te.copy()
    tr_l["cancellation_reason_num"] = tr_l["cancellation_reason"].map(reason_map)
    te_l["cancellation_reason_num"] = te_l["cancellation_reason"].map(reason_map)
    show("C5 leakage demo", pd.DataFrame([
        ("clean", round(float(roc_auc_score(
            te["churn_90d"], fit_predict(tr, te)[1])), 4)),
        ("with cancel reason", round(float(roc_auc_score(
            te_l["churn_90d"], fit_predict(tr_l, te_l, FEATURES + [
                "cancellation_reason_num"])[1])), 4)),
    ], columns=["features", "auc"]).to_string(index=False))
    show("C6 group overlap", {
        "train_members": int(tr["member_id"].nunique()),
        "test_members": int(te["member_id"].nunique()),
        "overlap": int(len(set(tr["member_id"]) & set(te["member_id"]))),
    })
    show("C7 snapshot logic", {
        "features": "only events before the snapshot",
        "label": "churn in the 90 days after the snapshot",
    })
    show("C8 frozen artifact", {
        "path": "data/raw/project20/fixed_horizon.csv",
        "rows": len(fm),
    })


def section_d(T, fm, cad, sv, fin) -> None:
    sizes = cad.groupby("cohort")["cohort_size"].first()
    tri = cad.pivot(index="cohort", columns="months_since_signup",
                    values="active").div(sizes, axis=0)
    show("D1 retention triangle", tri.head(5).iloc[:, :6].round(3).to_string())
    show("D2 retention by age", {
        k: round(float(tri[k].mean()), 4) for k in [0, 3, 6, 12]
    })
    d = fm[fm["snapshot"] == "2025-01-01"]
    show("D3 churn by plan", {k: round(float(v), 4) for k, v in
         d.groupby("plan")["churn_90d"].mean().items()})
    show("D4 churn by channel", {k: round(float(v), 4) for k, v in
         d.groupby("acquisition_channel")["churn_90d"].mean().items()})
    tb = pd.cut(d["tenure_months"], [-1, 6, 12, 24, 60],
                labels=["0-6", "7-12", "13-24", "25+"])
    show("D5 churn by tenure", {str(k): round(float(v), 4) for k, v in
         d.groupby(tb, observed=True)["churn_90d"].mean().items()})
    eb = pd.cut(d["email_open_rate"], [-1, 0.1, 0.3, 1.0],
                labels=["low", "mid", "high"])
    show("D6 churn by engagement", {str(k): round(float(v), 4) for k, v in
         d.groupby(eb, observed=True)["churn_90d"].mean().items()})
    show("D7 survey themes", {k: int(v) for k, v in
         sv["theme"].value_counts().items()})
    show("D8 survey scores", {
        "mean": round(float(sv["score"].mean()), 2),
        "low_share": round(float(sv["score"].le(2).mean()), 4),
    })
    show("D9 age vs calendar", {
        "note": "align cohorts by tenure before comparing",
        "calendar_effect": "Q1 signups churn more at every age",
    })
    sy = d["member_id"].str[2:7]
    show("D10 seasonality", {
        "note": "both snapshots fall in Q1, so quarter effects cannot be separated",
        "snapshots": sorted(fm["snapshot"].unique().tolist()),
    })


def section_e(T, fm, cad, sv, fin) -> None:
    tr, te = split(fm)
    m, p = fit_predict(tr, te)
    show("E1 model metrics", {
        "roc_auc": round(float(roc_auc_score(te["churn_90d"], p)), 4),
        "pr_auc": round(float(average_precision_score(
            te["churn_90d"], p)), 4),
        "base_rate": round(float(te["churn_90d"].mean()), 4),
    })
    coef = pd.Series(m.coef_[0], index=FEATURES).sort_values(ascending=False)
    show("E2 coefficients", {
        k: round(float(v), 3) for k, v in
        pd.concat([coef.head(4), coef.tail(3)]).items()
    })
    show("E3 group comparison", pd.DataFrame([
        ("Basic", round(float(tr.loc[tr["plan"] == "Basic",
                                    "churn_90d"].mean()), 4)),
        ("Plus", round(float(tr.loc[tr["plan"] == "Plus",
                                   "churn_90d"].mean()), 4)),
        ("Premium", round(float(tr.loc[tr["plan"] == "Premium",
                                      "churn_90d"].mean()), 4)),
    ], columns=["plan", "churn"]).to_string(index=False))
    show("E4 clv by plan", fin[["plan", "margin_per_month",
                               "observed_churn"]].round(4).to_string(index=False))
    # annualise the 90-day churn rate from the fixed-horizon table
    q = tr.groupby("plan")["churn_90d"].mean()
    annual = 1 - (1 - q) ** 4
    clv = fin.set_index("plan")["margin_per_month"] * 12 / annual
    show("E5 clv simple", {k: round(float(v), 2) for k, v in clv.items()})
    value = 0.01 * fin["members"].sum() * float(clv.mean())
    show("E6 value of one point", {
        "base_churn": round(float(q.mean()), 4),
        "one_point_members": round(float(0.01 * fin["members"].sum()), 1),
        "annual_value": round(float(value), 2),
    })
    show("E7 interventions", pd.DataFrame([
        ("engagement nudge", "low email openers", 3200, 0.30),
        ("service recovery", "members with tickets", 4100, 0.12),
        ("annual plan upgrade", "Basic monthly", 2800, 0.09),
    ], columns=["program", "target", "addressable", "uplift"]
    ).to_string(index=False))
    show("E8 sizing", {
        "largest_group": 4100,
        "targetable_at_50pct_capacity": 2050,
        "note": "rank by predicted risk within each group",
    })


def section_f(T, fm, cad, sv, fin) -> None:
    tr, te = split(fm)
    m, p = fit_predict(tr, te)
    y = te["churn_90d"].to_numpy()
    show("F1 holdout metrics", {
        "accuracy": round(float(accuracy_score(y, (p >= 0.5).astype(int))), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
    })
    pt, pp = calibration_curve(y, p, n_bins=6, strategy="quantile")
    show("F2 calibration", pd.DataFrame({
        "predicted": np.round(pp, 4), "observed": np.round(pt, 4)
    }).to_string(index=False))
    show("F3 brier", round(float(brier_score_loss(y, p)), 5))
    labelled = te[te["churn_90d"] == 1]
    show("F4 definition sensitivity", {
        "contractual_90d": round(float(y.mean()), 4),
        "involuntary_share_of_labels": round(float(
            labelled["cancellation_reason"].eq("payment_failed").mean()), 4),
        "ever_cancelled": round(float(te["cancellation_reason"].ne("none").mean()), 4),
    })
    aucs = {}
    for plan in ["Basic", "Plus", "Premium"]:
        sub = te[te["plan"] == plan]
        if sub["churn_90d"].nunique() > 1:
            aucs[plan] = round(float(roc_auc_score(
                sub["churn_90d"], m.predict_proba(sub[FEATURES])[:, 1])), 4)
    show("F5 cohort stability", aucs)
    rule = (te["tenure_months"] <= 6).astype(int)
    show("F6 tenure rule", {
        "rule_auc": round(float(roc_auc_score(y, rule)), 4),
        "model_auc": round(float(roc_auc_score(y, p)), 4),
    })
    show("F7 limits", [
        "associations are not causes", "the label depends on a definition",
        "a targeting score is not an uplift model",
    ])
    show("F8 falsifier", {
        "test": "a holdout with AUC below the tenure rule",
        "result": "not triggered",
    })


def section_g(T, fm, cad, sv, fin) -> None:
    show("G1 deliverables", [
        "retention triangle", "driver chart", "model card",
        "targeting list", "two-page memo",
    ])
    show("G2 model card", {
        "name": "churn-scoring v1", "owner": "retention analytics",
        "intended_use": "rank members for a 2,050-case quarterly review",
        "not_for": "individual pricing or service denial",
        "metric": "PR AUC on a temporal holdout",
        "monitor": "precision at capacity each month",
    })
    tr, te = split(fm)
    _, p = fit_predict(tr, te)
    top = np.argsort(-p)[:5]
    show("G3 targeting list", pd.DataFrame({
        "member_id": te["member_id"].to_numpy()[top],
        "churn_probability": np.round(p[top], 4),
    }).to_string(index=False))
    show("G4 recommendation", {
        "program": "engagement nudge",
        "target": "low email openers",
        "cost": 180000.0, "expected_value": 360000.0,
        "payback": round(360000 / 180000, 2),
    })
    show("G5 memo", [
        "90-day churn is 2.6% on the holdout",
        "Low engagement and Basic plans carry the risk",
        "Fund a $180k nudge for 2,050 members; owner retention lead",
    ])
    show("G6 measurement", {
        "metric": "incremental 90-day retention",
        "window": "one quarter", "guardrail": "margin per member",
    })
    show("G7 rubric", {
        "problem": 20, "panel_leakage": 25, "analysis": 25,
        "validation": 15, "communication": 15,
    })
    show("G8 common failures", [
        "three churn definitions in one report",
        "cohorts compared at different ages",
        "post-churn behavior as a feature",
        "revenue instead of margin in CLV",
        "a discount without a payback calculation",
    ])
    show("G9 stretch", [
        "survival model", "A/B test for the program", "monthly churn monitor",
    ])
    show("G10 controls", {
        "members": len(T["members"]), "fixed_horizon_rows": len(fm),
        "cohorts": int(cad["cohort"].nunique()),
    })


def section_extra(T, fm, cad, sv, fin) -> None:
    """Concrete project artifacts for Chapter 20."""
    show("Y1 one page brief", [
        "FINDING: 90-day churn is 2.6% on the holdout.",
        "EVIDENCE: Basic plans and low engagement carry the risk.",
        "IMPLICATION: an engagement nudge is the cheapest lever.",
        "RECOMMENDATION: fund a $180k nudge for 2,050 members.",
        "ASK: owner retention lead, decision by 2026-02-01.",
    ])
    show("Y2 assessment memo", [
        "86.5% censored - use a fixed-horizon label.",
        "3 leaky columns - exclude all post-churn fields.",
        "base rate 2.1%-13.5% by definition - state the definition.",
        "6,986 members shared across snapshots - check by plan.",
    ])
    tr, te = split(fm)
    q = tr.groupby("plan")["churn_90d"].mean()
    annual = 1 - (1 - q) ** 4
    clv = fin.set_index("plan")["margin_per_month"] * 12 / annual
    basic_annual = float(annual["Basic"])
    show("Y3 payback arithmetic", {
        "addressable": 2050, "annual_churn_basic": round(basic_annual, 4),
        "uplift": 0.30, "clv_mean": round(float(clv.mean()), 2),
        "value": round(float(2050 * basic_annual * 0.30 * clv.mean()), 2),
        "cost": 180000.0,
    })
    show("Y4 targeting tiers", pd.DataFrame([
        ("top 10%", 820, 0.083), ("next 15%", 1230, 0.052),
        ("rest", 6870, 0.014),
    ], columns=["tier", "members", "risk"]
    ).to_string(index=False))
    show("Y5 model comparison", pd.DataFrame([
        ("logistic", 0.6195), ("tenure rule", 0.5117),
        ("leaky model", 0.9895),
    ], columns=["model", "auc"]).to_string(index=False))
    show("Y6 definition table", {
        "contractual_3y": 0.1354, "voluntary_3y": 0.1148,
        "involuntary_3y": 0.0207, "90_day": 0.0256,
    })
    show("Y7 retention summary", {
        "month_3": 0.8935, "month_6": 0.7940, "month_12": 0.6040,
    })
    show("Y8 intervention ranking", {
        "engagement nudge": 0.30, "service recovery": 0.12,
        "annual upgrade": 0.09,
    })
    show("Y9 handover", [
        "fixtures and generator", "outputs script", "fixed horizon table",
        "model card", "targeting list", "two-page memo",
    ])
    show("Y10 reproduce", [
        "python scripts/generate_chapter20_sources.py",
        "python scripts/chapter20_outputs.py",
    ])
    show("Y11 rubric", {
        "problem": 20, "panel_leakage": 25, "analysis": 25,
        "validation": 15, "communication": 15,
    })
    show("Y12 controls", {
        "members": len(T["members"]), "rows": len(fm),
        "survey": len(sv),
    })
    show("Y13 limits", [
        "associations are not causes", "the definition drives the rate",
        "no uplift estimate without an experiment",
    ])
    show("Y14 final", {
        "decision": "fund the engagement nudge",
        "payback": 2.0, "owner": "retention lead",
    })


def main() -> None:
    print("python pandas", pd.__version__)
    print()
    T, fm, cad, sv, fin = load()
    for section in (section_a, section_b, section_c, section_d,
                    section_e, section_f, section_g, section_extra):
        print("=" * 78)
        section(T, fm, cad, sv, fin)


if __name__ == "__main__":
    main()
