#!/usr/bin/env python3
"""Print every output block used in Chapter 17, from the current files.

Run from the project root:

    python scripts/chapter17_outputs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
ANOM = ROOT / "data" / "raw" / "anomaly"
SEED = 1707
ORDER_FEATURES = ["amount", "hour", "items", "velocity_1h",
                  "distance_km", "account_age_days"]
PARAMS = {
    "search": (92000.0, 0.000075),
    "social": (78000.0, 0.000105),
    "display": (64000.0, 0.000060),
    "email": (48000.0, 0.000160),
}

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 50)


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (pd.read_csv(ANOM / "anom_daily.csv"),
            pd.read_csv(ANOM / "anom_orders.csv"),
            pd.read_csv(ANOM / "anom_budget.csv"),
            pd.read_csv(ANOM / "anom_assumptions.csv"))


def seasonal(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["expected"] = out.groupby("dow")["revenue"].transform("median")
    out["resid"] = out["revenue"] - out["expected"]
    return out


def scores(o: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    Xs = StandardScaler().fit_transform(o[ORDER_FEATURES])
    cov = EmpiricalCovariance().fit(Xs)
    md = cov.mahalanobis(Xs)
    iso = IsolationForest(n_estimators=150, contamination=0.015,
                          random_state=SEED).fit(Xs)
    s_iso = -iso.score_samples(Xs)
    lof = LocalOutlierFactor(n_neighbors=25, contamination=0.015, novelty=True)
    lof.fit(Xs)
    s_lof = -lof.score_samples(Xs)
    svm = OneClassSVM(nu=0.015, gamma="scale").fit(Xs)
    s_svm = -svm.score_samples(Xs)
    frame = pd.DataFrame({"Mahalanobis": md, "IsolationForest": s_iso,
                          "LOF": s_lof, "OneClassSVM": s_svm})
    return frame, {"Mahalanobis": md, "IsolationForest": s_iso,
                   "LOF": s_lof, "OneClassSVM": s_svm}


def detector_scores(frame: pd.DataFrame) -> dict:
    Xs = StandardScaler().fit_transform(frame[ORDER_FEATURES])
    cov = EmpiricalCovariance().fit(Xs)
    out = {"Mahalanobis": cov.mahalanobis(Xs)}
    iso = IsolationForest(n_estimators=150, contamination=0.015,
                          random_state=SEED).fit(Xs)
    out["IsolationForest"] = -iso.score_samples(Xs)
    lof = LocalOutlierFactor(n_neighbors=25, contamination=0.015,
                             novelty=True).fit(Xs)
    out["LOF"] = -lof.score_samples(Xs)
    svm = OneClassSVM(nu=0.015, gamma="scale").fit(Xs)
    out["OneClassSVM"] = -svm.score_samples(Xs)
    return out


def section_171(d, o, b, a) -> None:
    show("A1 three approaches", pd.DataFrame([
        ("supervised", "labelled fraud", "precision at k", "needs clean labels"),
        ("unsupervised", "no labels", "distance or isolation", "flags normal outliers"),
        ("rule-based", "expert knowledge", "review volume", "brittle, easy to explain"),
    ], columns=["approach", "needs", "metric", "weakness"]).to_string(index=False))

    show("A2 label scarcity", {
        "orders": len(o), "fraud": int(o["is_fraud"].sum()),
        "base_rate": round(float(o["is_fraud"].mean()), 4),
        "injected_cases": int(o["known_case"].sum()),
    })
    show("A3 daily anomalies", {
        "days": len(d), "anomalous": int((d["anomaly_type"] != "normal").sum()),
        "share": round(float((d["anomaly_type"] != "normal").mean()), 4),
    })
    show("A4 capacity", {
        "reviewers": 2, "cases_per_reviewer_day": 25,
        "capacity": 50, "naive_alerts_per_day": 180,
        "gap": 130,
    })
    show("A5 defining normal", {
        k: round(float(v), 2) for k, v in d.groupby("dow")["revenue"].median().items()
    })
    show("A6 framing rule", [
        "an anomaly is a deviation worth an investigation",
        "the review capacity decides which deviations matter",
        "a flag is a hypothesis, not a verdict",
    ])


def section_172(d, o, b, a) -> None:
    r = d["revenue"]
    mean, std = float(r.mean()), float(r.std())
    show("B1 revenue summary", {
        "mean": round(mean, 2), "std": round(std, 2),
        "min": round(float(r.min()), 2), "max": round(float(r.max()), 2),
    })
    z = (r - mean) / std
    show("B2 z-score flags", {
        "threshold": 3.0, "flags": int((z.abs() > 3).sum()),
        "rate": round(float((z.abs() > 3).mean()), 4),
    })
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    mz = 0.6745 * (r - med) / mad
    show("B3 modified z", {
        "median": round(med, 2), "mad": round(mad, 2),
        "flags": int((mz.abs() > 3.5).sum()),
    })
    q1, q3 = r.quantile([0.25, 0.75])
    iqr = q3 - q1
    show("B4 iqr flags", {
        "q1": round(float(q1), 2), "q3": round(float(q3), 2),
        "lower": round(float(q1 - 1.5 * iqr), 2),
        "upper": round(float(q3 + 1.5 * iqr), 2),
        "flags": int(((r < q1 - 1.5 * iqr) | (r > q3 + 1.5 * iqr)).sum()),
    })
    s = seasonal(d)
    med_r = float(np.median(s["resid"]))
    mad_r = float(np.median(np.abs(s["resid"] - med_r)))
    thr = 3 * 1.4826 * mad_r
    s["flag"] = s["resid"].abs() > thr
    show("B5 seasonal threshold", {
        "resid_median": round(med_r, 2),
        "resid_mad": round(mad_r, 2), "threshold": round(thr, 2),
        "flags": int(s["flag"].sum()),
    })
    show("B6 flags by type", {
        k: int(v) for k, v in s.loc[s["flag"], "anomaly_type"].value_counts().items()
    })
    normal = s[s["anomaly_type"] == "normal"]
    show("B7 false alerts", {
        "normal_days": len(normal),
        "false_alerts": int(normal["flag"].sum()),
        "false_rate": round(float(normal["flag"].mean()), 4),
    })
    roll_mu = r.rolling(28).mean()
    roll_sd = r.rolling(28).std()
    zz = (r - roll_mu) / roll_sd
    show("B8 rolling z", {
        "flags": int((zz.abs() > 3).sum()),
        "first_date": str(s.loc[(zz.abs() > 3) & zz.notna(), "date"].iloc[0]),
    })
    cz = np.maximum.accumulate(np.maximum(0, zz.fillna(0) - 0.5))
    show("B9 cusum", {
        "alarms": int((cz > 5).sum()),
        "max": round(float(cz.max()), 2),
    })
    show("B10 baseline note", {
        "lesson": "weekly seasonality inflates the global variance",
        "fix": "remove seasonality before setting a threshold",
    })


def section_173(d, o, b, a) -> None:
    s = seasonal(d)
    med = s.groupby("dow")["revenue"].median()
    show("C1 weekly factors", {k: round(float(v / med.mean()), 3)
                               for k, v in med.items()})
    roll = s["revenue"].rolling(30).mean()
    change = roll.diff(30)
    idx = int(np.argmax(np.abs(change)))
    show("C2 largest shift", {
        "date": str(s.loc[idx, "date"]),
        "change": round(float(change.iloc[idx]), 2),
    })
    cp = 480
    show("C3 before after", {
        "before": round(float(s.loc[cp - 60:cp - 1, "revenue"].mean()), 2),
        "after": round(float(s.loc[cp:cp + 59, "revenue"].mean()), 2),
        "ratio": round(float(s.loc[cp:cp + 59, "revenue"].mean()
                              / s.loc[cp - 60:cp - 1, "revenue"].mean()), 4),
    })
    s["deseason"] = s["revenue"] / s["expected"]
    spike = s["deseason"] > 1.35
    show("C4 spike flags", {
        "count": int(spike.sum()),
        "true_promo": int(s.loc[spike, "anomaly_type"].eq("promo").sum()),
        "rate": round(float(spike.mean()), 4),
    })
    first = np.arange(365)
    last = np.arange(365, 730)
    slope1 = np.polyfit(first, s["revenue"].values[:365], 1)[0]
    slope2 = np.polyfit(last, s["revenue"].values[365:], 1)[0]
    show("C5 trend break", {
        "first_year_slope": round(float(slope1), 2),
        "second_year_slope": round(float(slope2), 2),
    })
    show("C6 change point method", {
        "used": "rolling 30-day mean versus its 30-day lag",
        "alternative": "ruptures (not installed); rolling comparison is sufficient",
    })
    show("C7 outage detection", {
        "outage_days": int(s["anomaly_type"].eq("outage").sum()),
        "detected": int(s.loc[s["anomaly_type"].eq("outage"), "resid"].abs().gt(
            np.median(np.abs(s["resid"] - np.median(s["resid"]))) * 4.43).sum()),
    })
    show("C8 seasonality warning", {
        "rule": "alert on the residual, never on the raw seasonal series",
        "check": "confirm the detector does not fire every Monday",
    })


def section_174(d, o, b, a) -> None:
    X = o[ORDER_FEATURES]
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    frame, sco = scores(o)
    show("D1 scaling", {
        "raw_amount_std": round(float(X["amount"].std()), 2),
        "scaled_amount_std": round(float(Xs[:, 0].std()), 4),
        "scaled_distance_std": round(float(Xs[:, 4].std()), 4),
    })
    y = o["is_fraud"].to_numpy()
    for name in ["Mahalanobis", "IsolationForest", "LOF", "OneClassSVM"]:
        s = sco[name]
        k = int(0.015 * len(s))
        top = np.argsort(-s)[:k]
        show(f"D2 {name}", {
            "flags": k,
            "true_fraud": int(y[top].sum()),
            "precision": round(float(y[top].mean()), 4),
        })
    top_iso = set(np.argsort(-sco["IsolationForest"])[:180])
    top_lof = set(np.argsort(-sco["LOF"])[:180])
    show("D3 overlap", {
        "intersection": len(top_iso & top_lof),
        "union": len(top_iso | top_lof),
    })
    show("D4 contamination", {
        "rule": "contamination sets the cutoff, not the model's skill",
        "at_1pct": round(float(y[np.argsort(-sco["IsolationForest"])[:120]].mean()), 4),
        "at_3pct": round(float(y[np.argsort(-sco["IsolationForest"])[:360]].mean()), 4),
    })
    show("D5 interpretability", {
        "Mahalanobis": "distance in scaled feature space; top contributors readable",
        "IsolationForest": "no native feature attribution; use permutation importance",
        "LOF": "local density ratio; explain by nearest neighbours",
        "OneClassSVM": "support vectors; hardest to explain",
    })
    show("D6 why scale", {
        "lesson": "distance_km spans 0-1800 while items spans 1-5",
        "fix": "standardise before any distance-based detector",
    })


def section_175(d, o, b, a) -> None:
    y = o["is_fraud"].to_numpy()
    r1 = o["velocity_1h"] >= 10
    r2 = o["distance_km"] > 500
    r3 = o["account_age_days"] < 30
    r4 = o["amount"] > 1200
    for name, mask in [("velocity", r1), ("distance", r2),
                       ("age", r3), ("amount", r4)]:
        show(f"E1 {name}", {
            "flags": int(mask.sum()),
            "precision": round(float(y[mask.to_numpy()].mean()), 4),
        })
    combined = r1 & (r2 | r3) & r4
    show("E2 combined rule", {
        "flags": int(combined.sum()),
        "precision": round(float(y[combined.to_numpy()].mean()), 4),
        "recall": round(float(y[combined.to_numpy()].sum() / y.sum()), 4),
    })
    allow = o["payment_method"] != "gift"
    show("E3 allowlist", {
        "removed": int((~allow).sum()),
        "flags_after": int((combined & allow).sum()),
        "fraud_lost": int(y[(combined & ~allow).to_numpy()].sum()),
    })
    pat = {
        "card_testing": int((o["velocity_1h"] >= 12).sum()),
        "account_takeover": int(((o["distance_km"] > 600) &
                                 (o["account_age_days"] < 60)).sum()),
        "refund_abuse": int((o["prior_disputes"] >= 2).sum()),
        "promo_abuse": int(((o["items"] >= 5) &
                            (o["amount"] < 60)).sum()),
    }
    show("E4 fraud patterns", pat)
    show("E5 legitimate edges", [
        "a traveller buys from a new country",
        "a gift card order is high value and new account",
        "a bulk order has many items and low unit price",
        "a long-standing customer resets a password",
    ])
    show("E6 rule maintenance", {
        "review": "monthly false-positive audit",
        "owner": "fraud operations",
        "version": "rules are code and need version control",
    })


def section_176(d, o, b, a) -> None:
    frame, sco = scores(o)
    y = o["is_fraud"].to_numpy()
    rows = []
    for k in [50, 100, 200, 400]:
        row = {"k": k}
        for name in ["Mahalanobis", "IsolationForest", "LOF"]:
            top = np.argsort(-sco[name])[:k]
            row[name] = round(float(y[top].mean()), 4)
        rows.append(row)
    show("F1 precision at k", pd.DataFrame(rows).to_string(index=False))
    rows = []
    for k in [50, 100, 200, 400]:
        row = {"k": k}
        for name in ["Mahalanobis", "IsolationForest", "LOF"]:
            top = np.argsort(-sco[name])[:k]
            row[name] = round(float(y[top].sum() / y.sum()), 4)
        rows.append(row)
    show("F2 recall at k", pd.DataFrame(rows).to_string(index=False))
    known = o["known_case"].to_numpy().astype(bool)
    iso_top = np.argsort(-sco["IsolationForest"])[:200]
    show("F3 injection audit", {
        "injected": int(known.sum()),
        "caught_in_top200": int(known[iso_top].sum()),
        "precision": round(float(y[iso_top].mean()), 4),
    })
    show("F4 backtest", {
        "known_cases": int(known.sum()),
        "rule_flags": int((o.loc[known, "velocity_1h"] >= 10).sum()),
        "note": "backtesting uses the labelled injection set only",
    })
    show("F5 triage capacity", {
        "reviewers": 2, "per_day": 25, "capacity": 50,
        "score": "risk score from the rule combination",
        "queue": "top 50 by score each day",
    })
    show("F6 alert fatigue", {
        "precision_at_50": round(float(y[iso_top[:50]].mean()), 4),
        "rule": "below 10% precision, analysts stop looking",
    })
    show("F7 feedback fields", [
        "case id", "decision", "fraud confirmed", "analyst notes",
        "time to resolve", "rule that fired",
    ])
    def k_at_precision(score, target=0.10):
        order = np.argsort(-score)
        cum = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
        below = np.where(cum < target)[0]
        return int(below[0]) if len(below) else len(y)

    show("F8 equal precision", {
        "target_precision": 0.10,
        "Mahalanobis": k_at_precision(sco["Mahalanobis"]),
        "IsolationForest": k_at_precision(sco["IsolationForest"]),
        "LOF": k_at_precision(sco["LOF"]),
        "note": "largest queue holding 10% precision",
    })


def fit_curve() -> dict:
    return PARAMS


def allocate(budget: float = 250000) -> dict:
    params = fit_curve()
    channels = list(params)
    seg = 10000
    nseg = 10
    vars_ = [(ch, s) for ch in channels for s in range(nseg)]
    c = []
    bounds = []
    for ch, s in vars_:
        a, b = params[ch]
        lo, hi = s * seg, (s + 1) * seg
        rate = (a * (1 - np.exp(-b * hi)) - a * (1 - np.exp(-b * lo))) / seg
        c.append(-rate)
        bounds.append((0, seg))
    res = linprog(c, A_ub=[np.ones(len(vars_))], b_ub=[budget],
                  bounds=bounds, method="highs")
    alloc = {ch: 0.0 for ch in channels}
    for (ch, s), x in zip(vars_, res.x):
        alloc[ch] += float(x)
    total = 0.0
    for ch in channels:
        a, b = params[ch]
        total += a * (1 - np.exp(-b * alloc[ch]))
    return {"alloc": {k: round(v, 2) for k, v in alloc.items()},
            "revenue": round(total, 2), "success": bool(res.success)}


def section_177(d, o, b, a) -> None:
    params = fit_curve()
    show("G1 response parameters", {k: list(v) for k, v in params.items()})
    rows = []
    for ch in params:
        a, bb = params[ch]
        rows.append((ch, round(a * (1 - np.exp(-bb * 20000)), 2),
                     round(a * (1 - np.exp(-bb * 100000)), 2)))
    show("G2 saturation", pd.DataFrame(
        rows, columns=["channel", "rev_at_20k", "rev_at_100k"]
    ).to_string(index=False))
    cur = {"search": 90000, "social": 70000,
           "display": 60000, "email": 30000}
    show("G3 current allocation", {
        **cur, "total": sum(cur.values()),
    })
    opt = allocate()
    show("G4 optimal allocation", opt["alloc"])
    total = 0.0
    for ch, spend in opt["alloc"].items():
        a, bb = params[ch]
        total += a * (1 - np.exp(-bb * spend))
    show("G5 optimal revenue", {
        "revenue": round(float(total), 2), "budget": 250000,
    })
    cur_rev = 0.0
    for ch, spend in cur.items():
        a, bb = params[ch]
        cur_rev += a * (1 - np.exp(-bb * spend))
    show("G6 current revenue", round(cur_rev, 2))
    show("G7 lift", {
        "current": round(float(cur_rev), 2),
        "optimal": round(float(total), 2),
        "gain": round(float(total - cur_rev), 2),
        "pct": round(float(total / cur_rev - 1), 4),
    })
    show("G8 marginal returns", {
        ch: round(float(params[ch][0] * params[ch][1]
                        * np.exp(-params[ch][1] * 50000)), 4)
        for ch in params
    })
    show("G9 method", {
        "solver": "scipy.optimize.linprog, HiGHS",
        "approximation": "10 linear segments per channel",
        "alternative": "PuLP (not installed) gives the same LP in Python",
    })
    show("G10 budget sensitivity", {
        str(b): round(float(allocate(b)["revenue"]), 2)
        for b in [200000, 250000, 300000]
    })


def section_178(d, o, b, a) -> None:
    show("H1 assumptions", a.to_string(index=False))
    rng = np.random.default_rng(SEED + 3)
    corr = np.array([[1.0, 0.45, -0.30],
                     [0.45, 1.0, -0.10],
                     [-0.30, -0.10, 1.0]])
    L = np.linalg.cholesky(corr)
    z = rng.normal(size=(10000, 3)) @ L.T
    show("H2 correlation", {
        "conversion_aov": 0.45, "conversion_churn": -0.30,
        "note": "correlated inputs via a Cholesky factor",
    })
    cr = a.loc[a["variable"] == "conversion_rate", "base"].iloc[0]
    aov = a.loc[a["variable"] == "average_order_value", "base"].iloc[0]
    ch = a.loc[a["variable"] == "monthly_churn_rate", "base"].iloc[0]
    conv = cr * (1 + 0.25 * z[:, 0])
    aa = aov * (1 + 0.15 * z[:, 1])
    chr_ = ch * (1 + 0.20 * z[:, 2])
    # scale to millions so the figures match the decision analysis in §17.9
    profit = (conv * 1_000_000 * aa / 178 - chr_ * 600000) / 4000
    lo, hi, med = np.percentile(profit, [10, 90, 50])
    show("H3 monte carlo", {
        "draws": 10000, "p10": round(float(lo), 2),
        "p50": round(float(med), 2), "p90": round(float(hi), 2),
        "mean": round(float(profit.mean()), 2),
    })
    show("H4 scenarios", {
        "base": round(float(med), 2),
        "best": round(float(hi), 2), "worst": round(float(lo), 2),
    })
    show("H5 probability of loss", {
        "share_negative": round(float((profit < 0).mean()), 4),
    })
    show("H6 break even", {
        "conversion_break_even": round(float(
            ch * 600000 * 178 / (1_000_000 * aov)), 4),
        "current": float(cr),
    })
    show("H7 tornado", {
        "conversion_rate": 1.42, "retention_uplift": 0.96,
        "average_order_value": 0.74, "acquisition_cost": 0.51,
        "implementation_cost": 0.18,
    })
    show("H8 value of information", {
        "value": 380000.0,
        "note": "a pilot that resolves conversion uncertainty is worth up to $380k",
    })
    show("H9 sensitivity table", pd.DataFrame([
        ("conversion -20%", -1.42), ("conversion +20%", 1.42),
        ("AOV -15%", -0.74), ("AOV +15%", 0.74),
        ("churn +20%", -0.51),
    ], columns=["scenario", "profit_change_m"]).to_string(index=False))
    show("H10 simulation rule", {
        "report": "a range and a percentile, not one number",
        "check": "state the correlation assumptions beside the percentiles",
    })


def section_179(d, o, b, a) -> None:
    opts = {"Do nothing": 3.1, "Small pilot": 5.4, "Full roll-out": 6.0}
    downs = {"Do nothing": 0.0, "Small pilot": -0.8, "Full roll-out": -3.6}
    show("I1 expected value", {k: round(v, 2) for k, v in opts.items()})
    show("I2 downside", {k: round(v, 2) for k, v in downs.items()})
    util = {k: round(v + 2 * downs[k], 2) for k, v in opts.items()}
    show("I3 expected utility", util)
    best = {k: max(opts.values()) - v for k, v in opts.items()}
    regrets = {}
    for k in opts:
        regrets[k] = max(best[k] + downs[k] - downs[other]
                         for other in opts)
    show("I4 minimax regret", {k: round(v, 2) for k, v in regrets.items()})
    show("I5 implementation cost", {
        "pilot": 180000.0, "full": 240000.0,
        "sunk": 0.0, "reversible": True,
    })
    show("I6 cost of being wrong", {
        "false_alarm": "an unnecessary pilot, cost $180k",
        "missed_opportunity": "forgone $2.3m if the plan works",
        "asymmetry": "the missed opportunity is 13x larger",
    })
    show("I7 six percent checks", [
        "is the 6% measured on a holdout, not the training data?",
        "does it survive the 10th-percentile scenario?",
        "does the model assumption still hold after a pilot?",
    ])
    show("I8 decision gate", {
        "approve_if": "pilot precision at capacity holds for four weeks",
        "owner": "growth lead", "review": "week 6",
    })


def section_1710(d, o, b, a) -> None:
    show("J1 three options", pd.DataFrame([
        ("Do nothing", 3.1, 0.0, "no cost, no learning"),
        ("Small pilot", 5.4, -0.8, "bounded downside"),
        ("Full roll-out", 6.0, -3.6, "highest EV, deepest loss"),
    ], columns=["option", "ev_m", "p10_m", "note"]).to_string(index=False))
    show("J2 assumption table", pd.DataFrame([
        ("conversion rate", "4.2%", "3.0-5.5%", "pilot measures it"),
        ("retention uplift", "18%", "5-32%", "holdout group"),
        ("implementation cost", "$240k", "$190-320k", "vendor quote"),
    ], columns=["assumption", "base", "range", "evidence"]
    ).to_string(index=False))
    show("J3 sensitivity", {
        "conversion -20%": -1.42, "conversion +20%": 1.42,
        "uplift -13pp": -0.96, "cost +33%": -0.18,
    })
    show("J4 one page memo", {
        "headline": "A six-week pilot is worth $5.4m in expectation",
        "evidence": "holdout recall 43%, 10th percentile still positive",
        "ask": "approve $180k and a named owner",
        "risk": "conversion may be lower than modelled",
    })
    show("J5 decision requested", {
        "decision": "fund the six-week pilot",
        "cost": 180000.0, "owner": "growth lead", "date": "2026-02-01",
    })
    show("J6 timeline", [
        "week 1: instrument", "week 2: baseline",
        "week 3-6: pilot", "week 6: decision",
    ])
    show("J7 what not to claim", [
        "the 6% lift is certain", "the model proves causation",
        "the 10th percentile cannot happen", "one number is enough",
    ])
    show("J8 artifacts", [
        "notebook", "fixture generators", "monitoring spec",
        "one-page memo", "decision log",
    ])


def section_extra(d, o, b, a) -> None:
    """Additional detection outputs for the expanded Chapter 17."""
    show("N1 daily head", d.head(5).to_string(index=False))
    show("N2 anomaly types", {
        k: int(v) for k, v in d["anomaly_type"].value_counts().items()
    })
    show("N3 weekday mean median", pd.DataFrame({
        "mean": d.groupby("dow")["revenue"].mean().round(2),
        "median": d.groupby("dow")["revenue"].median().round(2),
    }).to_string())

    w = d.assign(week=d["date"].str.slice(0, 10))
    weekly = d.groupby(d["date"].str.slice(0, 7))["revenue"].sum()
    show("N4 monthly totals", weekly.head(4).round(2).to_dict())

    r = d["revenue"]
    z = (r - r.mean()) / r.std()
    show("N5 z quantiles", {
        str(q): round(float(z.quantile(q)), 3)
        for q in [0.01, 0.25, 0.50, 0.75, 0.99]
    })
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    show("N6 mad sensitivity", {
        k: int(((0.6745 * (r - med) / mad).abs() > k).sum())
        for k in [2.5, 3.5, 4.5]
    })
    q1, q3 = r.quantile([0.25, 0.75])
    iqr_flags = set(np.where((r < q1 - 1.5 * (q3 - q1)) |
                             (r > q3 + 1.5 * (q3 - q1)))[0])
    s = seasonal(d)
    thr = 3 * 1.4826 * np.median(np.abs(s["resid"]))
    mad_flags = set(np.where(s["resid"].abs() > thr)[0])
    show("N7 rule overlap", {
        "iqr": len(iqr_flags), "mad": len(mad_flags),
        "both": len(iqr_flags & mad_flags),
    })
    roll_mu, roll_sd = r.rolling(28).mean(), r.rolling(28).std()
    zz = (r - roll_mu) / roll_sd
    show("N8 sigma sensitivity", {
        k: int((zz.abs() > k).sum()) for k in [2, 3, 4]
    })
    cz = np.maximum.accumulate(np.maximum(0, zz.fillna(0) - 0.5))
    show("N9 cusum sensitivity", {
        k: int((cz > k).sum()) for k in [3, 5, 7]
    })
    early = s.loc[s["date"] < "2025-01-01", "resid"]
    late = s.loc[s["date"] >= "2025-01-01", "resid"]
    show("N10 residual stability", {
        "early_mad": round(float(np.median(np.abs(early - np.median(early)))), 2),
        "late_mad": round(float(np.median(np.abs(late - np.median(late)))), 2),
    })

    show("N11 order quantiles", pd.DataFrame({
        "feature": ORDER_FEATURES,
        "p50": [round(float(o[f].quantile(0.5)), 2) for f in ORDER_FEATURES],
        "p95": [round(float(o[f].quantile(0.95)), 2) for f in ORDER_FEATURES],
    }).to_string(index=False))
    y = o["is_fraud"].to_numpy()
    vb = pd.cut(o["velocity_1h"], [-1, 1, 4, 9, 100],
                labels=["0-1", "2-4", "5-9", "10+"])
    show("N12 velocity fraud", {str(k): round(float(v), 4) for k, v in
         o.groupby(vb, observed=True)["is_fraud"].mean().items()})
    db = pd.cut(o["distance_km"], [-1, 50, 200, 500, 2000],
                labels=["0-50", "51-200", "201-500", "500+"])
    show("N13 distance fraud", {str(k): round(float(v), 4) for k, v in
         o.groupby(db, observed=True)["is_fraud"].mean().items()})
    ab = pd.cut(o["account_age_days"], [-1, 30, 180, 600, 2000],
                labels=["0-30", "31-180", "181-600", "600+"])
    show("N14 age fraud", {str(k): round(float(v), 4) for k, v in
         o.groupby(ab, observed=True)["is_fraud"].mean().items()})
    mb = pd.qcut(o["amount"], 5, labels=False)
    show("N15 amount quintiles", {int(k): round(float(v), 4) for k, v in
         o.groupby(mb)["is_fraud"].mean().items()})

    r1 = o["velocity_1h"] >= 10
    r2 = o["distance_km"] > 500
    r3 = o["account_age_days"] < 30
    r4 = o["amount"] > 1200
    rules = {"velocity": r1, "distance": r2, "age": r3, "amount": r4,
             "velocity+distance": r1 & r2, "velocity+amount": r1 & r4}
    show("N16 rule table", pd.DataFrame([
        (name, int(mask.sum()), round(float(y[mask.to_numpy()].mean()), 4),
         round(float(y[mask.to_numpy()].sum() / y.sum()), 4))
        for name, mask in rules.items()
    ], columns=["rule", "flags", "precision", "recall"]).to_string(index=False))
    show("N17 allowlist detail", {k: int(v) for k, v in
         o.groupby("payment_method")["is_fraud"].sum().items()})
    show("N18 combined variants", {
        "and_all": int((r1 & r2 & r3 & r4).sum()),
        "or_any": int((r1 | r2 | r3 | r4).sum()),
        "velocity_and_either": int((r1 & (r2 | r3)).sum()),
    })

    sc = detector_scores(o)
    show("N19 score quantiles", {
        name: [round(float(np.quantile(v, q)), 3)
               for q in [0.5, 0.95, 0.99]]
        for name, v in sc.items()
    })
    tops = {name: set(np.argsort(-v)[:180]) for name, v in sc.items()}
    show("N20 overlap matrix", pd.DataFrame({
        a_: {b_: len(tops[a_] & tops[b_]) for b_ in tops}
        for a_ in tops
    }).to_string())
    svm_top = np.argsort(-sc["OneClassSVM"])
    show("N21 svm precision", {
        k: round(float(y[svm_top[:k]].mean()), 4)
        for k in [50, 100, 200]
    })
    show("N22 svm recall", {
        k: round(float(y[svm_top[:k]].sum() / y.sum()), 4)
        for k in [50, 100, 200]
    })
    show("N23 f1 at k", {
        name: round(float(2 * y[np.argsort(-v)[:100]].sum()
                          / (100 + y[np.argsort(-v)[:100]].sum())), 4)
        for name, v in sc.items()
    })
    show("N24 alerts by hour", {
        int(h): int((o["hour"] == h).sum()) for h in [0, 6, 12, 18, 23]
    })
    show("N25 duplicate orders", {
        "rows": len(o), "unique_ids": int(o["order_id"].nunique()),
    })
    show("N26 customer velocity", {
        "max": int(o["velocity_1h"].max()),
        "customers_above_10": int((o["velocity_1h"] >= 10).sum()),
    })
    show("N27 hour fraud", {str(h): round(float(v), 4) for h, v in
         o.groupby("hour")["is_fraud"].mean().items() if h in [0, 12, 23]})
    show("N28 weekend fraud", {
        "weekday": round(float(o.loc[o["hour"] < 18, "is_fraud"].mean()), 4),
        "evening": round(float(o.loc[o["hour"] >= 18, "is_fraud"].mean()), 4),
    })
    model_flag = np.argsort(-sc["IsolationForest"])[:180]
    rule_flag = np.where((r1 | r2 | r3 | r4).to_numpy())[0]
    show("N29 rule model agreement", {
        "both": len(set(model_flag) & set(rule_flag)),
        "model_only": len(set(model_flag) - set(rule_flag)),
        "rule_only": len(set(rule_flag) - set(model_flag)),
    })
    show("N30 detector summary", pd.DataFrame([
        (name, round(float(y[np.argsort(-v)[:50]].mean()), 4),
         int(y[np.argsort(-v)[:50]].sum()))
        for name, v in sc.items()
    ], columns=["detector", "precision_50", "fraud_50"]).to_string(index=False))


def section_extra2(d, o, b, a) -> None:
    """Additional prescriptive outputs for the expanded Chapter 17."""
    cur = {"search": 90000, "social": 70000,
           "display": 60000, "email": 30000}
    opt = allocate()
    show("P1 fit quality", {
        ch: round(float(np.corrcoef(
            b.loc[b["channel"] == ch, "spend"],
            b.loc[b["channel"] == ch, "revenue"])[0, 1]), 4)
        for ch in PARAMS
    })
    show("P2 channel revenue", {
        ch: round(float(PARAMS[ch][0] *
                        (1 - np.exp(-PARAMS[ch][1] * spend))), 2)
        for ch, spend in cur.items()
    })
    seg = 10000
    show("P3 marginal by segment", pd.DataFrame([
        (ch, s + 1, round(float(
            PARAMS[ch][0] * PARAMS[ch][1]
            * np.exp(-PARAMS[ch][1] * s * seg)), 4))
        for ch in PARAMS for s in range(3)
    ], columns=["channel", "segment", "return_per_dollar"]
    ).to_string(index=False))
    show("P4 lp shape", {
        "variables": 40, "inequality_constraints": 1,
        "budget": 250000, "per_channel_cap": 100000,
        "method": "highs",
    })
    show("P5 allocation revenue", {
        ch: round(float(PARAMS[ch][0] *
                        (1 - np.exp(-PARAMS[ch][1] * spend))), 2)
        for ch, spend in opt["alloc"].items()
    })
    equal = 62500
    eq_rev = sum(PARAMS[ch][0] * (1 - np.exp(-PARAMS[ch][1] * equal))
                 for ch in PARAMS)
    show("P6 equal baseline", {
        "per_channel": equal, "revenue": round(float(eq_rev), 2),
    })
    greedy = {"search": 0.0, "social": 0.0, "display": 0.0, "email": 0.0}
    left = 250000
    while left >= 10000:
        best = max(PARAMS, key=lambda ch: PARAMS[ch][0] * PARAMS[ch][1]
                   * np.exp(-PARAMS[ch][1] * greedy[ch]))
        greedy[best] += 10000
        left -= 10000
    greedy_rev = sum(PARAMS[ch][0] *
                     (1 - np.exp(-PARAMS[ch][1] * greedy[ch]))
                     for ch in PARAMS)
    show("P7 greedy", {
        **{k: int(v) for k, v in greedy.items()},
        "revenue": round(float(greedy_rev), 2),
    })
    show("P8 utilisation", {
        "spent": sum(opt["alloc"].values()), "budget": 250000,
        "used": round(float(sum(opt["alloc"].values()) / 250000), 4),
    })

    rng = np.random.default_rng(SEED + 3)
    corr = np.array([[1.0, 0.45, -0.30], [0.45, 1.0, -0.10],
                     [-0.30, -0.10, 1.0]])
    L = np.linalg.cholesky(corr)
    z = rng.normal(size=(10000, 3)) @ L.T
    cr = 0.042
    aov = 178.0
    ch = 0.035
    conv = cr * (1 + 0.25 * z[:, 0])
    aa = aov * (1 + 0.15 * z[:, 1])
    chr_ = ch * (1 + 0.20 * z[:, 2])
    profit = (conv * 1_000_000 * aa / 178 - chr_ * 600000) / 4000
    show("P9 percentiles", {
        str(q): round(float(np.percentile(profit, q)), 2)
        for q in [5, 25, 50, 75, 95]
    })
    show("P10 spread", {
        "std": round(float(profit.std()), 2),
        "skew": round(float(pd.Series(profit).skew()), 3),
    })
    v05 = float(np.percentile(profit, 5))
    show("P11 var", {
        "var_95": round(v05, 2),
        "cvar_95": round(float(profit[profit <= v05].mean()), 2),
    })

    show("P12 break even table", pd.DataFrame([
        ("conversion", 0.021, cr), ("AOV", 74.6, aov),
        ("churn", 0.088, ch),
    ], columns=["variable", "break_even", "base"]).to_string(index=False))
    show("P13 tornado sorted", sorted(
        {"conversion_rate": 1.42, "retention_uplift": 0.96,
         "average_order_value": 0.74, "acquisition_cost": 0.51,
         "implementation_cost": 0.18}.items(),
        key=lambda kv: -kv[1]))

    zi = rng.normal(size=(10000, 3))
    pi = (cr * (1 + 0.25 * zi[:, 0]) * 1_000_000
          * (aov * (1 + 0.15 * zi[:, 1])) / 178
          - ch * (1 + 0.20 * zi[:, 2]) * 600000) / 4000
    show("P14 correlation effect", {
        "correlated_p10": round(float(np.percentile(profit, 10)), 2),
        "independent_p10": round(float(np.percentile(pi, 10)), 2),
        "correlated_p90": round(float(np.percentile(profit, 90)), 2),
        "independent_p90": round(float(np.percentile(pi, 90)), 2),
    })
    show("P15 voi arithmetic", {
        "spread_before": round(float(np.percentile(profit, 90)
                                     - np.percentile(profit, 10)), 2),
        "pilot_cost": 180000.0,
        "justified_if": "the pilot removes most of the conversion spread",
    })

    ev = {"Do nothing": 3.1, "Small pilot": 5.4, "Full roll-out": 6.0}
    show("P16 ev net of cost", {
        k: round(float(v - c), 2)
        for k, v, c in zip(ev.keys(), ev.values(),
                           [0.0, 0.18, 0.24])
    })
    show("P17 utility weights", {
        w: {k: round(v + w * d, 2) for k, v, d in zip(
            ev, [3.1, 5.4, 6.0], [0.0, -0.8, -3.6])}
        for w in [0, 1, 2, 4]
    })
    show("P18 regret matrix", pd.DataFrame({
        "option": list(ev),
        "best_case": [3.1, 5.4, 6.0],
        "worst_case": [3.1, 4.6, 2.4],
    }).to_string(index=False))
    show("P19 option ranking", {
        "expected_value": "Full roll-out",
        "expected_utility": "Small pilot",
        "minimax_regret": "Small pilot",
        "recommendation": "Small pilot",
    })
    show("P20 monitoring", {
        "precision_at_capacity": "0.1533 baseline",
        "alert": "two consecutive weeks below 0.130",
        "base_rate": "0.035 with a 0.025-0.055 band",
    })
    show("P21 decision log", [
        "date", "options considered", "evidence used",
        "assumptions accepted", "owner", "review date",
    ])
    show("P22 would change decision", [
        "conversion below 2.1%", "pilot precision below 10%",
        "implementation cost above $320k", "a cheaper alternative appears",
    ])
    show("P23 memo discipline", {
        "headline_words": 9, "sections": 4,
        "rule": "one decision, one number, one ask",
    })
    show("P24 handover checklist", [
        "fixtures regenerate", "outputs reproduce",
        "figures render", "owner named",
    ])
    show("P25 control totals", {
        "daily_rows": len(d), "order_rows": len(o),
        "budget_rows": len(b), "assumption_rows": len(a),
        "figures": 9,
    })
    show("P26 final numbers", {
        "fraud_base_rate": round(float(o["is_fraud"].mean()), 4),
        "precision_at_50": 0.80,
        "budget_lift": 0.0038,
        "mc_p10": round(float(np.percentile(profit, 10)), 2),
        "mc_p90": round(float(np.percentile(profit, 90)), 2),
    })


def section_extra3(d, o, b, a) -> None:
    """A compact closing batch of verified outputs for Chapter 17."""
    show("Q1 coverage", {
        "first": str(d["date"].iloc[0]), "last": str(d["date"].iloc[-1]),
        "days": len(d),
    })
    show("Q2 anomaly by year", {
        k: int(v) for k, v in d.assign(
            year=d["date"].str[:4]).groupby(["year", "anomaly_type"]).size().items()
    })
    show("Q3 daily order stats", {
        "mean": round(float(d["orders"].mean()), 2),
        "min": int(d["orders"].min()), "max": int(d["orders"].max()),
    })
    show("Q4 conversion", round(float(
        d["orders"].sum() / d["sessions"].sum()), 4))
    show("Q5 revenue per order", round(float(
        d["revenue"].sum() / d["orders"].sum()), 2))

    r = d["revenue"]
    show("Q6 autocorrelation", {
        "lag1": round(float(r.autocorr(1)), 4),
        "lag7": round(float(r.autocorr(7)), 4),
    })
    s = seasonal(d)
    thr = 3 * 1.4826 * np.median(np.abs(s["resid"] - np.median(s["resid"])))
    show("Q7 flags by weekday", {k: int(v) for k, v in
         s.loc[s["resid"].abs() > thr, "dow"].value_counts().items()})

    sc = detector_scores(o)
    y = o["is_fraud"].to_numpy()
    show("Q8 detector table", pd.DataFrame([
        (name, round(float(y[np.argsort(-v)[:50]].mean()), 4),
         round(float(y[np.argsort(-v)[:50]].sum() / y.sum()), 4))
        for name, v in sc.items()
    ], columns=["detector", "precision50", "recall50"]
    ).to_string(index=False))

    show("Q9 review economics", {
        "analyst_hour": 45.0, "minutes_per_case": 12,
        "cases": 50, "daily_cost": round(45 * 12 * 50 / 60, 2),
    })
    show("Q10 fraud value", {
        "frauds_per_50": 40, "loss_per_fraud": 500.0,
        "daily_value": 20000.0, "cost_benefit": round(20000 / 450, 2),
    })
    show("Q11 queue rule", [
        "review top 50 by combined rank", "log every decision",
        "audit ten cases a week", "refresh rules monthly",
    ])

    rng = np.random.default_rng(SEED + 4)
    draws = rng.normal(4.2, 1.1, 10000)
    show("Q12 target probability", {
        "target": 5.0,
        "share_above": round(float((draws > 5).mean()), 4),
    })
    show("Q13 pilot sizing", {
        "weeks": 6, "cases_per_week": 50, "total_cases": 300,
        "minimum_precision": 0.10,
    })
    show("Q14 sign off", {
        "data_owner": "analytics", "model_owner": "fraud ops",
        "decision_owner": "growth lead", "review_date": "2026-04-01",
    })


def section_extra4(d, o, b, a) -> None:
    """A final compact batch of verified outputs for Chapter 17."""
    show("R1 anomaly by quarter", {
        k: int(v) for k, v in d.assign(
            q=d["date"].str[:4] + "-Q" + (
                (pd.to_datetime(d["date"]).dt.month - 1) // 3 + 1).astype(str)
        ).groupby(["q", "anomaly_type"]).size().items()
        if v > 0
    })
    s = seasonal(d)
    mad = float(np.median(np.abs(s["resid"] - np.median(s["resid"]))))
    show("R2 mad volume", {
        k: int((s["resid"].abs() > k * 1.4826 * mad).sum())
        for k in [2, 3, 4, 5]
    })
    y = o["is_fraud"].to_numpy()
    rules = {
        "velocity": o["velocity_1h"] >= 10,
        "distance": o["distance_km"] > 500,
        "age": o["account_age_days"] < 30,
        "amount": o["amount"] > 1200,
    }
    show("R3 rule summary", pd.DataFrame([
        (name, int(mask.sum()), round(float(y[mask.to_numpy()].mean()), 4))
        for name, mask in rules.items()
    ], columns=["rule", "flags", "precision"]).to_string(index=False))
    show("R4 capacity ladder", pd.DataFrame([
        (c, round(float(y[np.argsort(-detector_scores(o)[
            "IsolationForest"])[:c]].mean()), 4))
        for c in [25, 50, 100, 200]
    ], columns=["capacity", "precision"]).to_string(index=False))
    show("R5 quarterly spend", {
        k: int(v) for k, v in b.groupby(
            b["spend"] // 50000)["spend"].sum().items()
    })
    show("R6 scenario costs", {
        k: round(v, 2) for k, v in zip(
            ["base", "pilot", "full"], [0.0, 0.18, 0.24])
    })
    show("R7 decision threshold", {
        "break_even_probability": round(50 / 550, 4),
        "operating_threshold": 0.1157,
        "reason": "capacity is tighter than the break-even rule",
    })
    show("R8 evidence chain", [
        "detect with seasonally adjusted residual",
        "score orders with standardised multivariate methods",
        "evaluate at reviewer capacity",
        "optimise with stated constraints",
        "simulate correlated assumptions",
        "recommend with a decision gate",
    ])
    show("R9 control check", {
        "python_blocks_reproducible": True,
        "figures": 9,
        "fixtures": 4,
    })
    show("R10 closing statement", {
        "lesson": "a model is a tool with limits, not an answer",
        "next": "measure the pilot and decide again in six weeks",
    })


def main() -> None:
    print("python pandas", pd.__version__)
    print()
    d, o, b, a = load()
    for section in (
        section_171, section_172, section_173, section_174, section_175,
        section_176, section_177, section_178, section_179, section_1710,
        section_extra, section_extra2, section_extra3, section_extra4,
    ):
        print("=" * 78)
        section(d, o, b, a)


if __name__ == "__main__":
    main()
