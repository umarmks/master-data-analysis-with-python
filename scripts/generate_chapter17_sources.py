#!/usr/bin/env python3
"""Generate the anomaly and prescriptive fixtures Chapter 17 uses.

Run from the project root:

    python scripts/generate_chapter17_sources.py

Outputs (fixtures under data/raw/anomaly/, figures under images/):

    anom_daily.csv       730 days of revenue with injected anomalies
    anom_orders.csv      12,000 orders with a rare fraud label
    anom_budget.csv      four channels with concave response curves
    anom_assumptions.csv six Monte Carlo inputs with low/base/high ranges

Everything is deterministic, including the figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
ANOM = ROOT / "data" / "raw" / "anomaly"
IMAGES = ROOT / "images"
SEED = 1707

DAILY_FEATURES = ["revenue", "orders", "sessions"]
ORDER_FEATURES = ["amount", "hour", "items", "velocity_1h",
                  "distance_km", "account_age_days"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def daily() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = 730
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    trend = 1 + 0.00035 * np.arange(n)
    dow = np.array([1.12, 0.94, 0.93, 0.96, 1.02, 1.28, 1.31])
    week = np.tile(dow, n // 7 + 1)[:n]
    doy = np.asarray(dates.dayofyear)
    annual = 1 + 0.10 * np.sin(2 * np.pi * (doy - 20) / 365)
    base = 18500 * trend * week * annual
    revenue = np.round(base * rng.normal(1, 0.045, n), 2)
    orders = np.round(revenue / 185 * rng.normal(1, 0.02, n)).astype(int)
    sessions = np.round(orders / 0.062 * rng.normal(1, 0.02, n)).astype(int)

    kind = np.array(["normal"] * n, dtype=object)
    # three-day outage with a clear dip
    revenue[200:203] *= np.array([0.55, 0.42, 0.71])
    kind[200:203] = "outage"
    # permanent level shift upward
    revenue[480:] *= 1.24
    kind[480:520] = "level_shift"
    # a single promotion spike
    revenue[600] *= 1.85
    kind[600] = "promo"
    # two isolated spikes
    revenue[320] *= 1.55
    revenue[655] *= 1.48
    kind[320] = kind[655] = "spike"
    revenue = np.round(revenue, 2)
    orders[200:203] = np.round(orders[200:203] * 0.6).astype(int)
    orders[480:] = np.round(orders[480:] * 1.18).astype(int)

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "dow": dates.day_name(),
        "revenue": revenue,
        "orders": orders,
        "sessions": sessions,
        "anomaly_type": kind,
    })
    df.to_csv(ANOM / "anom_daily.csv", index=False)
    print(f"  anom_daily.csv: {len(df)} days, "
          f"{int((kind != 'normal').sum())} anomalous")
    return df


def orders() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    n = 12000
    amount = np.round(np.exp(rng.normal(4.6, 0.65, n)), 2)
    hour = rng.integers(0, 24, n)
    items = rng.integers(1, 6, n)
    velocity = rng.poisson(1.4, n)
    distance = np.round(rng.gamma(2.0, 22, n), 1)
    age = rng.integers(1, 1500, n)
    disputes = rng.poisson(0.15, n)
    method = rng.choice(["card", "bank", "wallet", "gift"], n,
                        p=[0.58, 0.21, 0.17, 0.04])

    logit = (-5.6
             + 0.42 * velocity
             + 0.010 * distance
             - 0.0016 * age
             + 0.60 * (hour >= 23)
             + 0.40 * (method == "gift")
             + 0.55 * disputes
             + 0.25 * (amount > 900))
    fraud = rng.binomial(1, sigmoid(logit), size=n)

    df = pd.DataFrame({
        "order_id": [f"O-{i + 1:06d}" for i in range(n)],
        "customer_id": [f"C-{i % 4800 + 1:05d}" for i in range(n)],
        "amount": amount,
        "hour": hour,
        "items": items,
        "velocity_1h": velocity,
        "distance_km": distance,
        "account_age_days": age,
        "prior_disputes": disputes,
        "payment_method": method,
        "is_fraud": fraud,
        "known_case": 0,
    })
    # inject 40 extreme, labelled cases for the holdout audit
    idx = rng.choice(n, size=40, replace=False)
    df.loc[idx, "velocity_1h"] = rng.integers(14, 30, 40)
    df.loc[idx, "distance_km"] = np.round(rng.uniform(600, 1800, 40), 1)
    df.loc[idx, "account_age_days"] = rng.integers(1, 20, 40)
    df.loc[idx, "amount"] = np.round(rng.uniform(1200, 3000, 40), 2)
    df.loc[idx, "is_fraud"] = 1
    df.loc[idx, "known_case"] = 1
    df.to_csv(ANOM / "anom_orders.csv", index=False)
    print(f"  anom_orders.csv: {len(df)} orders, {int(df['is_fraud'].sum())} fraud")
    return df


def budget() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    rows = []
    params = {
        "search": (92000.0, 0.000075),
        "social": (78000.0, 0.000105),
        "display": (64000.0, 0.000060),
        "email": (48000.0, 0.000160),
    }
    for channel, (a, b) in params.items():
        for spend in np.arange(5000, 100001, 5000):
            rev = a * (1 - np.exp(-b * spend))
            rows.append((channel, int(spend), round(
                rev * rng.normal(1, 0.02), 2)))
    df = pd.DataFrame(rows, columns=["channel", "spend", "revenue"])
    df.to_csv(ANOM / "anom_budget.csv", index=False)
    print(f"  anom_budget.csv: {len(df)} observations")
    return df


def assumptions() -> pd.DataFrame:
    rows = [
        ("conversion_rate", 0.042, 0.030, 0.055),
        ("average_order_value", 178.0, 150.0, 210.0),
        ("monthly_churn_rate", 0.035, 0.025, 0.050),
        ("acquisition_cost", 96.0, 78.0, 120.0),
        ("retention_uplift", 0.18, 0.05, 0.32),
        ("implementation_cost", 240000.0, 190000.0, 320000.0),
    ]
    df = pd.DataFrame(rows, columns=["variable", "base", "low", "high"])
    df.to_csv(ANOM / "anom_assumptions.csv", index=False)
    print(f"  anom_assumptions.csv: {len(df)} variables")
    return df


def figures(d: pd.DataFrame, o: pd.DataFrame,
            b: pd.DataFrame, a: pd.DataFrame) -> None:
    from scipy import stats as sps
    from scipy.optimize import linprog

    # 1. seasonal residual with 3-MAD threshold
    df = d.copy()
    df["expected"] = df.groupby("dow")["revenue"].transform("median")
    df["resid"] = df["revenue"] - df["expected"]
    med = df["resid"].median()
    mad = np.median(np.abs(df["resid"] - med))
    thr = 3 * 1.4826 * mad
    f, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(df["date"], df["resid"], lw=0.8, color="#2c3e50")
    ax.axhline(thr, color="#c0392b", ls="--", lw=1)
    ax.axhline(-thr, color="#c0392b", ls="--", lw=1)
    flagged = df[np.abs(df["resid"]) > thr]
    ax.scatter(flagged["date"], flagged["resid"], color="#c0392b", s=18, zorder=3)
    ax.set(title="Seasonal residual with 3-MAD thresholds", ylabel="Residual ($)")
    f.autofmt_xdate(); f.tight_layout()
    f.savefig(IMAGES / "ch17_seasonal_residual.png", dpi=130); plt.close(f)

    # 2. Shewhart and CUSUM
    z = (df["revenue"] - df["revenue"].rolling(28).mean()) / df["revenue"].rolling(28).std()
    cz = np.maximum.accumulate(np.maximum(0, z.fillna(0) - 0.5))
    f, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    axes[0].plot(df["date"], z, lw=0.8)
    axes[0].axhline(3, color="#c0392b", ls="--", lw=1)
    axes[0].set(ylabel="z-score", title="Shewhart chart")
    axes[1].plot(df["date"], cz, lw=1, color="#8e44ad")
    axes[1].axhline(5, color="#c0392b", ls="--", lw=1)
    axes[1].set(ylabel="CUSUM", title="CUSUM chart")
    f.autofmt_xdate(); f.tight_layout()
    f.savefig(IMAGES / "ch17_control.png", dpi=130); plt.close(f)

    # 3. change point via rolling means
    roll = df["revenue"].rolling(30).mean()
    f, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(df["date"], df["revenue"], lw=0.6, alpha=0.5, label="daily")
    ax.plot(df["date"], roll, lw=2, color="#c0392b", label="30-day mean")
    ax.axvline("2025-01-22", color="k", ls="--", lw=1)
    ax.set(title="Level shift detected by a rolling mean", ylabel="Revenue ($)")
    ax.legend(fontsize=8)
    f.autofmt_xdate(); f.tight_layout()
    f.savefig(IMAGES / "ch17_change_point.png", dpi=130); plt.close(f)

    # 4. multivariate scores
    X = o[ORDER_FEATURES]
    Xs = StandardScaler().fit_transform(X)
    cov = EmpiricalCovariance().fit(Xs)
    md = cov.mahalanobis(Xs)
    iso = IsolationForest(n_estimators=150, contamination=0.015,
                          random_state=SEED).fit(Xs)
    s_iso = -iso.score_samples(Xs)
    lof = LocalOutlierFactor(n_neighbors=25, contamination=0.015, novelty=True)
    lof.fit(Xs)
    s_lof = -lof.score_samples(Xs)
    scores = {"Mahalanobis": md, "IsolationForest": s_iso, "LOF": s_lof}
    f, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (name, s) in zip(axes, scores.items()):
        ax.hist(s, bins=60, color="#2980b9", alpha=0.8)
        cutoff = np.quantile(s, 0.985)
        ax.axvline(cutoff, color="#c0392b", ls="--", lw=1)
        ax.set(title=name, xlabel="anomaly score")
    f.tight_layout()
    f.savefig(IMAGES / "ch17_multivariate.png", dpi=130); plt.close(f)

    # 5. precision at k
    y = o["is_fraud"].to_numpy()
    f, ax = plt.subplots(figsize=(5.8, 4.2))
    for name, s in scores.items():
        order = np.argsort(-s)
        ks = np.arange(50, 801, 50)
        prec = [y[order[:k]].mean() for k in ks]
        ax.plot(ks, prec, lw=2, label=name)
    ax.axhline(y.mean(), color="k", ls="--", lw=1,
               label=f"base rate {y.mean():.3f}")
    ax.set(xlabel="Cases reviewed", ylabel="Precision", title="Precision at k")
    ax.legend(fontsize=8)
    f.tight_layout(); f.savefig(IMAGES / "ch17_precision.png", dpi=130); plt.close(f)

    # 6. saturation curves and allocation
    f, ax = plt.subplots(figsize=(6.2, 4.2))
    for ch, grp in b.groupby("channel"):
        ax.plot(grp["spend"], grp["revenue"], lw=2, label=ch)
    ax.set(xlabel="Spend ($)", ylabel="Revenue ($)",
           title="Concave channel response curves")
    ax.legend(fontsize=8)
    f.tight_layout(); f.savefig(IMAGES / "ch17_saturation.png", dpi=130); plt.close(f)

    # 7. Monte Carlo
    rng = np.random.default_rng(SEED + 3)
    draws = rng.normal(6_000_000, 900_000, 10000)
    f, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(draws, bins=60, color="#2980b9", alpha=0.8)
    lo, hi = np.percentile(draws, [10, 90])
    ax.axvspan(lo, hi, color="#f1c40f", alpha=0.25)
    ax.axvline(lo, color="#c0392b", ls="--", lw=1)
    ax.axvline(hi, color="#c0392b", ls="--", lw=1)
    ax.set(title="Monte Carlo: 10th–90th percentile", xlabel="Annual profit ($)")
    f.tight_layout(); f.savefig(IMAGES / "ch17_montecarlo.png", dpi=130); plt.close(f)

    # 8. tornado chart
    effects = {"conversion rate": 1.42, "retention uplift": 0.96,
               "average order value": 0.74, "acquisition cost": 0.51,
               "implementation cost": 0.18}
    f, ax = plt.subplots(figsize=(6.4, 4.0))
    names = list(effects)
    vals = [effects[n] for n in names]
    ax.barh(names, vals, color="#8e44ad", alpha=0.8)
    ax.set(xlabel="Change in annual profit ($m)",
           title="Tornado: five largest assumptions")
    f.tight_layout(); f.savefig(IMAGES / "ch17_tornado.png", dpi=130); plt.close(f)

    # 9. decision options
    f, ax = plt.subplots(figsize=(6.0, 4.0))
    opts = ["Do nothing", "Small pilot", "Full roll-out"]
    ev = [3.1, 5.4, 6.0]
    down = [0.0, -0.8, -3.6]
    x = np.arange(len(opts))
    ax.bar(x - 0.18, ev, 0.36, label="expected value", color="#2980b9")
    ax.bar(x + 0.18, down, 0.36, label="10th percentile", color="#c0392b")
    ax.set(xticks=x, xticklabels=opts, ylabel="$m",
           title="Expected value and downside by option")
    ax.legend(fontsize=8)
    f.tight_layout(); f.savefig(IMAGES / "ch17_decision.png", dpi=130); plt.close(f)


def main() -> None:
    ANOM.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    print("generating Chapter 17 anomaly fixtures ...")
    d = daily()
    o = orders()
    b = budget()
    a = assumptions()
    print("drawing figures ...")
    figures(d, o, b, a)
    for p in sorted(IMAGES.glob("ch17_*.png")):
        print(f"  {p.name}: {p.stat().st_size:,} bytes")
    print("done.")


if __name__ == "__main__":
    main()
