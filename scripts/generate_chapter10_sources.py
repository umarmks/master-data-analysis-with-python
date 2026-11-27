#!/usr/bin/env python3
"""Generate the regression fixtures and figures that Chapter 10 uses.

Run from the project root:

    python scripts/generate_chapter10_sources.py

Outputs (all under data/raw/audit/ except the figures):

    ad_spend.csv             365 daily marketing observations
    housing_listings.csv     1,200 listings with correlated predictors
    churn_regression.csv     3,000 customers for logistic regression
    correlation_cases.csv    six constructed relationships
    images/ch10_*.png        seven figures used in the chapter

Everything is deterministic, including the figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "raw" / "audit"
IMAGES = ROOT / "images"
SEED = 1007


def ad_spend() -> pd.DataFrame:
    """Daily spend and revenue with a real response and seasonal confounder."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    spend = np.round(rng.uniform(500, 9000, size=365), 2)
    weekend = dates.dayofweek.isin([5, 6]).astype(int)
    season = 1 + 0.25 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365)
    revenue = np.round(
        4200 + 2.35 * spend * season + 1800 * weekend
        + rng.normal(0, 2600, size=365), 2)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "ad_spend": spend,
        "revenue": revenue,
        "weekend": weekend,
        "season_index": np.round(season, 3),
        "channel": rng.choice(["search", "social", "display"], size=365,
                              p=[0.5, 0.3, 0.2]),
    })
    df.to_csv(AUDIT / "ad_spend.csv", index=False)
    print(f"  ad_spend.csv: {len(df)} days")
    return df


def housing() -> pd.DataFrame:
    """Listings where size and rooms are deliberately correlated."""
    rng = np.random.default_rng(SEED)
    n = 1200
    rooms = rng.integers(1, 6, size=n)
    size = np.round(rooms * 320 + rng.normal(180, 70, size=n), 1).clip(250)
    location = rng.choice(["A", "B", "C"], size=n, p=[0.45, 0.35, 0.20])
    rating = np.round(rng.uniform(3.0, 5.0, size=n), 1)
    amenities = rng.integers(0, 6, size=n)
    loc_effect = np.select([location == "A", location == "B"],
                           [0.0, 28000.0], default=65000.0)
    price = np.round(
        45000 + 95 * size + 6200 * rooms + loc_effect
        + 9000 * rating + 4200 * amenities
        + 12000 * ((location == "C") & (size > 1800))
        + rng.normal(0, 22000, size=n), 2)
    df = pd.DataFrame({
        "listing_id": [f"L-{i + 1:05d}" for i in range(n)],
        "size_sqft": size,
        "rooms": rooms,
        "location": location,
        "rating": rating,
        "amenities": amenities,
        "price": price,
    })
    df.to_csv(AUDIT / "housing_listings.csv", index=False)
    print(f"  housing_listings.csv: {len(df)} listings")
    return df


def churn() -> pd.DataFrame:
    """Customer churn with a known logistic generating process."""
    rng = np.random.default_rng(SEED)
    n = 3000
    tenure = rng.integers(1, 73, size=n)
    monthly = np.round(rng.normal(65, 18, size=n).clip(20), 2)
    tickets = rng.poisson(0.9, size=n)
    satisfaction = np.clip(5 - np.minimum(tickets, 4)
                           - rng.binomial(1, 0.2, size=n), 1, 5)
    income = np.round(rng.lognormal(11.0, 0.4, size=n), 0)
    contract = rng.choice(["monthly", "annual"], size=n, p=[0.6, 0.4])
    lin = (-2.6 - 0.045 * tenure + 0.018 * monthly
           + 0.65 * tickets - 0.55 * (satisfaction - 3)
           + np.where(contract == "annual", -0.8, 0.0))
    churn_flag = rng.binomial(1, 1 / (1 + np.exp(-lin)))
    df = pd.DataFrame({
        "customer_id": [f"R-{i + 1:05d}" for i in range(n)],
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "support_tickets": tickets,
        "satisfaction": satisfaction,
        "income": income,
        "contract": contract,
        "churn": churn_flag,
    })
    df.to_csv(AUDIT / "churn_regression.csv", index=False)
    print(f"  churn_regression.csv: {len(df)} customers, "
          f"churn rate {churn_flag.mean():.3f}")
    return df


def correlation_cases() -> pd.DataFrame:
    """Six relationships that share a story-breaking feature."""
    rng = np.random.default_rng(SEED)
    n = 200
    x = np.linspace(0, 10, n)
    cases = {}
    cases["linear"] = 1.5 * x + rng.normal(0, 2, n)
    cases["curved"] = 0.6 * (x - 5) ** 2 + rng.normal(0, 2, n)
    cases["noisy"] = rng.normal(0, 5, n)
    outlier = cases["linear"].copy()
    outlier[-1] += 90
    cases["outlier"] = outlier
    cases["trend_a"] = 2.0 * x + rng.normal(0, 2, n)
    cases["trend_b"] = 3.0 * x + rng.normal(0, 2, n)
    cases["part_whole"] = x + cases["linear"]
    out = pd.DataFrame({"x": x, **cases})
    out.to_csv(AUDIT / "correlation_cases.csv", index=False)
    print(f"  correlation_cases.csv: {len(out)} rows, "
          f"{len(out.columns)} columns")
    return out


def figures(ad: pd.DataFrame, hous: pd.DataFrame, churn_df: pd.DataFrame,
            corr: pd.DataFrame) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)

    # 1. simple regression scatter with fitted line
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(ad["ad_spend"], ad["revenue"], s=12, alpha=0.5, color="#4477aa")
    b = np.polyfit(ad["ad_spend"], ad["revenue"], 1)
    xs = np.array([ad["ad_spend"].min(), ad["ad_spend"].max()])
    ax.plot(xs, b[0] * xs + b[1], color="#cc6677")
    ax.set_xlabel("ad_spend")
    ax.set_ylabel("revenue")
    ax.set_title("Revenue versus ad spend with OLS line")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch10_simple_regression.png", dpi=110)
    plt.close(fig)

    # 2. four diagnostic panels
    X = np.column_stack([np.ones(len(ad)), ad["ad_spend"], ad["weekend"]])
    beta = np.linalg.lstsq(X, ad["revenue"].to_numpy(), rcond=None)[0]
    resid = ad["revenue"].to_numpy() - X @ beta
    fitted = X @ beta
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes[0, 0].scatter(fitted, resid, s=10, alpha=0.5)
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 0].set_title("Residuals vs fitted")
    axes[0, 1].hist(resid, bins=40, color="#4477aa")
    axes[0, 1].set_title("Residual distribution")
    axes[1, 0].scatter(fitted, np.abs(resid), s=10, alpha=0.5)
    axes[1, 0].set_title("Scale-location")
    axes[1, 1].plot(ad["date"], resid, lw=0.6)
    axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[1, 1].set_title("Residuals over time")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch10_diagnostics.png", dpi=110)
    plt.close(fig)

    # 3. correlation heatmap ordered by a simple clustering proxy
    num = corr.drop(columns=["x"]).corr()
    order = sorted(num.columns, key=lambda c: -num[c].abs().mean())
    num = num.loc[order, order]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(num, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticklabels(order, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Correlation matrix, ordered by mean |r|")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch10_correlation_heatmap.png", dpi=110)
    plt.close(fig)

    # 4. interaction / simple slopes
    fig, ax = plt.subplots(figsize=(7, 4))
    for loc, color in [("A", "#4477aa"), ("C", "#cc6677")]:
        sub = hous[hous["location"] == loc]
        b = np.polyfit(sub["size_sqft"], sub["price"], 1)
        xs = np.array([sub["size_sqft"].min(), sub["size_sqft"].max()])
        ax.scatter(sub["size_sqft"], sub["price"], s=8, alpha=0.25, color=color)
        ax.plot(xs, b[0] * xs + b[1], color=color, lw=2, label=f"location {loc}")
    ax.legend()
    ax.set_xlabel("size_sqft")
    ax.set_ylabel("price")
    ax.set_title("Simple slopes by location")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch10_interaction.png", dpi=110)
    plt.close(fig)

    # 5. ROC and calibration for the logistic model
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, roc_auc_score
    feats = ["tenure_months", "monthly_charges", "support_tickets", "satisfaction"]
    Xc = churn_df[feats]
    yc = churn_df["churn"]
    clf = LogisticRegression(max_iter=1000).fit(Xc, yc)
    prob = clf.predict_proba(Xc)[:, 1]
    fpr, tpr, _ = roc_curve(yc, prob)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(fpr, tpr, color="#4477aa")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title(f"ROC, AUC = {roc_auc_score(yc, prob):.3f}")
    bins = np.linspace(0, 1, 11)
    obs = [yc[(prob >= bins[i]) & (prob < bins[i + 1])].mean()
           if ((prob >= bins[i]) & (prob < bins[i + 1])).sum() else np.nan
           for i in range(10)]
    exp = [(bins[i] + bins[i + 1]) / 2 for i in range(10)]
    axes[1].plot(exp, obs, "o-", color="#cc6677")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_title("Calibration by decile")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch10_logit_roc.png", dpi=110)
    plt.close(fig)

    # 6. four correlation shapes
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, key in zip(axes.ravel(), ["linear", "curved", "outlier", "noisy"]):
        ax.scatter(corr["x"], corr[key], s=10, alpha=0.6)
        ax.set_title(f"{key}: r = {corr['x'].corr(corr[key]):.3f}")
    fig.suptitle("Four relationships, different stories")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch10_correlation_shapes.png", dpi=110)
    plt.close(fig)

    # 7. ridge / lasso coefficient paths (toy)
    from sklearn.linear_model import Lasso
    Xh = pd.get_dummies(hous[["size_sqft", "rooms", "rating", "amenities",
                              "location"]], drop_first=True).astype(float)
    yh = hous["price"]
    alphas = np.logspace(2, 6, 30)
    paths = []
    for a in alphas:
        m = Lasso(alpha=a, max_iter=5000).fit(Xh, yh)
        paths.append(m.coef_)
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, name in enumerate(Xh.columns):
        ax.plot(alphas, [p[j] for p in paths], label=name)
    ax.set_xscale("log")
    ax.legend(fontsize=7)
    ax.set_title("Lasso coefficient paths")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch10_lasso_paths.png", dpi=110)
    plt.close(fig)

    print(f"  images: {len(list(IMAGES.glob('ch10_*.png')))} figures")


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    print("generating Chapter 10 regression fixtures and figures ...")
    ad = ad_spend()
    hous = housing()
    churn_df = churn()
    corr = correlation_cases()
    figures(ad, hous, churn_df, corr)
    print("done.")


if __name__ == "__main__":
    main()
