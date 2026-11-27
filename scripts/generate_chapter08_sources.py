#!/usr/bin/env python3
"""Generate the EDA fixtures and figures that Chapter 8 uses.

Run from the project root:

    python scripts/generate_chapter08_sources.py

Outputs:

    data/raw/audit/churn_eda.csv          3,000 customers, 34 mixed-type columns
    data/raw/audit/anscombe_quartet.csv   four datasets with identical summaries
    data/raw/audit/simpson_segments.csv   aggregated two-segment reversal
    data/raw/audit/distribution_gallery.csv  six shapes for the shape section
    images/ch08_*.png                     nine figures used in the chapter

Everything is deterministic, including the figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "raw" / "audit"
IMAGES = ROOT / "images"

SEED = 808


def churn_eda() -> pd.DataFrame:
    """A 34-column customer table with planted relationships and defects."""
    rng = np.random.default_rng(SEED)
    n = 3000

    segment = rng.choice(["Basic", "Plus", "Premium"], size=n, p=[0.45, 0.35, 0.20])
    region = rng.choice(["Midwest", "Northeast", "South", "West"], size=n)
    channel = rng.choice(["Web", "App", "Phone"], size=n, p=[0.55, 0.30, 0.15])
    plan = rng.choice(["Monthly", "Annual"], size=n, p=[0.60, 0.40])
    loyalty = rng.choice(["Bronze", "Silver", "Gold", "Platinum"], size=n)
    device = rng.choice(["iOS", "Android", "Desktop"], size=n)
    zip_code = [f"{int(rng.integers(10000, 99999))}" for _ in range(n)]

    centers = {"Basic": 45000.0, "Plus": 68000.0, "Premium": 105000.0}
    income = np.exp(rng.normal(np.log([centers[s] for s in segment]), 0.35))
    income = np.round(income / 500) * 500
    p_miss = {"Basic": 0.16, "Plus": 0.10, "Premium": 0.05}
    income = income.astype(object)
    income[rng.random(n) < np.array([p_miss[s] for s in segment])] = np.nan

    age = rng.integers(18, 76, size=n)
    tenure = rng.integers(1, 73, size=n)
    monthly_charges = np.round(
        np.where(plan == "Annual", 42.0, 55.0)
        + np.where(segment == "Premium", 18.0, np.where(segment == "Plus", 7.0, 0.0))
        + rng.normal(0, 6, size=n), 2)

    support_tickets = rng.poisson(0.9, size=n)
    support_tickets[rng.random(n) < 0.45] = 0
    logins = rng.poisson(8, size=n)
    session_minutes = np.round(np.where(
        rng.random(n) < 0.55,
        rng.normal(9, 2.5, size=n),
        rng.normal(46, 11, size=n)), 1).clip(1)
    data_gb = np.round(rng.gamma(3.0, 2.0, size=n), 2)
    discount = rng.choice([0, 0, 0, 10, 15, 20, 25], size=n)
    num_products = rng.poisson(2.0, size=n) + 1
    order_count = rng.poisson(1 + tenure / 12.0).astype(int)
    avg_order_value = np.round(np.exp(rng.normal(4.2, 0.45, size=n)), 2)
    total_spend = np.round(order_count * avg_order_value * (1 - discount / 100), 2)
    big = rng.random(n) < 0.012
    total_spend[big] = np.round(total_spend[big] * rng.uniform(6, 14, size=big.sum()), 2)
    refunds = rng.binomial(order_count, 0.07)
    nps = rng.integers(0, 11, size=n)
    satisfaction = np.clip(5 - np.minimum(support_tickets, 4) - rng.binomial(1, 0.2, size=n), 1, 5)
    latency = np.round(np.exp(rng.normal(4.6, 0.5, size=n)), 1)

    lin = (1.1
           - 0.8 * (tenure - 36) / 36.0
           + 0.55 * np.minimum(support_tickets, 5)
           - 0.45 * (satisfaction - 3)
           + np.where(segment == "Basic", 0.45, np.where(segment == "Plus", 0.0, -0.3)))
    churn = rng.binomial(1, 1.0 / (1.0 + np.exp(-lin)))

    signup = pd.to_datetime("2026-03-31") - pd.to_timedelta(tenure, unit="D") * 30
    last_active = signup + pd.to_timedelta(
        rng.integers(0, np.maximum(tenure * 30, 1)), unit="D")
    reviews = [
        "fast delivery", "great value", "support was slow", "would recommend",
        "billing issue", "love the app", "package arrived damaged", "easy returns",
    ]
    review_text = rng.choice(reviews, size=n)

    df = pd.DataFrame({
        "customer_id": [f"C-{200000 + i}" for i in range(n)],
        "signup_date": signup.strftime("%Y-%m-%d"),
        "last_active_date": last_active.strftime("%Y-%m-%d"),
        "segment": segment,
        "region": region,
        "channel": channel,
        "plan": plan,
        "loyalty_tier": loyalty,
        "device": device,
        "zip_code": zip_code,
        "review_text": review_text,
        "income": income,
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "support_tickets": support_tickets,
        "logins_30d": logins,
        "session_minutes": session_minutes,
        "data_gb": data_gb,
        "discount_pct": discount,
        "num_products": num_products,
        "order_count": order_count,
        "avg_order_value": avg_order_value,
        "total_spend": total_spend,
        "refunds": refunds,
        "nps": nps,
        "satisfaction": satisfaction,
        "latency_ms": latency,
        "churn": churn,
        "is_promoter": (nps >= 9).astype(int),
        "paperless": rng.binomial(1, 0.7, size=n),
        "auto_pay": rng.binomial(1, 0.55, size=n),
        "has_dependents": rng.binomial(1, 0.35, size=n),
        "is_referral": rng.binomial(1, 0.12, size=n),
    })
    assert len(df.columns) == 34, len(df.columns)
    df.to_csv(AUDIT / "churn_eda.csv", index=False)
    print(f"  churn_eda.csv: {len(df)} rows, {len(df.columns)} columns")
    return df


def anscombe() -> pd.DataFrame:
    """The classic four datasets with (nearly) identical linear summaries."""
    x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
    df = pd.DataFrame({"x1": x, "y1": y1, "x2": x, "y2": y2,
                       "x3": x, "y3": y3, "x4": x4, "y4": y4})
    df.to_csv(AUDIT / "anscombe_quartet.csv", index=False)
    print(f"  anscombe_quartet.csv: {len(df)} rows")
    return df


def simpson() -> pd.DataFrame:
    """Treatment looks worse overall but better inside both segments."""
    df = pd.DataFrame([
        ("A_easy", "treatment", 81, 90),
        ("A_easy", "control", 234, 270),
        ("B_hard", "treatment", 192, 270),
        ("B_hard", "control", 55, 90),
    ], columns=["segment", "group", "successes", "trials"])
    df.to_csv(AUDIT / "simpson_segments.csv", index=False)
    print(f"  simpson_segments.csv: {len(df)} rows")
    return df


def gallery() -> pd.DataFrame:
    """Six shapes, 2,000 observations each, for the distribution section."""
    rng = np.random.default_rng(SEED)
    n = 2000
    df = pd.DataFrame({
        "normal": rng.normal(50, 10, n),
        "uniform": rng.uniform(0, 100, n),
        "lognormal": rng.lognormal(3.5, 0.6, n),
        "bimodal": np.where(rng.random(n) < 0.5,
                            rng.normal(20, 4, n), rng.normal(70, 5, n)),
        "zero_inflated": np.where(rng.random(n) < 0.55, 0, rng.poisson(6, n)),
        "sparse_binary": rng.binomial(1, 0.03, n),
    }).round(3)
    df.to_csv(AUDIT / "distribution_gallery.csv", index=False)
    print(f"  distribution_gallery.csv: {len(df)} rows")
    return df


def figures(df: pd.DataFrame, ansc: pd.DataFrame, simp: pd.DataFrame,
            gal: pd.DataFrame) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    rev = df["total_spend"]

    # 1. histogram / ECDF / box plot of the same variable
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].hist(rev, bins=50, color="#4477aa")
    axes[0].set_title("Histogram")
    axes[1].ecdf(rev)
    axes[1].set_title("ECDF")
    x2 = np.sort(rev)
    axes[2].boxplot(x2, vert=False)
    axes[2].set_title("Box plot")
    fig.suptitle("Three views of total_spend")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch08_hist_ecdf_box.png", dpi=110)
    plt.close(fig)

    # 2. distribution gallery
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, col in zip(axes.ravel(), gal.columns):
        ax.hist(gal[col], bins=40, color="#aa6677")
        ax.set_title(col)
    fig.suptitle("Six shapes")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch08_distribution_gallery.png", dpi=110)
    plt.close(fig)

    # 3. correlation heatmap
    num = df.select_dtypes(include=[np.number]).drop(columns=["churn"])
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Correlation matrix (numeric columns)")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch08_correlation_heatmap.png", dpi=110)
    plt.close(fig)

    # 4. pair plot of six variables coloured by segment
    cols = ["tenure_months", "monthly_charges", "support_tickets",
            "logins_30d", "avg_order_value", "total_spend"]
    colors = {"Basic": "#4477aa", "Plus": "#ee7733", "Premium": "#228833"}
    fig, axes = plt.subplots(3, 3, figsize=(9, 8))
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if i == j:
                for seg, c in colors.items():
                    ax.hist(df.loc[df["segment"] == seg, cols[i]], bins=25,
                            alpha=0.6, color=c)
            else:
                for seg, c in colors.items():
                    sub = df[df["segment"] == seg]
                    ax.scatter(sub[cols[j]], sub[cols[i]], s=4, alpha=0.4, color=c)
            if j == 0:
                ax.set_ylabel(cols[i], fontsize=7)
            if i == 2:
                ax.set_xlabel(cols[j], fontsize=7, rotation=30)
    fig.suptitle("Pair plot by segment")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch08_pairplot_segment.png", dpi=110)
    plt.close(fig)

    # 5. group medians with bootstrap intervals
    groups = df["segment"].unique()
    meds, lows, highs = [], [], []
    rng = np.random.default_rng(SEED)
    for g in groups:
        vals = df.loc[df["segment"] == g, "total_spend"].to_numpy()
        boots = np.median(rng.choice(vals, size=(1000, len(vals)), replace=True), axis=1)
        meds.append(float(np.median(vals)))
        lows.append(float(np.quantile(boots, 0.025)))
        highs.append(float(np.quantile(boots, 0.975)))
    fig, ax = plt.subplots(figsize=(6, 4))
    y = np.arange(len(groups))
    ax.errorbar(meds, y, xerr=[np.array(meds) - lows, np.array(highs) - meds],
                fmt="o", capsize=4, color="#4477aa")
    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.set_xlabel("Median total_spend (95% bootstrap interval)")
    ax.set_title("Group medians with uncertainty")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch08_group_medians_ci.png", dpi=110)
    plt.close(fig)

    # 6. Simpson reversal, two panels
    s = simp.copy()
    s["rate"] = s["successes"] / s["trials"]
    agg = s.groupby("group")[["successes", "trials"]].sum()
    agg["rate"] = agg["successes"] / agg["trials"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(agg.index, agg["rate"], color=["#4477aa", "#cc6677"])
    axes[0].set_title("Aggregate: control wins")
    axes[0].set_ylim(0, 1)
    for seg in s["segment"].unique():
        sub = s[s["segment"] == seg]
        axes[1].bar([f"{seg}\n{g}" for g in sub["group"]], sub["rate"])
    axes[1].set_title("Within segment: treatment wins")
    axes[1].set_ylim(0, 1)
    fig.suptitle("Simpson's reversal")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch08_simpson.png", dpi=110)
    plt.close(fig)

    # 7. Anscombe quartet
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    for i, ax in enumerate(axes.ravel(), 1):
        ax.scatter(ansc[f"x{i}"], ansc[f"y{i}"], color="#4477aa")
        ax.set_title(f"Dataset {i}")
    fig.suptitle("Four datasets, identical summaries")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch08_anscombe.png", dpi=110)
    plt.close(fig)

    # 8. Q-Q before and after log transform
    from scipy import stats as sps
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    sps.probplot(df["logins_30d"], dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q: raw logins")
    sps.probplot(np.log1p(df["total_spend"]), dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q: log1p(total_spend)")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch08_qq.png", dpi=110)
    plt.close(fig)

    # 9. monthly trend by segment
    t = df.assign(month=pd.to_datetime(df["signup_date"]).dt.strftime("%Y-%m"))
    trend = t.groupby(["month", "segment"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    trend.plot(ax=ax)
    ax.set_title("Signups by month and segment")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch08_time_trend.png", dpi=110)
    plt.close(fig)

    print(f"  images: {len(list(IMAGES.glob('ch08_*.png')))} figures")


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    print("generating Chapter 8 EDA fixtures and figures ...")
    df = churn_eda()
    ansc = anscombe()
    simp = simpson()
    gal = gallery()
    figures(df, ansc, simp, gal)
    print("done.")


if __name__ == "__main__":
    main()
