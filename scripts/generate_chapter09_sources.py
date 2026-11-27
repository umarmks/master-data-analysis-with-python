#!/usr/bin/env python3
"""Generate the inference fixtures and figures that Chapter 9 uses.

Run from the project root:

    python scripts/generate_chapter09_sources.py

Outputs (all under data/raw/audit/ except the figures):

    ab_checkout.csv          12,000 users in a two-arm checkout experiment
    order_values_ch09.csv    5,000 right-skewed order values
    sampling_frame.csv       50,000 customers with a known population mean
    paired_campaign.csv      800 customers with before/after spend
    images/ch09_*.png        six figures used in the chapter

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
SEED = 909


def ab_test() -> pd.DataFrame:
    """A two-arm checkout test with a small, real conversion lift."""
    rng = np.random.default_rng(SEED)
    n = 12000
    variant = rng.choice(["control", "treatment"], size=n, p=[0.5, 0.5])
    device = rng.choice(["iOS", "Android", "Desktop"], size=n,
                        p=[0.40, 0.35, 0.25])
    region = rng.choice(["Midwest", "Northeast", "South", "West"], size=n)
    base = 0.118
    p = base + np.where(variant == "treatment", 0.013, 0.0)
    p = p + np.where(device == "Desktop", 0.012, 0.0)
    converted = rng.binomial(1, p)
    revenue = np.round(np.where(converted == 1,
                                rng.lognormal(4.2, 0.55, size=n), 0.0), 2)
    support = rng.binomial(1, np.where(
        variant == "treatment", 0.031, 0.028))
    date = pd.to_datetime("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 14, size=n), unit="D")
    df = pd.DataFrame({
        "user_id": [f"U-{100000 + i}" for i in range(n)],
        "variant": variant, "device": device, "region": region,
        "converted": converted, "revenue": revenue,
        "support_contact": support,
        "exposure_date": date.strftime("%Y-%m-%d"),
    })
    df.to_csv(AUDIT / "ab_checkout.csv", index=False)
    print(f"  ab_checkout.csv: {len(df)} users, "
          f"conversion {df['converted'].mean():.4f}")
    return df


def order_values() -> pd.DataFrame:
    """Right-skewed money data for the CLT and bootstrap sections."""
    rng = np.random.default_rng(SEED)
    n = 5000
    v = np.round(rng.lognormal(4.1, 0.75, size=n), 2)
    extreme = rng.random(n) < 0.01
    v[extreme] = np.round(v[extreme] * 12, 2)
    df = pd.DataFrame({"order_id": [f"O-{i + 1:05d}" for i in range(n)],
                       "order_value": v})
    df.to_csv(AUDIT / "order_values_ch09.csv", index=False)
    print(f"  order_values_ch09.csv: {len(df)} values, "
          f"mean {v.mean():.2f}, median {np.median(v):.2f}")
    return df


def sampling_frame() -> pd.DataFrame:
    """A finite population with a known mean and two real biases."""
    rng = np.random.default_rng(SEED)
    n = 50000
    segment = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    centers = {"A": 40000.0, "B": 70000.0, "C": 110000.0}
    value = np.round(rng.normal([centers[s] for s in segment], 12000), 2)
    in_frame = rng.random(n) > 0.08
    propensity = 1.0 / (1.0 + np.exp(-(value - 70000) / 40000))
    responds = rng.random(n) < propensity
    df = pd.DataFrame({
        "customer_id": [f"C-{i + 1:06d}" for i in range(n)],
        "segment": segment,
        "value": value,
        "in_frame": in_frame.astype(int),
        "responds": responds.astype(int),
    })
    df.to_csv(AUDIT / "sampling_frame.csv", index=False)
    print(f"  sampling_frame.csv: {len(df)} customers, "
          f"population mean {value.mean():.2f}")
    return df


def paired_campaign() -> pd.DataFrame:
    """Before/after spend on the same customers, for paired tests."""
    rng = np.random.default_rng(SEED)
    n = 800
    before = np.round(rng.lognormal(4.2, 0.5, size=n), 2)
    after = np.round(before * rng.normal(1.06, 0.18, size=n)
                     + rng.normal(3, 5, size=n), 2)
    after = np.maximum(after, 0).round(2)
    df = pd.DataFrame({
        "customer_id": [f"P-{i + 1:05d}" for i in range(n)],
        "spend_before": before,
        "spend_after": after,
        "segment": rng.choice(["Basic", "Plus"], size=n, p=[0.6, 0.4]),
    })
    df.to_csv(AUDIT / "paired_campaign.csv", index=False)
    print(f"  paired_campaign.csv: {len(df)} pairs, "
          f"mean difference {(after - before).mean():.2f}")
    return df


def figures(ov: pd.DataFrame, frame: pd.DataFrame, ab: pd.DataFrame) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    v = ov["order_value"].to_numpy()

    # 1. sampling distribution at three n
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, n in zip(axes, [5, 30, 100]):
        means = np.mean(rng.choice(v, size=(4000, n), replace=True), axis=1)
        ax.hist(means, bins=40, color="#4477aa")
        ax.set_title(f"n = {n}")
    fig.suptitle("Sampling distribution of the mean")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch09_sampling_distribution.png", dpi=110)
    plt.close(fig)

    # 2. bootstrap distribution of the median
    boots = np.median(rng.choice(v, size=(4000, len(v)), replace=True), axis=1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(boots, bins=50, color="#aa6677")
    ax.axvline(np.median(v), color="black", linestyle="--")
    ax.set_title("Bootstrap distribution of the median")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch09_bootstrap_median.png", dpi=110)
    plt.close(fig)

    # 3. confidence interval coverage illustration
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(30):
        s = rng.choice(v, size=40, replace=True)
        lo, hi = np.quantile(s, [0.025, 0.975])
        ax.plot([lo, hi], [i, i], color="#4477aa" if lo <= np.median(v) <= hi else "#cc6677")
    ax.axvline(np.median(v), color="black", linestyle="--")
    ax.set_title("30 bootstrap percentile intervals (red = miss)")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch09_intervals.png", dpi=110)
    plt.close(fig)

    # 4. power curve for a two-proportion test
    from scipy import stats as sps
    p0 = 0.118
    lifts = np.linspace(0, 0.03, 60)
    power = []
    for lift in lifts:
        p1 = p0 + lift
        se = np.sqrt(p0 * (1 - p0) / 6000 + p1 * (1 - p1) / 6000)
        zcrit = sps.norm.ppf(0.975)
        power.append(sps.norm.cdf(lift / se - zcrit) + sps.norm.cdf(-lift / se - zcrit))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(lifts, power, color="#4477aa")
    ax.axhline(0.8, color="grey", linestyle="--")
    ax.set_xlabel("Absolute lift")
    ax.set_ylabel("Power")
    ax.set_title("Power at n = 6,000 per arm")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch09_power_curve.png", dpi=110)
    plt.close(fig)

    # 5. sampling design comparison
    truth = frame["value"].mean()
    rng2 = np.random.default_rng(SEED + 1)
    means = {
        "simple random": rng2.choice(frame["value"], size=500).mean(),
        "respondents only": frame.loc[frame["responds"] == 1, "value"].sample(
            500, random_state=SEED).mean(),
        "frame only": frame.loc[frame["in_frame"] == 1, "value"].sample(
            500, random_state=SEED).mean(),
    }
    fig, ax = plt.subplots(figsize=(7, 3.5))
    names = list(means)
    vals = list(means.values())
    ax.bar(names + ["truth"], vals + [truth],
           color=["#4477aa", "#cc6677", "#ee7733", "#333333"])
    ax.axhline(truth, color="black", linestyle="--")
    ax.set_title("Design bias: mean estimate by source")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch09_sampling_bias.png", dpi=110)
    plt.close(fig)

    # 6. A/B conversion with intervals
    rates = ab.groupby("variant")["converted"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(rates))
    err = [1.96 * np.sqrt(p * (1 - p) / n)
           for p, n in zip(rates["mean"], rates["size"])]
    ax.bar(x, rates["mean"], yerr=err, capsize=5,
           color=["#4477aa", "#ee7733"])
    ax.set_xticks(x)
    ax.set_xticklabels(rates.index)
    ax.set_title("Checkout conversion by arm (95% interval)")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch09_ab_conversion.png", dpi=110)
    plt.close(fig)

    print(f"  images: {len(list(IMAGES.glob('ch09_*.png')))} figures")


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    print("generating Chapter 9 inference fixtures and figures ...")
    ab = ab_test()
    ov = order_values()
    frame = sampling_frame()
    paired_campaign()
    figures(ov, frame, ab)
    print("done.")


if __name__ == "__main__":
    main()
