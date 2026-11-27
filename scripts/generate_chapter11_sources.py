#!/usr/bin/env python3
"""Generate the visualisation fixtures and figures that Chapter 11 uses.

Run from the project root:

    python scripts/generate_chapter11_sources.py

Outputs (all under data/raw/audit/ except the figures):

    viz_sales.csv          24 months x 4 regions x 3 categories
    viz_distributions.csv  five distributions for chart comparisons
    viz_cohort.csv         an 8x8 retention triangle
    viz_bad_chart.csv      data for before/after chart makeovers
    images/ch11_*.png      twelve figures used in the chapter

Every figure is drawn with Matplotlib, which is installed. The Seaborn and
Plotly sections of the chapter show the equivalent APIs as optional, guarded
examples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "raw" / "audit"
IMAGES = ROOT / "images"
SEED = 1107


def sales() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    regions = ["Midwest", "Northeast", "South", "West"]
    cats = ["Camping", "Footwear", "Apparel"]
    rows = []
    for m in months:
        for reg in regions:
            for cat in cats:
                base = {"Camping": 42000, "Footwear": 38000, "Apparel": 31000}[cat]
                trend = 1 + 0.008 * (m.year - 2024) * 12 + 0.008 * m.month
                season = 1 + 0.16 * np.sin(2 * np.pi * (m.month - 3) / 12)
                rev = base * trend * season * float(rng.normal(1, 0.06))
                orders = int(rev / float(rng.uniform(160, 220)))
                margin = rev * float(rng.uniform(0.18, 0.31))
                rows.append((m.strftime("%Y-%m"), reg, cat,
                             round(rev, 2), orders, round(margin, 2)))
    df = pd.DataFrame(rows, columns=["month", "region", "category",
                                     "revenue", "orders", "margin"])
    df.to_csv(AUDIT / "viz_sales.csv", index=False)
    print(f"  viz_sales.csv: {len(df)} rows")
    return df


def distributions() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = 1500
    df = pd.DataFrame({
        "normal": rng.normal(50, 10, n),
        "skewed": rng.lognormal(3.5, 0.7, n),
        "bimodal": np.where(rng.random(n) < 0.5,
                            rng.normal(20, 4, n), rng.normal(70, 6, n)),
        "uniform": rng.uniform(0, 100, n),
        "discrete": rng.poisson(5, n),
    })
    df.to_csv(AUDIT / "viz_distributions.csv", index=False)
    print(f"  viz_distributions.csv: {len(df)} rows")
    return df


def cohort() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(8):
        for j in range(i + 1):
            base = 100.0 * (0.72 ** j) * float(rng.normal(1, 0.04))
            rows.append((f"2025-{i+1:02d}", j, round(base, 1)))
    df = pd.DataFrame(rows, columns=["cohort", "month_index", "retention_pct"])
    df.to_csv(AUDIT / "viz_cohort.csv", index=False)
    print(f"  viz_cohort.csv: {len(df)} cells")
    return df


def bad_chart() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    cats = [f"Segment {chr(65+i)}" for i in range(12)]
    df = pd.DataFrame({
        "category": cats,
        "value": np.round(rng.uniform(80, 140, size=12), 1),
        "orders": rng.integers(50, 500, size=12),
        "month": rng.integers(1, 13, size=12),
    })
    df.to_csv(AUDIT / "viz_bad_chart.csv", index=False)
    print(f"  viz_bad_chart.csv: {len(df)} rows")
    return df


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.25, "grid.color": "#cccccc",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
        "figure.dpi": 110,
    })


def figures(s: pd.DataFrame, dist: pd.DataFrame, co: pd.DataFrame,
            bad: pd.DataFrame) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    style()

    monthly = s.groupby("month", as_index=False).agg(
        revenue=("revenue", "sum"), orders=("orders", "sum"))
    monthly["aov"] = (monthly["revenue"] / monthly["orders"]).round(2)

    # 1. four-panel quarterly figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(monthly["month"], monthly["revenue"], color="#4477aa")
    axes[0, 0].set_title("Revenue trend")
    axes[1, 0].hist(dist["normal"], bins=40, color="#4477aa", alpha=0.8)
    axes[1, 0].set_title("Order value distribution")
    comp = s.groupby("category")["revenue"].sum()
    axes[0, 1].bar(comp.index, comp.values,
                   color=["#4477aa", "#ee7733", "#228833"])
    axes[0, 1].set_title("Revenue by category")
    reg = s.groupby("region")["revenue"].mean()
    axes[1, 1].barh(reg.index, reg.values, color="#4477aa")
    axes[1, 1].set_title("Average revenue by region")
    for ax in axes.ravel():
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Quarterly performance: four views, one style")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch11_four_panel.png")
    plt.close(fig)

    # 2. truncated vs honest axis
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    q = monthly["revenue"].iloc[-6:].to_numpy()
    axes[0].plot(range(6), q, "o-", color="#cc6677")
    axes[0].set_ylim(q.min() - 400, q.max() + 400)
    axes[0].set_title("Truncated: looks dramatic")
    axes[1].plot(range(6), q, "o-", color="#4477aa")
    axes[1].set_ylim(0, q.max() * 1.1)
    axes[1].set_title("Zero baseline: honest scale")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch11_axis_compare.png")
    plt.close(fig)

    # 3. grouped bars with labels and error bars
    piv = s[s["month"] >= "2025-10"].groupby(["region", "category"])[
        "revenue"].agg(["mean", "std", "size"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    xpos = np.arange(len(piv["region"].unique()))
    width = 0.25
    for i, cat in enumerate(sorted(piv["category"].unique())):
        sub = piv[piv["category"] == cat].set_index("region")
        vals = sub.loc[sorted(piv["region"].unique()), "mean"].to_numpy()
        errs = (sub.loc[sorted(piv["region"].unique()), "std"].to_numpy()
                / np.sqrt(sub.loc[sorted(piv["region"].unique()), "size"].to_numpy()))
        bars = ax.bar(xpos + (i - 1) * width, vals, width, label=cat)
        ax.errorbar(xpos + (i - 1) * width, vals, yerr=errs,
                    fmt="none", ecolor="black", capsize=3)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{b.get_height()/1000:.0f}k", ha="center", va="bottom",
                    fontsize=7)
    ax.set_xticks(xpos)
    ax.set_xticklabels(sorted(piv["region"].unique()))
    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v/1000:.0f}k"))
    ax.set_title("Revenue by region and category, with error bars")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch11_grouped_bars.png")
    plt.close(fig)

    # 4. small multiples: 12 monthly distributions
    fig, axes = plt.subplots(3, 4, figsize=(12, 7), sharex=True, sharey=True)
    for ax, (m, grp) in zip(axes.ravel(), list(s.groupby("month"))[:12]):
        ax.hist(grp["revenue"], bins=15, color="#4477aa", alpha=0.8)
        ax.set_title(m, fontsize=8)
    fig.suptitle("Monthly revenue distributions, shared axes")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch11_small_multiples.png")
    plt.close(fig)

    # 5. annotated line with event band
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly["month"], monthly["revenue"], color="#4477aa")
    ax.axvspan(7.5, 9.5, color="#eeeeee", zorder=0)
    ax.text(8.5, monthly["revenue"].max(), " campaign", fontsize=8)
    peak = int(monthly["revenue"].idxmax())
    ax.annotate(f"peak {monthly['revenue'][peak]:,.0f}",
                xy=(peak, monthly["revenue"][peak]),
                xytext=(peak - 3, monthly["revenue"][peak] * 1.05),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_title("Revenue with event band and annotated peak")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(IMAGES / "ch11_annotated_line.png")
    plt.close(fig)

    # 6. box, violin, strip for the same data
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    data = [dist.loc[dist["discrete"] == k, "normal"].to_numpy()
            for k in range(4)]
    axes[0].boxplot(data, tick_labels=range(4))
    axes[1].violinplot(data, showmedians=True)
    for k, vals in enumerate(data):
        axes[2].scatter(np.full(len(vals), k), vals, s=3, alpha=0.35)
    for ax, title in zip(axes, ["Box", "Violin", "Strip"]):
        ax.set_title(f"{title} plot")
    fig.suptitle("Three views of the same groups")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch11_box_violin_strip.png")
    plt.close(fig)

    # 7. lower-triangle correlation heatmap
    corr = dist.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if not mask[i, j]:
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Lower-triangle correlation heatmap")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch11_heatmap.png")
    plt.close(fig)

    # 8. bad pie with 12 slices
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(bad["value"], labels=bad["category"], startangle=90)
    ax.set_title("Twelve slices: hard to compare")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch11_bad_pie.png")
    plt.close(fig)

    # 9. bubble area scaling: bad radius vs good area
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sizes = bad["orders"].to_numpy()
    for ax, scale, title in [
        (axes[0], sizes, "Radius ∝ value (misleading)"),
        (axes[1], sizes * 8, "Area ∝ value (correct)"),
    ]:
        ax.scatter(bad["value"], bad["month"], s=scale, alpha=0.5,
                   color="#4477aa")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(IMAGES / "ch11_bubble_scaling.png")
    plt.close(fig)

    # 10. cohort heatmap with annotated percentages
    piv = co.pivot(index="cohort", columns="month_index",
                   values="retention_pct")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(piv, cmap="YlGnBu", vmin=0, vmax=100)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"M{j}" for j in range(piv.shape[1])])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Retention cohort triangle")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch11_cohort.png")
    plt.close(fig)

    # 11. before/after bad chart makeover
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].pie(bad["value"][:5], labels=bad["category"][:5])
    axes[0].set_title("Before: pie, no units")
    axes[1].barh(bad["category"][:5], bad["value"][:5], color="#4477aa")
    axes[1].invert_yaxis()
    axes[1].set_title("After: labelled bars, USD thousands")
    axes[1].set_xlabel("Revenue (USD thousands)")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch11_makeover.png")
    plt.close(fig)

    # 12. final story chart
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly["month"], monthly["aov"], color="#4477aa", lw=2)
    ax.axhline(monthly["aov"].mean(), color="grey", linestyle="--", lw=1)
    ax.text(1, monthly["aov"].mean(), " 24-month average", fontsize=8,
            color="grey")
    ax.set_title("Average order value held near $190 despite revenue growth")
    ax.set_ylabel("AOV (USD)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(IMAGES / "ch11_story.png")
    plt.close(fig)

    print(f"  images: {len(list(IMAGES.glob('ch11_*.png')))} figures")


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    print("generating Chapter 11 visualisation fixtures and figures ...")
    s = sales()
    d = distributions()
    c = cohort()
    b = bad_chart()
    figures(s, d, c, b)
    print("done.")


if __name__ == "__main__":
    main()
