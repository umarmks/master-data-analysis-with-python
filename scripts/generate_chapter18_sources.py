#!/usr/bin/env python3
"""Generate the reporting fixtures and figures Chapter 18 uses.

Run from the project root:

    python scripts/generate_chapter18_sources.py

Outputs (fixtures under data/raw/reporting/, figures under images/):

    rep_findings.csv     30 findings from a mock 30-page analysis
    rep_kpi.csv          12 KPIs with actual, target, and prior values
    rep_scenarios.csv    base/low/high scenarios for four assumptions
    rep_questions.csv    five stakeholder questions with answers
    rep_audiences.csv    four audience profiles

    images/ch18_*.png    eight figures used in the chapter

Everything is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "data" / "raw" / "reporting"
IMAGES = ROOT / "images"
SEED = 1807


def findings() -> pd.DataFrame:
    rows = [
        (1, "Executive summary", "Revenue fell 6.2% year over year", "revenue_yoy", -6.2, "%", "all", 0.98, 10, "executive"),
        (2, "Executive summary", "The decline is concentrated in two regions", "decline_share", 71.0, "%", "South,West", 0.92, 10, "executive"),
        (3, "Executive summary", "A targeted promotion can recover about $1.4m", "recovery", 1.4, "USDm", "South,West", 0.70, 10, "executive"),
        (4, "Data quality", "Three months of cost data are missing", "missing_months", 3, "count", "all", 1.00, 5, "technical"),
        (5, "Data quality", "Duplicate orders inflate revenue by 1.1%", "duplicate_inflation", 1.1, "%", "all", 1.00, 8, "technical"),
        (6, "Data quality", "Guest checkouts have no customer key", "guest_share", 4.6, "%", "all", 1.00, 4, "technical"),
        (7, "Method", "Net revenue subtracts returns and shipping", "definition", 1, "binary", "all", 1.00, 6, "technical"),
        (8, "Method", "The bridge uses a sequential substitution order", "bridge_order", 1, "binary", "all", 1.00, 3, "technical"),
        (9, "Revenue trend", "Monthly revenue peaked in November 2024", "peak", 1184200, "USD", "all", 0.99, 5, "manager"),
        (10, "Revenue trend", "The trough is August 2025", "trough", 742100, "USD", "all", 0.99, 5, "manager"),
        (11, "Revenue trend", "Seasonality index for December is 1.31", "seasonality", 1.31, "ratio", "all", 0.90, 4, "manager"),
        (12, "Region", "South contributed 42% of the decline", "south_share", 42.0, "%", "South", 0.88, 9, "manager"),
        (13, "Region", "West contributed 29% of the decline", "west_share", 29.0, "%", "West", 0.85, 8, "manager"),
        (14, "Region", "Northeast grew 1.4%", "ne_growth", 1.4, "%", "Northeast", 0.82, 4, "manager"),
        (15, "Channel", "Web traffic fell 17%", "web_traffic", -17.0, "%", "web", 0.95, 9, "manager"),
        (16, "Channel", "App conversion held near 10.7%", "app_cvr", 10.7, "%", "app", 0.80, 4, "manager"),
        (17, "Channel", "Phone AOV is lowest at $168", "phone_aov", 168.0, "USD", "phone", 0.78, 3, "manager"),
        (18, "Category", "Camping units fell 5.1%", "camping_units", -5.1, "%", "Camping", 0.86, 6, "manager"),
        (19, "Category", "Footwear units grew 14.8%", "footwear_units", 14.8, "%", "Footwear", 0.86, 6, "manager"),
        (20, "Promotion", "December lift was 2.3% below November", "dec_lift", -2.3, "%", "all", 0.70, 5, "manager"),
        (21, "Promotion", "Pull-forward could not be ruled out", "pull_forward", 1, "binary", "all", 0.55, 6, "manager"),
        (22, "Margin", "Gross margin is 38.2%", "gross_margin", 38.2, "%", "all", 0.99, 6, "manager"),
        (23, "Margin", "Operating margin is 19.2%", "operating_margin", 19.2, "%", "all", 0.99, 6, "manager"),
        (24, "Margin", "Returns consume 6.6% of revenue", "return_rate", 6.6, "%", "all", 0.97, 5, "manager"),
        (25, "Customers", "RFM champions are 16% of the base", "champions", 16.3, "%", "all", 0.90, 5, "external"),
        (26, "Customers", "At-risk customers are 23% of the base", "at_risk", 23.4, "%", "all", 0.90, 7, "external"),
        (27, "Customers", "NPS is -0.41", "nps", -0.41, "score", "all", 0.85, 5, "external"),
        (28, "Operations", "Cycle time is 7.15 hours", "cycle_time", 7.15, "hours", "all", 0.99, 4, "manager"),
        (29, "Operations", "Transit is 56% of cost to serve", "transit_share", 56.0, "%", "all", 0.95, 5, "manager"),
        (30, "Recommendation", "Fund a regional promotion pilot", "pilot", 1, "binary", "South,West", 0.72, 10, "executive"),
    ]
    return pd.DataFrame(rows, columns=[
        "id", "section", "finding", "metric", "value", "unit",
        "segment", "confidence", "relevance", "audience"])


def kpi() -> pd.DataFrame:
    rows = [
        ("Revenue", 23.52, 24.80, 25.10, "USDm", "higher", "Sales lead"),
        ("Orders", 131922, 138000, 140210, "count", "higher", "Sales lead"),
        ("Conversion", 0.1066, 0.1120, 0.1108, "share", "higher", "Growth"),
        ("AOV", 178.30, 180.00, 179.10, "USD", "higher", "Merchandising"),
        ("Gross margin", 0.3816, 0.3850, 0.3830, "share", "higher", "Finance"),
        ("Operating margin", 0.1916, 0.1950, 0.1930, "share", "higher", "Finance"),
        ("Return rate", 0.0656, 0.0600, 0.0620, "share", "lower", "Quality"),
        ("Campaign spend", 1.07, 1.10, 1.02, "USDm", "lower", "Marketing"),
        ("ROAS", 1.73, 1.90, 1.82, "ratio", "higher", "Marketing"),
        ("New customers", 4000, 4300, 4200, "count", "higher", "Growth"),
        ("NPS", -0.409, -0.20, -0.35, "score", "higher", "CX"),
        ("Cycle time", 7.15, 6.50, 6.90, "hours", "lower", "Operations"),
    ]
    return pd.DataFrame(rows, columns=[
        "kpi", "actual", "target", "prior", "unit", "direction", "owner"])


def scenarios() -> pd.DataFrame:
    rows = [
        ("Conversion rate", 0.030, 0.042, 0.055),
        ("Average order value", 150.0, 178.0, 210.0),
        ("Retention uplift", 0.05, 0.18, 0.32),
        ("Implementation cost", 320000.0, 240000.0, 190000.0),
    ]
    return pd.DataFrame(rows, columns=["assumption", "low", "base", "high"])


def questions() -> pd.DataFrame:
    rows = [
        (1, "Why did revenue fall?", "Traffic, not price or conversion", "Revenue bridge; −$506k volume effect"),
        (2, "Where is it concentrated?", "South and West, 71% of the decline", "Regional decomposition"),
        (3, "What can we do about it?", "Fund a regional promotion pilot", "Expected value $1.4m"),
        (4, "How sure are you?", "Moderate; the pilot resolves it", "Confidence 0.70; holdout planned"),
        (5, "What would change your mind?", "A flat holdout response", "Pre-agreed stop rule"),
    ]
    return pd.DataFrame(rows, columns=["qid", "question", "answer", "evidence"])


def audiences() -> pd.DataFrame:
    rows = [
        ("Executive", "30 seconds", "Decision and size of the effect", "A one-line ask"),
        ("Manager", "2 minutes", "What changed and who owns it", "A ranked driver table"),
        ("Technical peer", "10 minutes", "Method, assumptions, and checks", "Notebook and diagnostics"),
        ("External client", "5 minutes", "Conclusion with caveats", "A signed-off summary"),
    ]
    return pd.DataFrame(rows, columns=[
        "audience", "attention", "first_need", "artifact"])


def figures(f: pd.DataFrame, k: pd.DataFrame, s: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED)
    months = pd.period_range("2024-09", periods=12, freq="M").astype(str)
    rev = np.round(950000 + np.arange(12) * 12000
                   + rng.normal(0, 40000, 12), 0)

    # 1. bad chart
    f1, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(months, rev, marker="o")
    ax.set_title("Revenue by month")
    ax.set_xlabel("Month"); ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=45)
    f1.tight_layout(); f1.savefig(IMAGES / "ch18_bad_chart.png", dpi=130)
    plt.close(f1)

    # 2. good chart
    f2, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(months, rev, marker="o", color="#c0392b", lw=2)
    ax.axhline(rev[0], color="k", ls="--", lw=1)
    ax.annotate("Revenue down 6.2% year over year",
                xy=(6, rev[6]), xytext=(2, rev.min()),
                arrowprops=dict(arrowstyle="->", color="#2c3e50"))
    ax.set_title("Revenue is down 6.2%; traffic explains the gap",
                 fontsize=12, weight="bold")
    ax.set_xlabel(""); ax.set_ylabel("Net revenue ($)")
    ax.tick_params(axis="x", rotation=45)
    f2.tight_layout(); f2.savefig(IMAGES / "ch18_good_chart.png", dpi=130)
    plt.close(f2)

    # 3. RAG KPI panel
    labels = ["Revenue", "Conversion", "Margin", "Returns", "ROAS", "Cycle"]
    status = ["red", "red", "amber", "red", "amber", "green"]
    colours = {"red": "#c0392b", "amber": "#f39c12", "green": "#27ae60"}
    f3, axes = plt.subplots(2, 3, figsize=(9.5, 4.8))
    for ax, lab, st in zip(axes.ravel(), labels, status):
        ax.set_facecolor(colours[st]); ax.set_alpha(0.25)
        ax.text(0.5, 0.62, lab, ha="center", transform=ax.transAxes, fontsize=11)
        ax.text(0.5, 0.28, st.upper(), ha="center", transform=ax.transAxes,
                fontsize=10, weight="bold", color=colours[st])
        ax.set_xticks([]); ax.set_yticks([])
    f3.suptitle("Monthly KPI review — RAG status")
    f3.tight_layout(); f3.savefig(IMAGES / "ch18_rag.png", dpi=130)
    plt.close(f3)

    # 4. one-page brief as an image
    f4, ax = plt.subplots(figsize=(8.0, 10.0))
    ax.axis("off")
    text = ("FINDING FIRST\n\n"
            "Revenue fell 6.2% year over year.\n"
            "South and West account for 71% of the gap.\n\n"
            "EVIDENCE\n"
            "Traffic −17%; conversion flat; AOV +1.1%.\n"
            "The sequential bridge closes to $0.\n\n"
            "IMPLICATION\n"
            "This is an acquisition problem, not pricing.\n\n"
            "RECOMMENDATION\n"
            "Fund a $180k regional pilot. Owner: Sales lead.\n"
            "Measure precision at capacity in four weeks.\n\n"
            "LIMITATIONS\n"
            "Pull-forward cannot be ruled out without a holdout.")
    ax.text(0.06, 0.94, text, va="top", family="monospace", fontsize=12,
            linespacing=1.6, transform=ax.transAxes)
    f4.savefig(IMAGES / "ch18_onepager.png", dpi=130, bbox_inches="tight")
    plt.close(f4)

    # 5. deck storyboard
    f5, axes = plt.subplots(2, 5, figsize=(13, 5))
    titles = ["Title", "Finding", "Evidence", "Cause", "Driver 1",
              "Driver 2", "Uncertainty", "Options", "Recommendation", "Appendix"]
    for ax, t_, i in zip(axes.ravel(), titles, range(1, 11)):
        ax.set_facecolor("#ecf0f1")
        ax.text(0.5, 0.6, f"{i}. {t_}", ha="center", transform=ax.transAxes, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    f5.suptitle("Ten-slide monthly review storyboard")
    f5.tight_layout(); f5.savefig(IMAGES / "ch18_deck.png", dpi=130)
    plt.close(f5)

    # 6. uncertainty ranges
    f6, ax = plt.subplots(figsize=(8.0, 4.2))
    names = ["Conversion", "AOV", "Uplift", "Cost"]
    base = [4.2, 178, 18, 240]
    lo = [3.0, 150, 5, 190]
    hi = [5.5, 210, 32, 320]
    y = np.arange(len(names))
    ax.errorbar(base, y, xerr=[[b - l for b, l in zip(base, lo)],
                               [h - b for b, h in zip(base, hi)]],
                fmt="o", color="#c0392b", capsize=4)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_title("Ranges beat false precision")
    f6.tight_layout(); f6.savefig(IMAGES / "ch18_uncertainty.png", dpi=130)
    plt.close(f6)

    # 7. tornado of communication priorities
    f7, ax = plt.subplots(figsize=(7.0, 4.0))
    eff = {"Finding": 5, "Evidence": 4, "Recommendation": 3,
           "Method": 2, "Appendix": 1}
    ax.barh(list(eff), list(eff.values()), color="#8e44ad", alpha=0.85)
    ax.set(title="Attention by slide position", xlabel="Relative recall")
    f7.tight_layout(); f7.savefig(IMAGES / "ch18_attention.png", dpi=130)
    plt.close(f7)

    # 8. handover pipeline
    f8, ax = plt.subplots(figsize=(9.0, 3.2))
    stages = ["Raw data", "Generator", "Notebook", "Report", "Decision"]
    xs = np.arange(len(stages))
    ax.plot(xs, [1] * len(stages), "o-", lw=2, color="#2980b9", markersize=10)
    for x, s_ in zip(xs, stages):
        ax.text(x, 1.16, s_, ha="center", fontsize=9)
    ax.set_yticks([]); ax.set_xticks([]); ax.set_ylim(0.7, 1.35)
    ax.set(title="Data lineage a stranger can follow")
    f8.tight_layout(); f8.savefig(IMAGES / "ch18_lineage.png", dpi=130)
    plt.close(f8)


def main() -> None:
    REP.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    print("generating Chapter 18 reporting fixtures ...")
    f = findings(); f.to_csv(REP / "rep_findings.csv", index=False)
    k = kpi(); k.to_csv(REP / "rep_kpi.csv", index=False)
    s = scenarios(); s.to_csv(REP / "rep_scenarios.csv", index=False)
    q = questions(); q.to_csv(REP / "rep_questions.csv", index=False)
    au = audiences(); au.to_csv(REP / "rep_audiences.csv", index=False)
    print("drawing figures ...")
    figures(f, k, s)
    for p in sorted(REP.glob("rep_*.csv")):
        print(f"  {p.name}: {len(pd.read_csv(p))} rows")
    for p in sorted(IMAGES.glob("ch18_*.png")):
        print(f"  {p.name}: {p.stat().st_size:,} bytes")
    print("done.")


if __name__ == "__main__":
    main()
