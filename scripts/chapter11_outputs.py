#!/usr/bin/env python3
"""Print every output block used in Chapter 11, from the current files.

Run from the project root:

    python scripts/chapter11_outputs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "raw" / "audit"
SEED = 1107

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 80)


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def sales() -> pd.DataFrame:
    return pd.read_csv(AUDIT / "viz_sales.csv")


def dist() -> pd.DataFrame:
    return pd.read_csv(AUDIT / "viz_distributions.csv")


def bad() -> pd.DataFrame:
    return pd.read_csv(AUDIT / "viz_bad_chart.csv")


def contrast(hex1: str, hex2: str) -> float:
    def lum(h):
        rgb = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        rgb = np.where(rgb <= 0.03928, rgb / 12.92,
                       ((rgb + 0.055) / 1.055) ** 2.4)
        return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    a, b = lum(hex1), lum(hex2)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


# ======================================================== 11.1 principles
def section_111() -> None:
    s = sales()
    show("P1 dataset shape", {"rows": len(s),
                               "months": int(s["month"].nunique()),
                               "regions": int(s["region"].nunique()),
                               "categories": int(s["category"].nunique()),
                               "grain": "month x region x category"})

    jobs = pd.DataFrame([
        ("comparison", "bar, dot plot", "position/length"),
        ("trend", "line", "position over time"),
        ("distribution", "histogram, box, violin", "area/position"),
        ("relationship", "scatter", "position"),
        ("composition", "stacked bar", "length"),
        ("geography", "map", "position"),
    ], columns=["job", "default_chart", "visual_channel"])
    show("P2 chart selection table", jobs.to_string(index=False))

    channels = pd.DataFrame([
        ("position on common scale", 1, "most accurate"),
        ("length", 2, "accurate with a zero baseline"),
        ("angle", 3, "moderate"),
        ("area", 4, "poor beyond small ratios"),
        ("hue / saturation", 5, "weakest for magnitude"),
    ], columns=["channel", "accuracy_rank", "note"])
    show("P3 perceptual ranking", channels.to_string(index=False))

    m = s.groupby("month", as_index=False).agg(
        revenue=("revenue", "sum"), orders=("orders", "sum"))
    q = m["revenue"].iloc[-6:].to_numpy()
    show("P4 truncated vs zero axis", {
        "truncated_min": round(float(q.min()), 2),
        "truncated_max": round(float(q.max()), 2),
        "truncated_span": round(float(q.max() - q.min()), 2),
        "zero_span": round(float(q.max()), 2),
        "visual_amplification": round(float(q.max() / (q.max() - q.min())), 2),
    })

    show("P5 log scale candidate", {
        "min": round(float(s["revenue"].min()), 2),
        "max": round(float(s["revenue"].max()), 2),
        "ratio": round(float(s["revenue"].max() / s["revenue"].min()), 2),
        "log_recommended": bool(s["revenue"].max() / s["revenue"].min() > 100),
    })

    show("P6 dual axis test", {
        "revenue_range": [round(float(m["revenue"].min()), 2),
                          round(float(m["revenue"].max()), 2)],
        "aov_range": [round(float((m["revenue"] / m["orders"]).min()), 2),
                      round(float((m["revenue"] / m["orders"]).max()), 2)],
        "note": "independent scales let the presenter imply any correlation",
    })

    show("P7 figures", ["images/ch11_four_panel.png",
                         "images/ch11_axis_compare.png"])


# ====================================================== 11.2 catalogue
def section_112() -> None:
    cat = pd.DataFrame([
        ("bar", "compare categories", "truncated baseline"),
        ("line", "trend over time", "too many series"),
        ("histogram", "one distribution", "bin-width dependence"),
        ("box", "group comparison", "hides modality"),
        ("violin", "group shape", "KDE on discrete data"),
        ("scatter", "two numeric variables", "overplotting"),
        ("bubble", "three variables", "area scaling by radius"),
        ("pie", "few parts of a whole", "more than 5 slices"),
        ("heatmap", "matrix", "unordered rows"),
        ("area", "composition over time", "hard to read middle bands"),
        ("treemap", "nested composition", "area comparison"),
        ("waterfall", "bridge from start to end", "unsigned bars"),
        ("funnel", "stages", "unequal stage widths"),
        ("map", "geography", "area distortion"),
    ], columns=["chart", "question", "failure_mode"])
    show("C1 chart catalogue", cat.to_string(index=False))

    b = bad()
    show("C2 bad pie inputs", {"slices": len(b),
                                "smallest_share": round(float(
                                    b["value"].min() / b["value"].sum()), 4),
                                "largest_share": round(float(
                                    b["value"].max() / b["value"].sum()), 4)})

    comp = sales().groupby("category")["revenue"].sum()
    show("C3 composition with 3 categories", comp.round(2).to_dict())

    show("C4 box vs violin", {
        "groups": 4, "same_underlying_data": True,
        "box_shows": "median, IQR, whiskers, outliers",
        "violin_adds": "density shape",
    })

    show("C5 heatmap ordering", {"rows": 5, "cols": 5,
                                  "lower_triangle_cells": 10})

    show("C6 figure", "images/ch11_bad_pie.png")


# ================================================== 11.3 colour/annotation
def section_113() -> None:
    palette = {"blue": "#4477aa", "orange": "#ee7733",
               "green": "#228833", "red": "#cc6677",
               "purple": "#aa3377", "grey": "#bbbbbb"}
    show("K1 categorical palette", palette)
    show("K2 contrast against white", {
        k: round(contrast(v, "#ffffff"), 2) for k, v in palette.items()})

    show("K3 colourblind-safe pair", {
        "okabe_ito_blue": "#0072B2", "okabe_ito_orange": "#E69F00",
        "contrast": round(contrast("#0072B2", "#E69F00"), 2),
    })

    seq = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1",
           "#6baed6", "#4292c6", "#2171b5", "#084594"]
    show("K4 sequential palette", {"steps": len(seq), "colors": seq})

    div = ["#b2182b", "#d6604d", "#f4a582", "#fddbc7",
           "#f7f7f7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac"]
    show("K5 diverging palette", {"center": div[len(div)//2],
                                   "zero_centered": True})

    m = sales().groupby("month", as_index=False)["revenue"].sum()
    peak = int(m["revenue"].idxmax())
    show("K6 annotation target", {
        "peak_month": m["month"][peak], "peak_revenue": round(float(m["revenue"][peak]), 2),
        "position": [peak, round(float(m["revenue"][peak]), 2)],
    })

    show("K7 title as takeaway", {
        "weak": "Revenue by month",
        "strong": "Revenue held near $1.9m per month despite a 12% cost increase",
    })


# ======================================================= 11.4 matplotlib
def section_114() -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    show("M1 environment", {"matplotlib": matplotlib.__version__,
                             "backend": matplotlib.get_backend()})

    m = sales().groupby("month", as_index=False).agg(
        revenue=("revenue", "sum"), orders=("orders", "sum"))
    show("M2 figure data", {"rows": len(m),
                             "first": m.head(2).round(2).to_dict("records"),
                             "last": m.tail(2).round(2).to_dict("records")})

    show("M3 object-oriented pieces", [
        "Figure is the canvas", "Axes is one plot",
        "Axis controls ticks and labels", "Artist is anything drawn",
    ])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(m["month"], m["revenue"])
    show("M4 figure anatomy", {
        "axes_count": len(fig.axes),
        "lines": len(ax.lines),
        "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
    })
    plt.close(fig)

    show("M5 style settings", {
        "axes.grid": matplotlib.rcParams["axes.grid"],
        "figure.dpi": matplotlib.rcParams["figure.dpi"],
        "axes.spines.top": matplotlib.rcParams["axes.spines.top"],
    })

    fig, axes = plt.subplots(2, 2)
    show("M6 subplots", {"shape": np.array(axes).shape,
                          "axes_total": int(np.array(axes).size)})
    plt.close(fig)

    show("M7 formatters", {
        "example_1000": f"{1000:,.0f}",
        "percent_example": f"{0.1267:.1%}",
        "currency_example": f"${12345.67:,.2f}",
    })

    show("M8 exports", ["PNG for slides", "SVG for web and editing",
                         "PDF for print", "all from one Figure object"])


# ========================================================== 11.5 seaborn
def section_115() -> None:
    import importlib.util as u
    installed = u.find_spec("seaborn") is not None
    show("S1 availability", {"seaborn_installed": installed,
                              "status": "optional; equivalent figures drawn with Matplotlib"})

    s = sales()
    grp = s.groupby(["month", "category"])["revenue"].mean().reset_index()
    sub = grp[grp["month"] == "2025-06"]
    show("S2 seaborn barplot semantics", {
        "category_means": sub.set_index("category")["revenue"].round(2).to_dict(),
        "default_ci": "95% bootstrap CI of the mean at each category",
        "n_per_category": int(s[(s["month"] == "2025-06")].groupby(
            "category").size().iloc[0]),
    })

    d = dist()
    show("S3 distribution functions", {
        "histplot": "counts, bin-driven", "kdeplot": "estimated density",
        "ecdfplot": "empirical CDF, no bandwidth", "discrete_column": "discrete",
    })
    show("S4 KDE caution", {
        "discrete_unique_values": int(d["discrete"].nunique()),
        "recommendation": "use ecdfplot or a bar chart for discrete counts",
    })

    show("S5 axes-level vs figure-level", {
        "axes_level": "histplot, boxplot, scatterplot - draw on an existing Axes",
        "figure_level": "displot, catplot, relplot - own a Figure and facets",
    })

    show("S6 figure", "images/ch11_box_violin_strip.png")


# ==================================================== 11.6 pandas plotting
def section_116() -> None:
    m = sales().groupby("month", as_index=False)["revenue"].sum()
    series = m.set_index("month")["revenue"]
    show("PD1 series plot", {"type": type(series).__name__,
                              "points": len(series),
                              "index_name": series.index.name})

    piv = sales().pivot_table(index="month", columns="region",
                              values="revenue", aggfunc="sum")
    show("PD2 frame plot", {"shape": piv.shape,
                             "columns": piv.columns.tolist()})

    show("PD3 when convenience is enough", [
        "a quick line for an internal notebook",
        "a bar chart for a one-off check",
        "a histogram during exploration",
    ])
    show("PD4 when it is not", [
        "publication figures need explicit styles",
        "annotations and reference lines need the object API",
        "facets with shared axes need subplots or a figure-level API",
    ])


# ========================================================== 11.7 plotly
def section_117() -> None:
    import importlib.util as u
    installed = u.find_spec("plotly") is not None
    show("PL1 availability", {"plotly_installed": installed,
                               "status": "optional; static equivalents drawn with Matplotlib"})

    m = sales().groupby("month", as_index=False).agg(
        revenue=("revenue", "sum"), orders=("orders", "sum"))
    show("PL2 interactive line data", {"rows": len(m),
                                        "hover_fields": ["revenue", "orders"],
                                        "slider": "month range selector"})

    show("PL3 graph objects vs express", {
        "express": "one call from a DataFrame, sensible defaults",
        "graph_objects": "explicit traces, layouts, and fine control",
    })

    show("PL4 dropdown design", {
        "control": "region selector",
        "regions": sorted(sales()["region"].unique().tolist()),
        "behaviour": "one trace per region, visible toggled by dropdown",
    })

    show("PL5 export", ["HTML with embedded JS", "PNG requires kaleido",
                         "static PDF via Matplotlib for print"])

    show("PL6 WebGL threshold", {"points": len(sales()),
                                  "recommendation": "scattergl above ~100k points"})


# ======================================================== 11.8 dashboards
def section_118() -> None:
    s = sales()
    show("D1 dashboard data", {
        "rows": len(s), "aggregated_rows": int(s.groupby(
            ["month", "region"]).ngroups),
        "reduction": round(1 - s.groupby(["month", "region"]).ngroups / len(s), 3),
    })

    show("D2 layout", ["title and period", "filters on one row",
                        "KPI cards", "trend chart", "detail table"])

    show("D3 state", {
        "filters": ["month range", "region", "category"],
        "caching": "aggregate once, not per callback",
        "empty_state": "show 'no data for this selection', not a blank chart",
    })

    show("D4 performance", {
        "raw_rows": len(s), "month_region_rows": int(s.groupby(
            ["month", "region"]).ngroups),
        "rule": "aggregate before sending to the browser",
    })

    show("D5 when a dashboard is wrong", [
        "the question has one answer - write a memo",
        "the audience will look once - send a static figure",
        "the metric needs a definition - write it down first",
    ])


# ================================================== 11.9 misleading charts
def section_119() -> None:
    m = sales().groupby("month", as_index=False)["revenue"].sum()
    q = m["revenue"].iloc[-6:].to_numpy()
    show("X1 truncated axis", {
        "shown_span": round(float(q.max() - q.min()), 2),
        "true_span": round(float(q.max()), 2),
        "factor": round(float(q.max() / (q.max() - q.min())), 2),
    })

    sizes = bad()["orders"].to_numpy()
    show("X2 bubble radius error", {
        "max_value": int(sizes.max()), "min_value": int(sizes.min()),
        "radius_ratio": round(float(np.sqrt(sizes.max() / sizes.min())), 2),
        "value_ratio": round(float(sizes.max() / sizes.min()), 2),
    })

    show("X3 cherry-picked window", {
        "full_months": len(m), "last_3_growth": round(float(
            m["revenue"].iloc[-1] / m["revenue"].iloc[-3] - 1), 4),
        "full_growth": round(float(
            m["revenue"].iloc[-1] / m["revenue"].iloc[0] - 1), 4),
    })

    show("X4 cumulative-only", {
        "cumulative_always_rises": True,
        "fix": "show monthly increments beside the cumulative line",
    })

    show("X5 inconsistent baselines", {
        "panel_a_baseline": 0, "panel_b_baseline": round(float(q.min()), 2),
        "fix": "share one y-axis across panels",
    })

    show("X6 unlabeled units", {
        "bad_label": "Revenue", "good_label": "Revenue (USD thousands, per month)",
    })

    checklist = [
        "Does the y-axis start at zero for bars?",
        "Are units in the axis label or subtitle?",
        "Is the time window stated?",
        "Are all compared panels on the same scale?",
        "Is colour used for categories only?",
        "Is the palette colourblind safe?",
        "Are bubbles area-scaled?",
        "Are denominators stated for percentages?",
        "Is the baseline meaningful?",
        "Are labels direct rather than in a long legend?",
        "Does the title state the takeaway?",
        "Would the chart survive a screenshot without the presenter?",
    ]
    show("X7 twelve-question checklist", "\n".join(
        f"{i+1:2d}. {q}" for i, q in enumerate(checklist)))


# ==================================================== 11.10 storytelling
def section_1110() -> None:
    m = sales().groupby("month", as_index=False).agg(
        revenue=("revenue", "sum"), orders=("orders", "sum"))
    m["aov"] = m["revenue"] / m["orders"]
    show("ST1 one message", {
        "finding": "AOV stayed near its average while revenue grew",
        "aov_min": round(float(m["aov"].min()), 2),
        "aov_max": round(float(m["aov"].max()), 2),
        "revenue_growth": round(float(
            m["revenue"].iloc[-1] / m["revenue"].iloc[0] - 1), 4),
    })

    show("ST2 sequence", ["context: revenue trend", "tension: is AOV rising too?",
                           "resolution: AOV flat", "action: growth is volume-driven"])

    show("ST3 annotation as narrative", {
        "weak": "line chart with a peak",
        "strong": "annotated peak with a one-line cause in the subtitle",
    })

    show("ST4 so-what caption", {
        "caption": "Revenue rose 15.5% over 24 months; average order value "
                   "stayed between $179 and $201, so the gain came from order "
                   "volume rather than basket size.",
    })

    show("ST5 static vs interactive", {
        "static": "a memo or slide with one takeaway",
        "interactive": "an explorer for an audience with different questions",
    })


# ====================================================== 11.11 example bank
def section_1111() -> None:
    s = sales()
    show("E1 months", int(s["month"].nunique()))
    show("E2 grain unique", bool(not s.duplicated(
        ["month", "region", "category"]).any()))
    show("E3 total revenue", round(float(s["revenue"].sum()), 2))
    show("E4 total orders", int(s["orders"].sum()))
    show("E5 average margin pct", round(float(
        s["margin"].sum() / s["revenue"].sum() * 100), 2))
    show("E6 max single month", s.groupby("month")["revenue"].sum().idxmax())
    show("E7 category shares", s.groupby("category")["revenue"].sum().div(
        s["revenue"].sum()).round(4).to_dict())
    show("E8 region order", s.groupby("region")["revenue"].sum().sort_values(
        ascending=False).round(2).to_dict())


# ========================================================= 11.12 capstone
def section_1112() -> None:
    s = sales()
    m = s.groupby("month", as_index=False).agg(
        revenue=("revenue", "sum"), orders=("orders", "sum"),
        margin=("margin", "sum"))
    m["aov"] = (m["revenue"] / m["orders"]).round(2)
    show("K1 scorecard summary", m.head(4).round(2).to_string(index=False))

    show("K2 chart decisions", {
        "trend": "line, zero baseline, months on x",
        "comparison": "sorted horizontal bars, direct labels",
        "distribution": "histogram with fixed bins across months",
        "composition": "stacked area only if order matters",
        "colour": "Okabe-Ito categorical, diverging centred at zero",
    })

    show("K3 control totals", {
        "rows": len(s), "months": int(s["month"].nunique()),
        "revenue": round(float(s["revenue"].sum()), 2),
        "orders": int(s["orders"].sum()),
        "margin": round(float(s["margin"].sum()), 2),
    })

    show("K4 figures", [
        "images/ch11_four_panel.png", "images/ch11_annotated_line.png",
        "images/ch11_grouped_bars.png", "images/ch11_story.png",
    ])


def section_extra() -> None:
    """Extra worked blocks for the expanded Chapter 11 sections."""
    s = sales()
    d = dist()
    b = bad()
    co = pd.read_csv(AUDIT / "viz_cohort.csv")
    m = s.groupby("month", as_index=False).agg(
        revenue=("revenue", "sum"), orders=("orders", "sum"))
    m["growth"] = m["revenue"].pct_change().round(4)
    m["share"] = (m["revenue"] / m["revenue"].sum()).round(4)

    show("X1 monthly scorecard", m.head(5).round(2).to_string(index=False))

    ct = pd.crosstab(s["region"], s["category"], values=s["revenue"],
                     aggfunc="sum").round(2)
    show("X2 region by category", ct.to_string())

    bins = [10, 25, 50, 100]
    counts = {f"{k}_bins": int(np.histogram(d["normal"], bins=k)[0].max())
              for k in bins}
    show("X3 histogram bin sensitivity", counts)

    q = [0.1, 0.5, 0.9]
    show("X4 ECDF quantiles", {str(k): round(float(d["normal"].quantile(k)), 2)
                                for k in q})

    g = d.groupby("discrete")["normal"].agg(["count", "median", "std"])
    show("X5 box plot source stats", g.head(4).round(2).to_string())

    corr = d.corr()
    pairs = [(corr.columns[i], corr.columns[j], round(float(corr.iloc[i, j]), 3))
             for i in range(len(corr)) for j in range(i + 1, len(corr))]
    show("X6 lower triangle pairs", pairs[:6])

    ret = co.groupby("month_index")["retention_pct"].mean().round(2)
    show("X7 cohort decay", ret.to_dict())

    peak = int(m["revenue"].idxmax())
    show("X8 annotation coordinates", {
        "x": peak, "y": round(float(m["revenue"][peak]), 2),
        "label": f"Peak {m['month'][peak]}",
        "text_offset": [-3, 0.05],
    })

    pal = {"blue": "#4477aa", "orange": "#ee7733", "green": "#228833",
           "red": "#cc6677", "purple": "#aa3377", "grey": "#bbbbbb"}
    show("X9 pairwise contrast", {
        "blue/orange": round(contrast(pal["blue"], pal["orange"]), 2),
        "blue/green": round(contrast(pal["blue"], pal["green"]), 2),
        "blue/purple": round(contrast(pal["blue"], pal["purple"]), 2),
        "purple/grey": round(contrast(pal["purple"], pal["grey"]), 2),
    })

    seq = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1",
           "#6baed6", "#4292c6", "#2171b5", "#084594"]
    lum = []
    for hx in seq:
        rgb = np.array([int(hx[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        lum.append(float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]))
    show("X10 sequential luminance", {
        "monotone_decreasing": bool(all(a > b for a, b in zip(lum, lum[1:]))),
        "values": [round(v, 3) for v in lum],
    })

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2)
    for ax in axes.ravel():
        ax.plot([0, 1], [0, 1])
    show("X11 four-panel artists", {
        "axes": len(fig.axes),
        "lines_total": int(sum(len(ax.lines) for ax in fig.axes)),
    })
    plt.close(fig)

    show("X12 GridSpec layout", {
        "rows": 2, "cols": 2, "cells": 4,
        "span_example": "top row spans two columns",
    })

    show("X13 date ticks", {
        "months": len(m), "tick_positions": [0, 6, 12, 18, 23],
        "labels": [m["month"][i] for i in [0, 6, 12, 18, 23]],
    })

    rng = np.random.default_rng(1107)
    sub = s[s["month"] == "2025-06"]
    means = {}
    for cat, grp in sub.groupby("category"):
        boots = [rng.choice(grp["revenue"].to_numpy(), size=len(grp),
                            replace=True).mean() for _ in range(1000)]
        means[cat] = {"mean": round(float(grp["revenue"].mean()), 2),
                      "ci_low": round(float(np.quantile(boots, 0.025)), 2),
                      "ci_high": round(float(np.quantile(boots, 0.975)), 2)}
    show("X14 bootstrap bar means", pd.DataFrame(means).T.round(2).to_string())

    show("X15 pandas plot dtypes", {
        str(k): str(v) for k, v in m.dtypes.items()
    })

    fig_spec = {
        "data": [{"x": m["month"].tolist(), "y": m["revenue"].tolist(),
                  "type": "scatter", "mode": "lines+markers"}],
        "layout": {"title": "Monthly revenue", "hovermode": "x unified"},
    }
    show("X16 graph object skeleton", {
        "traces": len(fig_spec["data"]),
        "trace_type": fig_spec["data"][0]["type"],
        "hovermode": fig_spec["layout"]["hovermode"],
    })

    by_region = s.groupby(["month", "region"])["revenue"].sum().reset_index()
    filt = by_region[by_region["region"] == "West"]
    show("X17 dashboard filter", {
        "base_rows": len(by_region), "filtered_rows": len(filt),
        "filtered_revenue": round(float(filt["revenue"].sum()), 2),
    })

    show("X18 growth windows", {
        "last_3": round(float(m["revenue"].iloc[-1] / m["revenue"].iloc[-3] - 1), 4),
        "last_6": round(float(m["revenue"].iloc[-1] / m["revenue"].iloc[-6] - 1), 4),
        "full": round(float(m["revenue"].iloc[-1] / m["revenue"].iloc[0] - 1), 4),
    })

    checks = {
        "zero_baseline": True,
        "units_labeled": True,
        "window_stated": False,
        "shared_scale": True,
        "direct_labels": True,
    }
    show("X19 pre-publication check", {
        "passed": sum(checks.values()), "total": len(checks),
        "failures": [k for k, v in checks.items() if not v],
    })

    show("X20 figure inventory", sorted(
        str(p.name) for p in (ROOT / "images").glob("ch11_*.png")))


def main() -> None:
    print("python", "pandas", pd.__version__, "numpy", np.__version__)
    print()
    for section in (
        section_111, section_112, section_113, section_114, section_115,
        section_116, section_117, section_118, section_119, section_1110,
        section_1111, section_1112, section_extra,
    ):
        print("=" * 78)
        section()


if __name__ == "__main__":
    main()
