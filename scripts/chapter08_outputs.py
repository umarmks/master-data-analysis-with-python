#!/usr/bin/env python3
"""Print every output block used in Chapter 8, from the current files.

Run from the project root:

    python scripts/chapter08_outputs.py

Reads the fixtures written by generate_chapter08_sources.py. The figures are
written by that script into images/ch08_*.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "raw" / "audit"
SEED8 = 808

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 80)


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def eda() -> pd.DataFrame:
    return pd.read_csv(AUDIT / "churn_eda.csv")


def gallery() -> pd.DataFrame:
    return pd.read_csv(AUDIT / "distribution_gallery.csv")


# ============================================================ 8.1 workflow
def section_81() -> None:
    df = eda()
    kinds = {
        "text": df.select_dtypes(include="object").columns.tolist(),
        "integer": df.select_dtypes(include="integer").columns.tolist(),
        "float": df.select_dtypes(include="floating").columns.tolist(),
    }
    show("W1 structure", {"rows": len(df), "columns": len(df.columns),
                          "kinds": {k: len(v) for k, v in kinds.items()}})
    show("W1 text columns", kinds["text"])
    show("W1 numeric columns", kinds["integer"] + kinds["float"])

    sweep = pd.DataFrame([
        ("1 structure", "shape, dtypes, memory", "34 columns, 3,000 rows"),
        ("2 grain", "one row per customer", "customer_id unique"),
        ("3 missingness", "per-column counts", "income missing 12.5%"),
        ("4 univariate", "profile each column by type", "34 profiles"),
        ("5 bivariate", "pairs chosen by question", "churn vs tenure, tickets, satisfaction"),
        ("6 groups", "segment, region, channel", "3, 4, 3 levels"),
        ("7 time", "signup month", "27 months"),
        ("8 anomalies", "fences and ratios", "total_spend tail"),
    ], columns=["step", "view", "result"])
    show("W2 standard sweep", sweep.to_string(index=False))

    log = pd.DataFrame([
        ("F1", "Tenure is strongly related to churn", "median tenure 12 vs 45 months", "confirm with a survival curve"),
        ("F2", "Support tickets raise churn", "0 tickets 55%, 3+ tickets 92%", "hold tenure constant"),
        ("F3", "Premium customers churn least", "51% vs 76% Basic", "check plan mix"),
        ("F4", "total_spend is bimodal", "peaks near $150 and $600", "segment by plan"),
        ("F5", "A small group has extreme spend", "36 lines above $1,650", "investigate, do not delete"),
    ], columns=["id", "finding", "evidence", "next_check"])
    show("W3 insight log", log.to_string(index=False))

    def describe_column(frame, col):
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s):
            return {"kind": "numeric", "n": int(s.notna().sum()),
                    "mean": round(float(s.mean()), 2), "median": round(float(s.median()), 2),
                    "sd": round(float(s.std()), 2)}
        return {"kind": "categorical", "n": int(s.notna().sum()),
                "unique": int(s.nunique(dropna=True)),
                "top": str(s.mode(dropna=True).iloc[0])}

    prof = pd.DataFrame([{"column": c, **describe_column(df, c)}
                         for c in df.columns[:8]])
    show("W4 describe_column over the first eight columns", prof.to_string(index=False))

    show("W5 missingness ranking", (df.isna().mean().sort_values(ascending=False).head(4) * 100).round(3).to_dict())
    show("W6 stop rule", {"findings_logged": len(log),
                          "open_questions": int((log["next_check"] != "").sum()),
                          "rule": "stop when new views stop changing the ranked list"})


# ========================================================= 8.2 center
def section_82() -> None:
    df = eda()
    rev = df["total_spend"]
    trimmed = stats.trim_mean(rev, 0.05)
    show("C1 four centers", {
        "mean": round(float(rev.mean()), 2),
        "median": round(float(rev.median()), 2),
        "trimmed_mean_5pct": round(float(trimmed), 2),
        "mode": round(float(rev.mode().iloc[0]), 2),
    })

    hi = rev.quantile(0.99)
    kept = rev[rev <= hi]
    show("C2 outlier sensitivity", {
        "mean_all": round(float(rev.mean()), 2),
        "mean_without_top_1pct": round(float(kept.mean()), 2),
        "median_all": round(float(rev.median()), 2),
        "median_without_top_1pct": round(float(kept.median()), 2),
        "p99": round(float(hi), 2),
    })

    w = df["total_spend"]
    x = df["discount_pct"]
    wm = float((w * x).sum() / w.sum())
    show("C3 weighted mean", {"plain_discount": round(float(x.mean()), 3),
                              "spend_weighted_discount": round(wm, 3)})

    growth = np.array([1.10, 0.95, 1.20, 1.08, 1.02])
    show("C4 geometric vs arithmetic growth", {
        "arithmetic": round(float(growth.mean()), 4),
        "geometric": round(float(stats.gmean(growth)), 4),
        "actual_cumulative": round(float(np.prod(growth)), 4),
        "arithmetic_implies": round(float(growth.mean() ** len(growth)), 4),
        "geometric_implies": round(float(stats.gmean(growth) ** len(growth)), 4),
    })

    speeds = np.array([40.0, 60.0])
    show("C5 harmonic mean for rates", {
        "arithmetic_speed": round(float(speeds.mean()), 2),
        "harmonic_speed": round(float(stats.hmean(speeds)), 2),
    })

    choice = pd.DataFrame([
        ("symmetric, no outliers", "mean", "uses all information"),
        ("skewed money", "median", "typical value, robust to tail"),
        ("growth rates", "geometric", "multiplicative process"),
        ("speeds, prices per unit", "harmonic", "rate with equal distance"),
        ("most common category", "mode", "only centre defined for text"),
    ], columns=["situation", "measure", "why"])
    show("C6 choosing the average", choice.to_string(index=False))

    show("C7 mode of a categorical", {
        "segment": str(df["segment"].mode().iloc[0]),
        "channel": str(df["channel"].mode().iloc[0]),
        "tier": str(df["loyalty_tier"].mode().iloc[0]),
    })

    inc = df["income"].dropna()
    show("C8 centre under skew", {"income_mean": round(float(inc.mean()), 2),
                                   "income_median": round(float(inc.median()), 2),
                                   "ratio": round(float(inc.mean() / inc.median()), 3)})


# ========================================================= 8.3 spread
def section_83() -> None:
    df = eda()
    x = df["total_spend"]
    q1, q3 = [float(v) for v in x.quantile([0.25, 0.75])]
    show("S1 spread measures", {
        "range": round(float(x.max() - x.min()), 2),
        "iqr": round(q3 - q1, 2),
        "variance_ddof0": round(float(x.var(ddof=0)), 2),
        "variance_ddof1": round(float(x.var(ddof=1)), 2),
        "sd": round(float(x.std()), 2),
        "mad": round(float((x - x.median()).abs().median()), 2),
        "cv": round(float(x.std() / x.mean()), 3),
    })

    methods = {m: round(float(x.quantile(0.75, interpolation=m)), 3)
               for m in ["linear", "lower", "higher", "midpoint", "nearest"]}
    show("S2 percentile methods disagree", methods)

    dec = x.quantile(np.arange(0, 1.01, 0.1)).round(2)
    show("S3 deciles", dec.to_dict())

    v = float(x.iloc[0])
    show("S4 rank and percentile", {
        "value": round(v, 2),
        "percentile": round(float((x < v).mean()), 4),
        "rank_pct": round(float(x.rank(pct=True).iloc[0]), 4),
    })

    z = (x - x.mean()) / x.std()
    rz = 0.6745 * (x - x.median()) / (x - x.median()).abs().median()
    show("S5 z and robust z", {
        "z_gt_3": int((z.abs() > 3).sum()),
        "robust_z_gt_3_5": int((rz.abs() > 3.5).sum()),
    })

    cv = df.groupby("segment")["total_spend"].agg(
        n="size", mean="mean", sd="std")
    cv["cv"] = (cv["sd"] / cv["mean"]).round(3)
    show("S6 coefficient of variation by group", cv.round(2).to_string())

    grand = float(df["total_spend"].mean())
    between = sum(len(g) * (float(g["total_spend"].mean()) - grand) ** 2
                  for _, g in df.groupby("segment"))
    within = sum(float(((g["total_spend"] - g["total_spend"].mean()) ** 2).sum())
                 for _, g in df.groupby("segment"))
    show("S7 variance decomposition", {
        "between_groups": round(between / 2, 2),
        "within_groups": round(within / (len(df) - 3), 2),
        "share_between": round(between / (between + within), 4),
    })

    k = 2
    show("S8 empirical rule", {
        "within_1_sd": round(float(((x - x.mean()).abs() <= x.std()).mean()), 4),
        "within_2_sd": round(float(((x - x.mean()).abs() <= k * x.std()).mean()), 4),
        "chebyshev_lower_bound": 1 - 1 / k ** 2,
    })


# ========================================================= 8.4 shape
def section_84() -> None:
    g = gallery()
    rows = []
    for col in g.columns:
        s = g[col]
        rows.append((col, round(float(skew := stats.skew(s)), 3),
                     round(float(stats.kurtosis(s)), 3),
                     int((s == 0).sum()) if col in ["zero_inflated", "sparse_binary"] else 0))
    show("H1 shape of six distributions", pd.DataFrame(
        rows, columns=["column", "skew", "excess_kurtosis", "zeros"]).to_string(index=False))

    rev = eda()["total_spend"]
    counts = [10, 25, 50, 100, 250]
    show("H2 bin-count sensitivity", {
        f"{b}_bins": int(np.histogram(rev, bins=b)[0].max()) for b in counts})

    q = [0.01, 0.25, 0.50, 0.75, 0.99]
    show("H3 ECDF quantiles", {str(v): round(float(rev.quantile(v)), 2) for v in q})

    q1, q3 = [float(v) for v in rev.quantile([0.25, 0.75])]
    hi = q3 + 1.5 * (q3 - q1)
    show("H4 box-plot fence", {"upper_fence": round(hi, 2),
                                "flagged": int((rev > hi).sum()),
                                "share": round(float((rev > hi).mean()), 4)})

    logins = eda()["logins_30d"]
    show("H5 modality of logins", {
        "distinct_values": int(logins.nunique()),
        "top_values": logins.value_counts().head(3).to_dict(),
        "mean": round(float(logins.mean()), 3),
    })

    show("H6 Q-Q evidence", {"shapiro_p": float(f"{stats.shapiro(logins).pvalue:.3g}"),
                             "skew": round(float(stats.skew(logins)), 3)})

    base = eda()["total_spend"]
    bc, lam = stats.boxcox(base + 1)
    show("H7 transformations to symmetry", pd.DataFrame([
        ("raw", stats.skew(base)),
        ("log1p", stats.skew(np.log1p(base))),
        ("sqrt", stats.skew(np.sqrt(base))),
        ("boxcox", stats.skew(bc)),
    ], columns=["transform", "skew"]).round(3).to_string(index=False))
    show("H7 boxcox lambda", round(float(lam), 3))

    show("H8 figures", [
        "images/ch08_hist_ecdf_box.png",
        "images/ch08_distribution_gallery.png",
        "images/ch08_qq.png",
    ])


# ======================================================= 8.5 categorical
def section_85() -> None:
    df = eda()
    seg = df["segment"]
    tab = seg.value_counts()
    cat = pd.DataFrame({"count": tab, "pct": (tab / len(df) * 100).round(2),
                        "cum_pct": (tab.cumsum() / len(df) * 100).round(2)})
    show("K1 frequency with cumulative", cat.to_string())

    show("K2 cardinality and rare levels", {
        c: {"unique": int(df[c].nunique(dropna=True)),
            "rare_levels_lt_5pct": int((df[c].value_counts(normalize=True) < 0.05).sum())}
        for c in ["segment", "region", "channel", "plan", "loyalty_tier", "device"]
    })

    sat = df["satisfaction"].value_counts().sort_index()
    show("K3 ordered category", pd.DataFrame({"count": sat,
         "pct": (sat / len(df) * 100).round(2)}).to_string())

    inc_miss = df["income"].isna().groupby(df["segment"]).mean().round(4)
    show("K4 missing as a category", inc_miss.to_dict())

    ct = pd.crosstab(df["segment"], df["churn"], normalize="index").round(4)
    show("K5 contingency row percentages", ct.to_string())

    tab2 = pd.crosstab(df["segment"], df["churn"])
    chi2, p, dof, _ = stats.chi2_contingency(tab2)
    show("K6 chi-square preview", {"chi2": round(float(chi2), 3),
                                    "p": float(f"{p:.3g}"), "dof": int(dof)})


# ======================================================== 8.6 bivariate
def section_86() -> None:
    df = eda()
    cols = ["tenure_months", "monthly_charges", "support_tickets", "logins_30d",
            "session_minutes", "avg_order_value", "total_spend", "satisfaction", "churn"]
    corr = df[cols].corr(numeric_only=True).round(3)
    show("B1 correlation matrix", corr.to_string())

    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    top = pairs.stack().abs().sort_values(ascending=False).head(6)
    show("B2 strongest pairs", {f"{a}~{b}": round(float(v), 3)
                                 for (a, b), v in top.items()})

    g = df.groupby("segment").agg(
        n=("customer_id", "size"),
        mean_spend=("total_spend", "mean"),
        median_spend=("total_spend", "median"),
        churn_rate=("churn", "mean")).round(3)
    show("B3 numeric by categorical", g.to_string())

    rng = np.random.default_rng(SEED8)
    cis = {}
    for seg, sub in df.groupby("segment"):
        vals = sub["total_spend"].to_numpy()
        boots = np.median(rng.choice(vals, size=(1000, len(vals)), replace=True), axis=1)
        cis[seg] = {"median": round(float(np.median(vals)), 2),
                    "lo": round(float(np.quantile(boots, 0.025)), 2),
                    "hi": round(float(np.quantile(boots, 0.975)), 2)}
    show("B4 bootstrap medians", pd.DataFrame(cis).T.to_string())

    show("B5 categorical pair", pd.crosstab(
        df["plan"], df["churn"], normalize="index").round(4).to_string())

    trend = df.assign(month=pd.to_datetime(df["signup_date"]).dt.strftime("%Y-%m")).groupby(
        "month").agg(customers=("customer_id", "size"),
                     churn=("churn", "mean")).round(4)
    show("B6 time trend (head/tail)", pd.concat([trend.head(3), trend.tail(3)]).to_string())

    within = {s: round(float(g2.corr(numeric_only=True).loc["total_spend", "tenure_months"]), 3)
              for s, g2 in df.groupby("segment")}
    overall = round(float(df["total_spend"].corr(df["tenure_months"])), 3)
    show("B7 correlation overall vs within segment", {"overall": overall, "within": within})

    show("B8 figures", ["images/ch08_correlation_heatmap.png",
                         "images/ch08_pairplot_segment.png"])


# ===================================================== 8.7 multivariate
def section_87() -> None:
    df = eda()
    facet = df.groupby(["segment", "channel"]).agg(
        n=("customer_id", "size"), churn=("churn", "mean"),
        spend=("total_spend", "median")).round(3)
    show("M1 a three-variable facet", facet.head(9).to_string())

    enc = df.assign(
        size=np.where(df["total_spend"] > df["total_spend"].median(), "high", "low"),
        tone=np.where(df["churn"] == 1, "churn", "kept"),
    ).groupby(["segment", "size", "tone"]).size().rename("n").reset_index()
    show("M2 four variables in one table", enc.head(8).to_string(index=False))

    show("M3 pair plot figure", "images/ch08_pairplot_segment.png")

    cond = df.groupby("segment").apply(
        lambda g: pd.Series({
            "corr_spend_tenure": g["total_spend"].corr(g["tenure_months"]),
            "corr_tickets_churn": g["support_tickets"].corr(g["churn"]),
        }), include_groups=False).round(3)
    show("M4 conditional relationships", cond.to_string())

    simp = pd.read_csv(AUDIT / "simpson_segments.csv")
    simp["rate"] = simp["successes"] / simp["trials"]
    agg = simp.groupby("group")[["successes", "trials"]].sum()
    agg["rate"] = (agg["successes"] / agg["trials"]).round(4)
    show("M5 Simpson aggregate", agg.to_string())
    show("M5 Simpson within segment", simp[["segment", "group", "rate"]].round(4).to_string(index=False))

    num = df.select_dtypes(include=[np.number]).drop(columns=["churn"]).fillna(0)
    X = StandardScaler().fit_transform(num)
    pca = PCA(n_components=5).fit(X)
    show("M6 PCA variance explained", {
        "components": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "cumulative_3": round(float(pca.explained_variance_ratio_[:3].sum()), 4),
    })
    show("M7 Simpson figure", "images/ch08_simpson.png")


# =================================================== 8.8 group discipline
def section_88() -> None:
    df = eda()
    sizes = df.groupby("segment").size()
    show("G1 unequal group sizes", {"counts": sizes.to_dict(),
                                     "ratio_max_min": round(float(sizes.max() / sizes.min()), 3)})

    rng = np.random.default_rng(SEED8)
    p = rng.beta(1.2, 6, size=20)
    m = len(p)
    p_bonf = np.minimum(p * m, 1.0)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    p_bh = np.empty(m)
    p_bh[order] = np.minimum(adj, 1.0)
    show("G2 multiple comparisons", pd.DataFrame({
        "raw": p.round(4), "bonferroni": p_bonf.round(4), "bh": p_bh.round(4),
    }).head(6).to_string(index=False))
    show("G2 counts", {"raw_lt_05": int((p < 0.05).sum()),
                        "bonferroni_lt_05": int((p_bonf < 0.05).sum()),
                        "bh_lt_05": int((p_bh < 0.05).sum())})

    a = df.loc[df["segment"] == "Basic", "total_spend"]
    b = df.loc[df["segment"] == "Premium", "total_spend"]
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / pooled
    show("G3 effect size", {"cohens_d": round(float(d), 3),
                             "mean_difference": round(float(a.mean() - b.mean()), 2),
                             "pooled_sd": round(float(pooled), 2)})

    first = rng.normal(50, 10, 500)
    top = first >= np.quantile(first, 0.9)
    second = rng.normal(50, 10, 500)
    show("G4 regression to the mean", {
        "top_group_first_mean": round(float(first[top].mean()), 2),
        "top_group_second_mean": round(float(second[top].mean()), 2),
        "population_mean": 50.0,
    })

    sat = df["satisfaction"]
    show("G5 ceiling and floor", {"at_ceiling_5": int((sat == 5).sum()),
                                   "at_floor_1": int((sat == 1).sum()),
                                   "ceiling_share": round(float((sat == 5).mean()), 4)})


# ====================================================== 8.9 hypotheses
def section_89() -> None:
    findings = pd.DataFrame([
        ("F1", "Short tenure drives churn", "median 12 vs 45 months", "large", "survival model"),
        ("F2", "Tickets drive churn", "55% to 92%", "large", "hold tenure fixed"),
        ("F3", "Premium churns least", "51% vs 76%", "medium", "check plan mix"),
        ("F4", "Spend is bimodal", "two peaks", "medium", "fit a mixture"),
        ("F5", "A tiny group has extreme spend", "36 above p99", "small", "investigate rows"),
    ], columns=["id", "finding", "evidence", "magnitude", "confirm_or_kill"])
    show("F1 candidate findings", findings.to_string(index=False))

    scored = findings.assign(
        score=findings["magnitude"].map({"large": 3, "medium": 2, "small": 1})
        * 10 - findings["id"].str[1:].astype(int) / 10).sort_values(
            "score", ascending=False)
    show("F2 ranked by magnitude then id", scored[
        ["id", "finding", "score"]].to_string(index=False))

    show("F3 kill criteria", {
        "F1": "killed if the tenure effect vanishes within segment",
        "F2": "killed if tickets correlate with tenure only",
        "F3": "killed if premium customers have annual plans only",
    })

    artifacts = pd.DataFrame([
        ("profile", "one page per column", "eda_profile.md"),
        ("relationship map", "pairs worth a view", "eda_relationships.md"),
        ("findings table", "ranked with checks", "eda_findings.csv"),
        ("open questions", "what exploration cannot answer", "eda_questions.md"),
    ], columns=["artifact", "contents", "file"])
    show("F4 four artifacts", artifacts.to_string(index=False))


# ======================================================= 8.10 reporting
def section_810() -> None:
    df = eda()
    table = pd.DataFrame([
        ("median total spend", "USD", "per customer", "median", round(float(df["total_spend"].median()), 2)),
        ("churn rate", "share", "all customers", "mean of 0/1", round(float(df["churn"].mean()), 4)),
        ("median tenure", "months", "all customers", "median", float(df["tenure_months"].median())),
        ("income", "USD", "non-null only", "median", round(float(df["income"].median()), 2)),
    ], columns=["measure", "unit", "denominator", "statistic", "value"])
    show("R1 a table with units and denominators", table.to_string(index=False))

    defs = {
        "mean spend per customer": round(float(df["total_spend"].mean()), 2),
        "median spend per customer": round(float(df["total_spend"].median()), 2),
        "mean spend per order": round(float((df["total_spend"] / df["order_count"]).mean()), 2),
        "mean spend of Premium customers": round(float(
            df.loc[df["segment"] == "Premium", "total_spend"].mean()), 2),
    }
    show("R2 four definitions of average customer", defs)

    claimed = 185.78
    rebuilt = float(df["total_spend"].sum() / df["order_count"].sum())
    show("R3 reconstruct a published average", {
        "published": claimed, "rebuilt": round(rebuilt, 2),
        "difference": round(rebuilt - claimed, 2),
        "definition": "sum of line revenue / number of orders",
    })

    bad = pd.DataFrame([
        ("churn", "0.68", "average customer churns"),
        ("spend", "309.30", "average customer spends this"),
    ], columns=["metric", "value", "claim"])
    good = pd.DataFrame([
        ("churn rate", "0.679 (2,000 customers)", "share of customers flagged churn in the last 30 days"),
        ("median total spend", "$213.77 (IQR 105.10-380.03)", "per customer, right-skewed so median reported"),
    ], columns=["metric", "value_with_n", "definition"])
    show("R4 bad table", bad.to_string(index=False))
    show("R4 rewritten", good.to_string(index=False))

    show("R5 reporting checklist", [
        "name the statistic, the unit, the denominator and the n",
        "state the definition when the name is ambiguous",
        "pair a skewed mean with a median and a spread",
        "distinguish a finding from a hypothesis",
        "publish a figure when the summary could hide a shape",
    ])


# ======================================================= 8.11 example bank
def section_811() -> None:
    df = eda()
    show("E1 shape and grain", {"rows": len(df), "columns": len(df.columns),
                                 "unique_id": bool(df["customer_id"].is_unique)})
    show("E2 missing ranking", (df.isna().mean().sort_values(ascending=False).head(3) * 100).round(2).to_dict())
    show("E3 skew ranking", df.select_dtypes(include=[np.number]).skew().abs().sort_values(
        ascending=False).head(4).round(3).to_dict())
    show("E4 constant columns", int((df.nunique(dropna=True) <= 1).sum()))
    show("E5 high cardinality", {c: int(df[c].nunique())
                                 for c in df.select_dtypes(include="object").columns
                                 if df[c].nunique() > 20})
    show("E6 outlier sweep", {c: int((df[c] > df[c].quantile(0.99)).sum())
                              for c in ["total_spend", "avg_order_value", "latency_ms"]})


# ========================================================= 8.12 capstone
def section_812() -> None:
    df = eda()
    q = df["total_spend"].quantile([0.25, 0.5, 0.75])
    summary = pd.DataFrame([
        ("rows", len(df), "", ""),
        ("median spend", round(float(q.iloc[1]), 2), "USD", "per customer"),
        ("IQR", round(float(q.iloc[2] - q.iloc[0]), 2), "USD", "25th to 75th percentile"),
        ("churn rate", round(float(df["churn"].mean()), 4), "share", "all customers"),
        ("income median", round(float(df["income"].median()), 2), "USD", "non-null only"),
    ], columns=["measure", "value", "unit", "basis"])
    show("K1 headline table", summary.to_string(index=False))

    ranked = pd.DataFrame([
        ("1", "tenure and churn", "median 12 vs 45 months", "confirm within segment"),
        ("2", "support tickets and churn", "55% to 92%", "hold tenure fixed"),
        ("3", "spend bimodality", "two peaks", "segment by plan"),
    ], columns=["rank", "finding", "evidence", "check"])
    show("K2 ranked findings", ranked.to_string(index=False))

    show("K3 control totals", {
        "customers": len(df),
        "sum_spend": round(float(df["total_spend"].sum()), 2),
        "churned": int(df["churn"].sum()),
        "income_null": int(df["income"].isna().sum()),
    })

    show("K4 four artifacts", ["eda_profile.md", "eda_relationships.md",
                                "eda_findings.csv", "eda_questions.md"])


def section_extra() -> None:
    """Extra worked blocks for the expanded Chapter 8 sections."""
    df = eda()
    rev = df["total_spend"]

    show("X1 trimmed means", {f"{int(p * 100)}pct": round(float(stats.trim_mean(rev, p)), 2)
                               for p in [0.0, 0.01, 0.05, 0.10, 0.20]})

    counts = df["region"].value_counts()
    pareto = pd.DataFrame({"count": counts,
                           "pct": (counts / len(df) * 100).round(2),
                           "cumulative": (counts.cumsum() / len(df) * 100).round(2)})
    show("X2 cumulative region frequencies", pareto.to_string())

    a = df.loc[df["segment"] == "Basic", "churn"]
    b = df.loc[df["segment"] == "Premium", "churn"]
    diff = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    show("X3 practical vs statistical", {
        "difference": round(diff, 4), "standard_error": round(se, 4),
        "interval_low": round(diff - 1.96 * se, 4),
        "interval_high": round(diff + 1.96 * se, 4),
        "practical_threshold": 0.05, "exceeds_threshold": bool(diff > 0.05),
    })

    show("X4 numeric columns", int(df.select_dtypes(include=[np.number]).shape[1]))
    show("X5 top correlation", round(float(df[["total_spend", "avg_order_value"]].corr().iloc[0, 1]), 3))
    show("X6 satisfaction ceiling", {"at_5": int((df["satisfaction"] == 5).sum()),
                                      "at_4_or_5": int((df["satisfaction"] >= 4).sum())})
    q1, q3 = [float(v) for v in df["avg_order_value"].quantile([0.25, 0.75])]
    hi = q3 + 1.5 * (q3 - q1)
    show("X7 aov fence", {"upper": round(hi, 2),
                           "flagged": int((df["avg_order_value"] > hi).sum())})
    show("X8 mean/median ratio by segment", {
        s: round(float(sub["total_spend"].mean() / sub["total_spend"].median()), 3)
        for s, sub in df.groupby("segment")})
    show("X9 churn by channel", df.groupby("channel").agg(
        n=("customer_id", "size"), churn=("churn", "mean")).round(4).to_string())
    dates = pd.to_datetime(df["signup_date"])
    show("X10 date span", {"min": str(dates.min().date()),
                            "max": str(dates.max().date()),
                            "days": int((dates.max() - dates.min()).days)})


def main() -> None:
    print("python", "pandas", pd.__version__, "numpy", np.__version__)
    print()
    for section in (
        section_81, section_82, section_83, section_84, section_85, section_86,
        section_87, section_88, section_89, section_810, section_811, section_812,
        section_extra,
    ):
        print("=" * 78)
        section()


if __name__ == "__main__":
    main()
