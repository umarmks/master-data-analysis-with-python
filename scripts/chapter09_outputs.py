#!/usr/bin/env python3
"""Print every output block used in Chapter 9, from the current files.

Run from the project root:

    python scripts/chapter09_outputs.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "raw" / "audit"
SEED = 909

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 80)


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def ab() -> pd.DataFrame:
    return pd.read_csv(AUDIT / "ab_checkout.csv")


def values() -> np.ndarray:
    return pd.read_csv(AUDIT / "order_values_ch09.csv")["order_value"].to_numpy()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return center - half, center + half


def two_prop(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    return p1, p2, z, 2 * (1 - stats.norm.cdf(abs(z)))


def required_n(p1, lift, alpha=0.05, power=0.80):
    p2 = p1 + lift
    pbar = (p1 + p2) / 2
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    num = (z_a * np.sqrt(2 * pbar * (1 - pbar))
           + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
    return int(np.ceil(num / lift**2))


# ================================================= 9.1 probability foundations
def section_91() -> None:
    space = list(product([0, 1], repeat=2))
    show("P1 sample space of two flips", {"outcomes": space,
                                           "count": len(space)})

    df = ab()
    a = df["converted"].eq(1)
    b = df["device"].eq("Desktop")
    show("P2 complement and addition", {
        "P(converted)": round(float(a.mean()), 4),
        "P(not converted)": round(float((~a).mean()), 4),
        "sum": round(float(a.mean() + (~a).mean()), 4),
        "P(converted or Desktop)": round(float((a | b).mean()), 4),
        "P(converted) + P(Desktop) - P(both)": round(float(
            a.mean() + b.mean() - (a & b).mean()), 4),
    })

    show("P3 conditional probability", {
        "P(converted | Desktop)": round(float(a[b].mean()), 4),
        "P(converted | not Desktop)": round(float(a[~b].mean()), 4),
        "P(Desktop | converted)": round(float(b[a].mean()), 4),
    })

    joint = pd.crosstab(df["device"], df["converted"], normalize="all").round(4)
    show("P4 joint probabilities as a tree", joint.to_string())

    da = df["variant"].eq("treatment")
    pa, pb = float(da.mean()), float(a.mean())
    show("P5 independence check", {
        "P(treatment)": round(pa, 4), "P(converted)": round(pb, 4),
        "product": round(pa * pb, 5),
        "P(both)": round(float((da & a).mean()), 5),
        "ratio": round(float((da & a).mean() / (pa * pb)), 4),
    })

    c = df["region"].eq("West")
    show("P6 inclusion-exclusion for three events", {
        "P(A or B or C)": round(float((a | b | c).mean()), 4),
        "formula": round(float(a.mean() + b.mean() + c.mean()
                               - (a & b).mean() - (a & c).mean()
                               - (b & c).mean() + (a & b & c).mean()), 4),
    })


# ============================================================= 9.2 Bayes
def section_92() -> None:
    prev, sens, spec = 0.005, 0.95, 0.99
    ppos = sens * prev + (1 - spec) * (1 - prev)
    post = sens * prev / ppos
    show("B1 fraud test posterior", {
        "prevalence": prev, "sensitivity": sens, "specificity": spec,
        "P(positive)": round(float(ppos), 5),
        "P(fraud | positive)": round(float(post), 4),
    })

    n = 10000
    fraud = int(n * prev)
    tp = int(round(fraud * sens))
    fp = int(round((n - fraud) * (1 - spec)))
    show("B2 natural frequencies", pd.DataFrame([
        ("fraudster", tp, fraud - tp, fraud),
        ("honest", fp, n - fraud - fp, n - fraud),
    ], columns=["group", "positive", "negative", "total"]).to_string(index=False))
    show("B2 posterior from counts", round(tp / (tp + fp), 4))

    prev2 = 0.02
    ppos2 = sens * prev2 + (1 - spec) * (1 - prev2)
    show("B3 higher base rate", {
        "prevalence": prev2,
        "P(fraud | positive)": round(float(sens * prev2 / ppos2), 4),
    })

    show("B4 P(A|B) is not P(B|A)", {
        "P(positive | fraud)": sens,
        "P(fraud | positive)": round(float(post), 4),
        "ratio": round(sens / post, 2),
    })

    post2 = post
    for _ in range(2):
        num = sens * post2
        den = sens * post2 + (1 - spec) * (1 - post2)
        post2 = num / den
    show("B5 two independent positives", round(float(post2), 4))

    show("B6 base-rate neglect", {
        "naive_guess": 0.95,
        "correct": round(float(post), 4),
        "error_factor": round(0.95 / post, 1),
    })


# ================================================= 9.3 random variables
def section_93() -> None:
    p = 0.1267
    show("D1 Bernoulli", {"mean": p, "variance": round(p * (1 - p), 5)})

    k = np.arange(0, 101)
    pmf = stats.binom.pmf(k, 100, p)
    show("D2 binomial", {
        "P(10 conversions in 100)": round(float(stats.binom.pmf(10, 100, p)), 4),
        "P(at most 10)": round(float(stats.binom.cdf(10, 100, p)), 4),
        "expected conversions": round(100 * p, 2),
        "sd": round(float(np.sqrt(100 * p * (1 - p))), 2),
        "pmf_sum": round(float(pmf.sum()), 4),
    })

    lam = 0.9
    show("D3 Poisson", {
        "P(0 tickets)": round(float(stats.poisson.pmf(0, lam)), 4),
        "P(3 or more)": round(float(1 - stats.poisson.cdf(2, lam)), 4),
        "mean": lam, "variance": lam,
    })

    show("D4 uniform", {
        "mean": 50.0, "variance": round(100**2 / 12, 3),
        "P(between 20 and 30)": round(float(
            stats.uniform.cdf(30, 0, 100) - stats.uniform.cdf(20, 0, 100)), 3),
    })

    show("D5 exponential", {
        "mean": 5.0, "sd": 5.0,
        "P(wait more than 10)": round(float(stats.expon.sf(10, scale=5)), 4),
    })

    v = values()
    show("D6 normal vs lognormal", {
        "mean": round(float(v.mean()), 2),
        "median": round(float(np.median(v)), 2),
        "log_mean": round(float(np.log(v).mean()), 3),
        "log_sd": round(float(np.log(v).std()), 3),
        "skew": round(float(stats.skew(v)), 3),
    })

    show("D7 expectation from simulation", {
        "theoretical_mean": round(float(np.exp(4.1 + 0.75**2 / 2)), 2),
        "sample_mean": round(float(v.mean()), 2),
        "theoretical_median": round(float(np.exp(4.1)), 2),
        "sample_median": round(float(np.median(v)), 2),
    })


# ======================================================== 9.4 simulation
def section_94() -> None:
    r1 = np.random.default_rng(SEED)
    r2 = np.random.default_rng(SEED)
    show("S1 reproducibility", {
        "run1": [round(float(x), 4) for x in r1.random(3)],
        "run2": [round(float(x), 4) for x in r2.random(3)],
        "global_seed_result": [round(float(np.random.rand()), 4)],
    })
    np.random.seed(SEED)
    show("S1 global seed again", [round(float(np.random.rand()), 4)])

    n_people, trials = 23, 20000
    rng = np.random.default_rng(SEED)
    matches = np.array([len(set(rng.integers(0, 365, size=n_people))) < n_people
                        for _ in range(1000)])
    show("S2 birthday problem", {
        "people": n_people, "estimated_probability": round(float(matches.mean()), 4),
        "shared_birthday_trials": int(matches.sum()), "trials": 1000,
    })

    waits = np.cumsum(rng.exponential(5, size=1000))
    show("S3 simple queue", {
        "arrivals": 1000, "mean_interarrival": round(float(np.diff(waits).mean()), 2),
        "max_wait": round(float(waits[-1]), 1),
    })

    n = 2000
    ctrl = rng.binomial(1, 0.12, size=n)
    trt = rng.binomial(1, 0.135, size=n)
    p1, p2, z, pval = two_prop(ctrl.sum(), n, trt.sum(), n)
    show("S4 promotion simulation", {
        "control_rate": round(float(p1), 4),
        "treatment_rate": round(float(p2), 4),
        "lift_pp": round(float((p2 - p1) * 100), 2),
        "z": round(float(z), 3), "p": round(float(pval), 4),
    })

    pts = rng.random((300000, 2))
    inside = (pts[:, 0]**2 + pts[:, 1]**2 <= 1).mean()
    show("S5 Monte Carlo pi", {"estimate": round(float(4 * inside), 4),
                                "truth": round(float(np.pi), 4)})

    p_a, p_b, p_c = 0.05, 0.06, 0.07
    hits = ((rng.random(200000) < p_a) &
            (rng.random(200000) < p_b) &
            (rng.random(200000) < p_c)).mean()
    show("S6 hard-to-compute event", {
        "trials": 200000,
        "simulated_probability": round(float(hits), 5),
        "product_if_independent": round(p_a * p_b * p_c, 5),
    })


# ========================================================= 9.5 sampling
def section_95() -> None:
    frame = pd.read_csv(AUDIT / "sampling_frame.csv")
    truth = float(frame["value"].mean())
    rng = np.random.default_rng(SEED)
    srs = frame.sample(500, random_state=SEED)
    show("SM1 population and SRS", {
        "population_mean": round(truth, 2),
        "srs_mean": round(float(srs["value"].mean()), 2),
        "error": round(float(srs["value"].mean() - truth), 2),
    })

    step = len(frame) // 500
    systematic = frame.iloc[np.arange(0, len(frame), step)[:500]]
    show("SM2 systematic", {"n": len(systematic),
                             "mean": round(float(systematic["value"].mean()), 2)})

    strata = []
    for seg, grp in frame.groupby("segment"):
        take = int(round(500 * len(grp) / len(frame)))
        strata.append(grp.sample(take, random_state=SEED))
    strat = pd.concat(strata)
    show("SM3 stratified", {
        "n": len(strat), "mean": round(float(strat["value"].mean()), 2),
        "allocation": frame["segment"].value_counts(normalize=True).round(3).to_dict(),
    })

    clusters = frame.sample(20, random_state=SEED)["segment"].tolist()
    cl = frame[frame["segment"].isin(clusters)].sample(
        500, random_state=SEED)
    show("SM4 cluster", {"clusters": len(set(clusters)),
                          "n": len(cl), "mean": round(float(cl["value"].mean()), 2)})

    respondents = frame[frame["responds"] == 1]
    show("SM5 self-selection", {
        "respondents": len(respondents),
        "respondent_mean": round(float(respondents["value"].mean()), 2),
        "population_mean": round(truth, 2),
        "bias": round(float(respondents["value"].mean() - truth), 2),
    })

    covered = frame[frame["in_frame"] == 1]
    show("SM6 coverage error", {
        "covered_share": round(float(frame["in_frame"].mean()), 4),
        "covered_mean": round(float(covered["value"].mean()), 2),
        "bias": round(float(covered["value"].mean() - truth), 2),
    })

    bias = pd.DataFrame([
        ("coverage", "frame excludes 8% of customers", "negatively biased"),
        ("self-selection", "high-value customers respond more", "positively biased"),
        ("nonresponse", "busy customers skip the survey", "direction unknown"),
        ("convenience", "only one store sampled", "unknown"),
        ("measurement", "self-reported income", "toward the middle"),
    ], columns=["source", "mechanism", "direction"])
    show("SM7 bias catalogue", bias.to_string(index=False))

    show("SM8 figures", ["images/ch09_sampling_bias.png"])


# ================================================ 9.6 sampling distributions
def section_96() -> None:
    v = values()
    rng = np.random.default_rng(SEED)
    n = 30
    means = np.array([rng.choice(v, size=n, replace=True).mean()
                      for _ in range(10000)])
    show("SD1 sampling distribution", {
        "samples": 10000, "n": n,
        "mean_of_means": round(float(means.mean()), 3),
        "sd_of_means": round(float(means.std()), 3),
        "theoretical_se": round(float(v.std() / np.sqrt(n)), 3),
    })

    shapes = {}
    for k in [5, 30, 100]:
        m = np.array([rng.choice(v, size=k, replace=True).mean()
                      for _ in range(5000)])
        shapes[k] = round(float(stats.skew(m)), 3)
    show("SD2 CLT by sample size", {"population_skew": round(float(stats.skew(v)), 3),
                                     "skew_of_sample_means": shapes})

    show("SD3 standard error check", {
        "population_sd": round(float(v.std()), 3),
        "n": 30,
        "se": round(float(v.std() / np.sqrt(30)), 3),
    })

    N = 50000
    fpc = np.sqrt((N - 30) / (N - 1))
    show("SD4 finite population correction", {
        "fpc": round(float(fpc), 5),
        "se_with_fpc": round(float(v.std() / np.sqrt(30) * fpc), 3),
    })

    boots = np.median(rng.choice(v, size=(4000, len(v)), replace=True), axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    show("SD5 bootstrap median", {
        "median": round(float(np.median(v)), 3),
        "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3),
        "bootstrap_sd": round(float(boots.std()), 3),
    })

    small = np.sort(v)[:20]
    extreme_boots = np.max(rng.choice(small, size=(1000, 20), replace=True), axis=1)
    show("SD6 bootstrap failure", {
        "true_max": round(float(small.max()), 2),
        "bootstrap_max_mean": round(float(extreme_boots.mean()), 2),
        "share_equal_truth": round(float((extreme_boots == small.max()).mean()), 3),
    })

    show("SD7 figures", ["images/ch09_sampling_distribution.png",
                          "images/ch09_bootstrap_median.png"])


# ============================================ 9.7 estimation and intervals
def section_97() -> None:
    v = values()
    n = len(v)
    mean, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(n))
    tcrit = stats.t.ppf(0.975, n - 1)
    show("CI1 t-interval for the mean", {
        "mean": round(mean, 3), "se": round(se, 3),
        "t_critical": round(float(tcrit), 3),
        "low": round(float(mean - tcrit * se), 3),
        "high": round(float(mean + tcrit * se), 3),
    })

    k = int(ab()["converted"].sum())
    N = len(ab())
    lo, hi = wilson(k, N)
    show("CI2 Wilson interval", {
        "conversions": k, "n": N, "rate": round(float(k / N), 4),
        "low": round(float(lo), 4), "high": round(float(hi), 4),
        "naive_low": round(float(k / N - 1.96 * np.sqrt((k / N) * (1 - k / N) / N)), 4),
    })

    d = ab()
    c = d[d["variant"] == "control"]
    t = d[d["variant"] == "treatment"]
    p1, p2, z, pv = two_prop(c["converted"].sum(), len(c),
                             t["converted"].sum(), len(t))
    se_diff = np.sqrt(p1 * (1 - p1) / len(c) + p2 * (1 - p2) / len(t))
    show("CI3 difference in proportions", {
        "control": round(float(p1), 4), "treatment": round(float(p2), 4),
        "difference": round(float(p2 - p1), 4),
        "interval_low": round(float(p2 - p1 - 1.96 * se_diff), 4),
        "interval_high": round(float(p2 - p1 + 1.96 * se_diff), 4),
        "p": round(float(pv), 4),
    })

    x = c["revenue"].to_numpy()
    y = t["revenue"].to_numpy()
    res = stats.ttest_ind(y, x, equal_var=False)
    sed = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    diff = y.mean() - x.mean()
    show("CI4 Welch difference in means", {
        "difference": round(float(diff), 3),
        "se": round(float(sed), 3),
        "interval": [round(float(diff - 1.96 * sed), 3),
                     round(float(diff + 1.96 * sed), 3)],
        "p": round(float(res.pvalue), 4),
    })

    rng = np.random.default_rng(SEED)
    boots = np.median(rng.choice(v, size=(3000, n), replace=True), axis=1)
    show("CI5 bootstrap median", {
        "median": round(float(np.median(v)), 3),
        "interval": [round(float(x), 3) for x in np.quantile(boots, [0.025, 0.975])],
    })

    show("CI6 precision planning", {
        "baseline": 0.12, "target_margin_pp": 1.0,
        "n_for_95pct": required_n(0.12, 0.01),
        "n_for_99pct": required_n(0.12, 0.01, alpha=0.01),
    })

    ratio_boots = []
    for _ in range(1000):
        s = rng.choice(v, size=n, replace=True)
        ratio_boots.append(s.sum() / max((s > 0).sum(), 1))
    show("CI7 ratio via bootstrap", {
        "point": round(float(v.sum() / (v > 0).sum()), 3),
        "label": "revenue per converting order",
        "interval": [round(float(q), 3) for q in np.quantile(ratio_boots, [0.025, 0.975])],
    })

    cover = []
    rng2 = np.random.default_rng(SEED + 1)
    pop = rng2.normal(50, 10, size=10000)
    for _ in range(1000):
        s = rng2.choice(pop, size=30, replace=True)
        lo_s, hi_s = s.mean() - 1.96 * s.std(ddof=1) / np.sqrt(30), \
                     s.mean() + 1.96 * s.std(ddof=1) / np.sqrt(30)
        cover.append(lo_s <= pop.mean() <= hi_s)
    show("CI8 coverage simulation", {"intervals": 1000,
                                      "coverage": round(float(np.mean(cover)), 4)})
    show("CI9 figure", "images/ch09_intervals.png")


# ================================================== 9.8 hypothesis testing
def section_98() -> None:
    v = values()
    res = stats.ttest_1samp(v, 85)
    show("H1 one-sample t-test", {
        "mean": round(float(v.mean()), 3), "hypothesised": 85,
        "t": round(float(res.statistic), 3), "p": round(float(res.pvalue), 5),
    })

    rng = np.random.default_rng(SEED)
    null = np.array([rng.normal(0, 1, size=30).mean() * np.sqrt(30)
                     for _ in range(20000)])
    observed = abs(float(res.statistic))
    show("H2 p-value by simulation", {
        "observed_abs_t": round(observed, 3),
        "simulated_two_sided_p": round(float((np.abs(null) >= observed).mean()), 5),
        "formula_p": round(float(res.pvalue), 5),
    })

    show("H3 one-sided vs two-sided", {
        "two_sided": round(float(res.pvalue), 5),
        "one_sided_greater": round(float(stats.t.sf(res.statistic, len(v) - 1)), 5),
    })

    false = []
    for i in range(2000):
        s = rng.normal(100, 15, size=30)
        false.append(stats.ttest_1samp(s, 100).pvalue < 0.05)
    show("H4 Type I error simulation", {
        "rejections": int(np.sum(false)), "runs": 2000,
        "rate": round(float(np.mean(false)), 4),
    })

    p1, lift = 0.12, 0.013
    n = 6000
    p2 = p1 + lift
    se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    zc = stats.norm.ppf(0.975)
    power = stats.norm.cdf(lift / se - zc) + stats.norm.cdf(-lift / se - zc)
    show("H5 power", {
        "baseline": p1, "lift": lift, "n_per_arm": n,
        "se": round(float(se), 5), "power": round(float(power), 4),
    })

    p = np.array([0.001, 0.008, 0.019, 0.033, 0.044, 0.061, 0.075, 0.09,
                  0.12, 0.16, 0.21, 0.27, 0.33, 0.40, 0.48, 0.55, 0.63,
                  0.71, 0.82, 0.94])
    m = len(p)
    bonf = np.minimum(p * m, 1.0)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    bh = np.empty(m)
    bh[order] = np.minimum(adj, 1.0)
    show("H6 multiple comparisons", {
        "raw_lt_05": int((p < 0.05).sum()),
        "bonferroni_lt_05": int((bonf < 0.05).sum()),
        "bh_lt_05": int((bh < 0.05).sum()),
    })

    show("H7 practical significance", {
        "detected_lift_pp": round(lift * 100, 2),
        "business_threshold_pp": 2.0,
        "actionable": bool(lift * 100 >= 2.0),
    })


# ============================================== 9.9 choosing tests
def section_99() -> None:
    tree = pd.DataFrame([
        ("one numeric vs constant", "one-sample t"),
        ("two paired numeric", "paired t / Wilcoxon"),
        ("two independent numeric", "Welch t / Mann-Whitney"),
        ("3+ independent groups", "ANOVA / Kruskal-Wallis"),
        ("two categorical", "chi-square / Fisher"),
        ("binary outcome vs binary group", "two-proportion z"),
    ], columns=["question", "default_test"])
    show("T1 test selection", tree.to_string(index=False))

    pair = pd.read_csv(AUDIT / "paired_campaign.csv")
    d = pair["spend_after"] - pair["spend_before"]
    t = stats.ttest_rel(pair["spend_after"], pair["spend_before"])
    w = stats.wilcoxon(pair["spend_after"], pair["spend_before"])
    show("T2 paired tests", {
        "mean_difference": round(float(d.mean()), 3),
        "t": round(float(t.statistic), 3),
        "t_p": float(f"{t.pvalue:.3g}"),
        "wilcoxon_p": float(f"{w.pvalue:.3g}"),
    })

    abdf = ab()
    x = abdf.loc[abdf["variant"] == "control", "revenue"]
    y = abdf.loc[abdf["variant"] == "treatment", "revenue"]
    welch = stats.ttest_ind(y, x, equal_var=False)
    student = stats.ttest_ind(y, x, equal_var=True)
    show("T3 Welch vs Student", {
        "welch_t": round(float(welch.statistic), 4),
        "welch_p": round(float(welch.pvalue), 4),
        "student_t": round(float(student.statistic), 4),
        "student_p": round(float(student.pvalue), 4),
    })

    mw = stats.mannwhitneyu(y, x, alternative="two-sided")
    show("T4 Mann-Whitney", {"statistic": round(float(mw.statistic), 1),
                              "p": round(float(mw.pvalue), 4)})

    show("T5 Wilcoxon on pairs", {"p": float(f"{w.pvalue:.3g}")})

    groups = [pair.loc[pair["segment"] == s, "spend_after"]
              for s in pair["segment"].unique()]
    kw = stats.kruskal(*groups)
    f = stats.f_oneway(*groups)
    show("T6 Kruskal-Wallis and ANOVA", {
        "kw_h": round(float(kw.statistic), 3),
        "kw_p": round(float(kw.pvalue), 4),
        "anova_f": round(float(f.statistic), 3),
        "anova_p": round(float(f.pvalue), 4),
    })

    tab = pd.crosstab(abdf["device"], abdf["converted"])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    odd, fisher_p = stats.fisher_exact(tab.iloc[:2, :])
    show("T7 chi-square and Fisher", {
        "chi2": round(float(chi2), 3), "p": round(float(p), 4),
        "fisher_odds_ratio": round(float(odd), 3),
        "fisher_p": round(float(fisher_p), 4),
    })

    pooled = np.sqrt(((len(x) - 1) * x.var(ddof=1)
                      + (len(y) - 1) * y.var(ddof=1))
                     / (len(x) + len(y) - 2))
    dv = float((y.mean() - x.mean()) / pooled)
    show("T8 Cohen's d", {
        "cohens_d": round(dv, 4),
        "mean_difference": round(float(y.mean() - x.mean()), 3),
        "pooled_sd": round(float(pooled), 3),
    })

    show("T9 Cramer's V", {
        "cramers_v": round(float(np.sqrt(chi2 / (tab.to_numpy().sum()))), 4),
    })


# ============================================== 9.10 experiments
def section_910() -> None:
    d = ab()
    c = d[d["variant"] == "control"]
    t = d[d["variant"] == "treatment"]
    p1, p2, z, pv = two_prop(c["converted"].sum(), len(c),
                             t["converted"].sum(), len(t))
    show("E1 conversion result", {
        "control_n": len(c), "control_rate": round(float(p1), 4),
        "treatment_n": len(t), "treatment_rate": round(float(p2), 4),
        "absolute_lift_pp": round(float((p2 - p1) * 100), 3),
        "relative_lift_pct": round(float((p2 / p1 - 1) * 100), 2),
        "z": round(float(z), 3), "p": round(float(pv), 4),
    })

    g1, g2 = c["support_contact"].mean(), t["support_contact"].mean()
    show("E2 guardrail", {
        "control_support_rate": round(float(g1), 4),
        "treatment_support_rate": round(float(g2), 4),
        "difference": round(float(g2 - g1), 4),
    })

    share = len(t) / len(d)
    show("E3 sample ratio mismatch", {
        "treatment_share": round(float(share), 4),
        "expected": 0.5, "z": round(float(
            (len(t) - len(d) / 2) / np.sqrt(len(d) * 0.25)), 3),
    })

    show("E4 required sample size", {
        "baseline": 0.12, "lift_2pp_n_per_arm": required_n(0.12, 0.02),
        "lift_1pp_n_per_arm": required_n(0.12, 0.01),
        "lift_05pp_n_per_arm": required_n(0.12, 0.005),
    })

    rng = np.random.default_rng(SEED)
    false_at_looks = []
    for looks in [1, 5, 20]:
        flags = 0
        for _ in range(1000):
            seq = rng.binomial(1, 0.12, size=2000)
            for k in np.linspace(100, 2000, looks).astype(int):
                if stats.binomtest(seq[:k].sum(), k, 0.12).pvalue < 0.05:
                    flags += 1
                    break
        false_at_looks.append((looks, flags / 1000))
    show("E5 peeking", pd.DataFrame(
        false_at_looks, columns=["looks", "false_positive_rate"]).round(4).to_string(index=False))

    show("E6 causal caution", [
        "randomisation makes arms exchangeable; an observational comparison does not",
        "a post-hoc segment that wins may be noise from many looks",
        "difference-in-differences needs a parallel pre-trend",
        "propensity scores cannot balance an unmeasured confounder",
    ])
    show("E7 figure", "images/ch09_power_curve.png")


# ================================================= 9.11 reporting inference
def section_911() -> None:
    show("R1 reporting template", [
        "estimate with unit", "95% interval", "n and method",
        "effect size", "decision and guardrails",
    ])

    bad = pd.DataFrame([
        ("lift was significant", "p = 0.04", "roll out"),
        ("conversion improved", "+20%", "looks good"),
    ], columns=["claim", "number", "decision"])
    good = pd.DataFrame([
        ("absolute conversion lift 1.3 pp (95% CI 0.2 to 2.4), "
         "two-proportion z, n = 6,000 per arm, relative lift 10.8%",
         "support-contact guardrail unchanged", "roll out to 100%"),
        ("from 12.0% to 13.3%, p = 0.04, not corrected for multiple looks",
         "single pre-registered primary metric", "hold until replicated"),
    ], columns=["estimate", "guardrail", "decision"])
    show("R2 bad examples", bad.to_string(index=False))
    show("R2 rewritten", good.to_string(index=False))

    checks = pd.DataFrame([
        ("1", "absolute lift and interval", "20% relative may be 0.2 pp"),
        ("2", "n and power", "180 per arm may be underpowered"),
        ("3", "multiple looks", "p = 0.04 after many peeks is not 0.04"),
        ("4", "randomisation and guardrails", "check sample ratio and harm"),
        ("5", "practical threshold and costs", "is the lift worth the change"),
    ], columns=["check", "question", "why"])
    show("R3 checkpoint checks", checks.to_string(index=False))
    show("R4 decision memo", {
        "recommendation": "do not roll out on this evidence",
        "reason": "relative lift hides absolute size; n is small; p is uncorrected",
        "next_step": "re-run with pre-registered metric, planned n, and one look",
    })


# ======================================================= 9.12 example bank
def section_912() -> None:
    d = ab()
    show("X1 arm balance", d["variant"].value_counts().to_dict())
    show("X2 conversion by arm", d.groupby("variant")["converted"].mean().round(4).to_dict())
    show("X3 revenue by arm", d.groupby("variant")["revenue"].mean().round(2).to_dict())
    show("X4 support by arm", d.groupby("variant")["support_contact"].mean().round(4).to_dict())
    show("X5 date coverage", {"days": int(d["exposure_date"].nunique()),
                               "min": d["exposure_date"].min(),
                               "max": d["exposure_date"].max()})
    show("X6 devices", d["device"].value_counts().to_dict())


# ========================================================= 9.13 capstone
def section_913() -> None:
    d = ab()
    c = d[d["variant"] == "control"]
    t = d[d["variant"] == "treatment"]
    p1, p2, z, pv = two_prop(c["converted"].sum(), len(c),
                             t["converted"].sum(), len(t))
    lo, hi = wilson(t["converted"].sum(), len(t))
    summary = pd.DataFrame([
        ("control conversion", f"{p1:.4f}", "share", f"n={len(c)}"),
        ("treatment conversion", f"{p2:.4f}", "share", f"n={len(t)}"),
        ("absolute lift", f"{p2 - p1:.4f}", "share", "difference"),
        ("relative lift", f"{(p2 / p1 - 1) * 100:.1f}%", "percent", "ratio"),
        ("p value", f"{pv:.4f}", "probability", "two-sided z"),
        ("treatment Wilson CI", f"{lo:.4f}-{hi:.4f}", "share", "95%"),
    ], columns=["measure", "value", "unit", "basis"])
    show("K1 experiment summary", summary.to_string(index=False))

    show("K2 decision inputs", {
        "primary_metric": "checkout conversion",
        "guardrail": "support contact rate",
        "n_per_arm": len(c),
        "power_at_observed_effect": round(float(
            stats.norm.cdf(abs(z) - stats.norm.ppf(0.975))), 4),
        "sample_ratio_ok": bool(abs(len(c) - len(t)) < 3 * np.sqrt(len(d) * 0.25)),
    })

    show("K3 control totals", {
        "users": len(d), "conversions": int(d["converted"].sum()),
        "revenue": round(float(d["revenue"].sum()), 2),
        "support_contacts": int(d["support_contact"].sum()),
    })

    show("K4 artifacts", ["ab_analysis.csv", "power_analysis.csv",
                          "decision_memo.md", "inference_report.md"])


def main() -> None:
    print("python", "pandas", pd.__version__, "numpy", np.__version__)
    print()
    for section in (
        section_91, section_92, section_93, section_94, section_95, section_96,
        section_97, section_98, section_99, section_910, section_911,
        section_912, section_913,
    ):
        print("=" * 78)
        section()


if __name__ == "__main__":
    main()
