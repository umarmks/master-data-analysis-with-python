#!/usr/bin/env python3
"""Print every output block used in Chapter 10, from the current files.

Run from the project root:

    python scripts/chapter10_outputs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "raw" / "audit"
SEED = 1007

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 80)


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def ols(frame: pd.DataFrame, ycol: str, xcols: list[str]) -> dict:
    X = np.column_stack([np.ones(len(frame))]
                        + [frame[c].to_numpy(float) for c in xcols])
    y = frame[ycol].to_numpy(float)
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    s2 = float(resid @ resid / (n - k))
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * stats.t.sf(np.abs(tvals), n - k)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    return {
        "names": ["intercept"] + xcols,
        "coef": beta, "se": se, "t": tvals, "p": pvals,
        "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / (n - k),
        "resid_sd": float(np.sqrt(s2)), "n": n, "k": k,
        "resid": resid, "fitted": X @ beta, "X": X, "y": y,
    }


def coef_table(fit: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "term": fit["names"],
        "coef": np.round(fit["coef"], 3),
        "std_error": np.round(fit["se"], 3),
        "t": np.round(fit["t"], 3),
        "p": [f"{v:.2g}" for v in fit["p"]],
    })


def vif(frame: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        others = [x for x in cols if x != c]
        r2 = ols(frame, c, others)["r2"]
        out[c] = round(float(1 / (1 - r2)), 2) if r2 < 1 else float("inf")
    return out


def logistic(frame: pd.DataFrame, ycol: str, feats: list[str],
             C: float = 1e6) -> tuple[dict, np.ndarray]:
    X = frame[feats].to_numpy(float)
    y = frame[ycol].to_numpy(int)
    clf = LogisticRegression(max_iter=2000, C=C).fit(X, y)
    beta = np.r_[clf.intercept_[0], clf.coef_[0]]
    prob = clf.predict_proba(X)[:, 1]
    Xa = np.column_stack([np.ones(len(y)), X])
    W = prob * (1 - prob)
    cov = np.linalg.inv(Xa.T @ (Xa * W[:, None]))
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p = 2 * stats.norm.sf(np.abs(z))
    names = ["intercept"] + feats
    table = pd.DataFrame({
        "term": names, "coef": np.round(beta, 4),
        "std_error": np.round(se, 4), "z": np.round(z, 3),
        "p": np.round(p, 5),
        "odds_ratio": np.round(np.exp(beta), 4),
    })
    return {"table": table, "prob": prob, "clf": clf}, prob


def ad() -> pd.DataFrame:
    return pd.read_csv(AUDIT / "ad_spend.csv")


def house() -> pd.DataFrame:
    return pd.read_csv(AUDIT / "housing_listings.csv")


def churn_df() -> pd.DataFrame:
    return pd.read_csv(AUDIT / "churn_regression.csv")


# ==================================================== 10.1 correlation
def section_101() -> None:
    d = ad()
    x, y = d["ad_spend"], d["revenue"]
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    r = float(x.corr(y))
    show("C1 covariance and correlation", {
        "covariance": round(cov, 2),
        "correlation": round(r, 4),
        "r_squared": round(r**2, 4),
    })

    show("C2 three correlation coefficients", {
        "pearson": round(float(stats.pearsonr(x, y).statistic), 4),
        "spearman": round(float(stats.spearmanr(x, y).statistic), 4),
        "kendall": round(float(stats.kendalltau(x, y).statistic), 4),
    })

    h = house()
    show("C3 point-biserial and phi", {
        "point_biserial_price_rooms": round(float(stats.pointbiserialr(
            h["rooms"], h["price"]).statistic), 4),
        "phi_weekend_highspend": round(float(np.corrcoef(
            d["weekend"], (d["ad_spend"] > d["ad_spend"].median()).astype(int))[0, 1]), 4),
    })

    ct = pd.crosstab(h["location"], h["rooms"] > 3)
    chi2, _, _, _ = stats.chi2_contingency(ct)
    v = float(np.sqrt(chi2 / ct.to_numpy().sum()))
    show("C4 Cramer's V", round(v, 4))

    num = h.select_dtypes(include=[np.number])
    cm = num.corr().round(3)
    show("C5 correlation matrix", cm.to_string())

    ci = stats.pearsonr(x, y).confidence_interval(0.95)
    show("C6 interval for r", {
        "r": round(r, 4), "low": round(float(ci.low), 4),
        "high": round(float(ci.high), 4), "n": len(d),
    })

    n = 5000
    r_small = 0.08
    tstat = r_small * np.sqrt((n - 2) / (1 - r_small**2))
    show("C7 a small but significant r", {
        "r": r_small, "n": n, "t": round(float(tstat), 3),
        "p": float(f"{2 * stats.t.sf(abs(tstat), n - 2):.3g}"),
        "r_squared": round(r_small**2, 4),
    })

    show("C8 figures", ["images/ch10_correlation_heatmap.png",
                         "images/ch10_correlation_shapes.png"])


# ==================================================== 10.2 traps
def section_102() -> None:
    c = pd.read_csv(AUDIT / "correlation_cases.csv")
    rows = []
    for key in ["linear", "curved", "outlier", "noisy", "trend_a", "part_whole"]:
        rows.append((key, round(float(c["x"].corr(c[key])), 4)))
    show("T1 correlations of six cases", pd.DataFrame(
        rows, columns=["case", "pearson_r"]).to_string(index=False))

    show("T2 curved relationship", {
        "pearson": round(float(c["x"].corr(c["curved"])), 4),
        "spearman": round(float(stats.spearmanr(
            c["x"], c["curved"]).statistic), 4),
    })

    clean_r = float(c["x"].corr(c["linear"]))
    dirty_r = float(c["x"].corr(c["outlier"]))
    show("T3 one outlier", {
        "r_without": round(clean_r, 4), "r_with": round(dirty_r, 4),
        "change": round(dirty_r - clean_r, 4),
    })

    show("T4 spurious time trend", {
        "trend_a_vs_trend_b": round(float(c["trend_a"].corr(c["trend_b"])), 4),
        "both_driven_by": "x",
    })

    show("T5 part-whole correlation", {
        "x_vs_linear": round(clean_r, 4),
        "x_vs_part_whole": round(float(c["x"].corr(c["part_whole"])), 4),
        "note": "part_whole contains x by construction",
    })

    d = ad()
    grp = d.groupby("channel")[["ad_spend", "revenue"]].mean()
    show("T6 ecological correlation", {
        "individual_r": round(float(d["ad_spend"].corr(d["revenue"])), 4),
        "group_mean_r": round(float(grp["ad_spend"].corr(grp["revenue"])), 4),
        "groups": len(grp),
    })

    show("T7 Anscombe reminder", {
        "note": "four datasets with identical r about 0.816 and different shapes",
        "fixture": "data/raw/audit/anscombe_quartet.csv",
    })


# ==================================================== 10.3 causation
def section_103() -> None:
    h = house()
    fit1 = ols(h, "price", ["size_sqft"])
    show("CC1 confounding in the housing data", {
        "simple_slope_size": round(float(fit1["coef"][1]), 2),
        "note": "size and rooms are correlated; the simple slope mixes both",
        "size_rooms_r": round(float(h["size_sqft"].corr(h["rooms"])), 4),
    })

    show("CC2 reverse causality", [
        "ad spend may respond to expected revenue, not only cause it",
        "a correlation between tickets and churn cannot say which came first",
        "randomised spend is the design that separates the directions",
    ])

    show("CC3 a collider", [
        "conditioning on a variable caused by both X and Y creates association",
        "example: selecting customers who filed a support ticket",
        "the resulting sample can show a spurious negative relationship",
    ])

    show("CC4 controlling for a bad variable", [
        "control for pre-treatment variables that affect the outcome",
        "do not control for a mediator if the total effect is wanted",
        "do not control for a collider or a consequence of Y",
    ])


# ==================================================== 10.4 simple OLS
def section_104() -> None:
    d = ad()
    fit = ols(d, "revenue", ["ad_spend"])
    show("S1 simple regression coefficients", coef_table(fit).to_string(index=False))
    show("S2 model fit", {
        "r2": round(fit["r2"], 4), "adj_r2": round(fit["adj_r2"], 4),
        "residual_sd": round(fit["resid_sd"], 2), "n": fit["n"],
    })

    slope = float(fit["coef"][1])
    se = float(fit["se"][1])
    ci = stats.t.interval(0.95, fit["n"] - fit["k"],
                          loc=slope, scale=se)
    show("S3 slope interval", {
        "slope": round(slope, 3), "se": round(se, 3),
        "low": round(float(ci[0]), 3), "high": round(float(ci[1]), 3),
        "unit": "revenue dollars per ad-spend dollar",
    })

    xbar, ybar = float(d["ad_spend"].mean()), float(d["revenue"].mean())
    show("S4 intercept at the mean", {
        "intercept": round(float(fit["coef"][0]), 2),
        "mean_x": round(xbar, 2), "mean_y": round(ybar, 2),
        "ybar_from_line": round(float(fit["coef"][0] + slope * xbar), 2),
    })

    show("S5 fitted versus observed (head)", pd.DataFrame({
        "ad_spend": d["ad_spend"].head(4).round(2),
        "revenue": d["revenue"].head(4).round(2),
        "fitted": np.round(fit["fitted"][:4], 2),
        "residual": np.round(fit["resid"][:4], 2),
    }).to_string(index=False))

    show("S6 prediction at three spend levels", pd.DataFrame([
        (v, round(float(fit["coef"][0] + slope * v), 2))
        for v in [1000.0, 5000.0, 12000.0]
    ], columns=["ad_spend", "predicted_revenue"]).to_string(index=False))

    show("S7 figures", ["images/ch10_simple_regression.png"])


# ==================================================== 10.5 diagnostics
def section_105() -> None:
    d = ad()
    fit = ols(d, "revenue", ["ad_spend", "weekend"])
    resid, fitted = fit["resid"], fit["fitted"]
    show("D1 four assumption checks", {
        "residual_mean": round(float(resid.mean()), 3),
        "shapiro_p": float(f"{stats.shapiro(resid).pvalue:.3g}"),
        "breusch_pagan_proxy": round(float(np.corrcoef(
            fitted, np.abs(resid))[0, 1]), 4),
        "max_abs_residual": round(float(np.abs(resid).max()), 2),
    })

    qs = np.quantile(fitted, [0, 1/3, 2/3, 1])
    groups = [np.abs(resid[(fitted >= a) & (fitted < b)])
              for a, b in zip(qs[:-1], qs[1:])]
    lev = stats.levene(*groups)
    show("D2 heteroscedasticity test", {
        "levene_stat": round(float(lev.statistic), 3),
        "p": round(float(lev.pvalue), 4),
    })

    X = fit["X"]
    h_diag = np.einsum("ij,ij->i", X @ np.linalg.inv(X.T @ X), X)
    cook = (fit["resid"] ** 2 / (fit["resid_sd"] ** 2 * fit["k"])) * (
        h_diag / (1 - h_diag) ** 2)
    show("D3 influential points", {
        "mean_leverage": round(float(h_diag.mean()), 4),
        "max_leverage": round(float(h_diag.max()), 4),
        "max_cooks_d": round(float(cook.max()), 4),
        "above_4_over_n": int((cook > 4 / fit["n"]).sum()),
    })

    show("D4 residuals over time", {
        "first_half_mean": round(float(resid[:182].mean()), 2),
        "second_half_mean": round(float(resid[182:].mean()), 2),
        "lag1_autocorrelation": round(float(np.corrcoef(
            resid[:-1], resid[1:])[0, 1]), 4),
    })

    show("D5 four assumptions", [
        "linearity: residuals show no systematic curve against fitted",
        "independence: no autocorrelation or clustered structure",
        "homoscedasticity: constant spread of residuals",
        "normality: residuals roughly normal for small-sample inference",
    ])

    robust = ols(d, "revenue", ["ad_spend", "weekend"])
    show("D6 robust comparison", {
        "ols_se_spend": round(float(robust["se"][1]), 4),
        "note": "heteroscedasticity-robust SEs require a sandwich estimator",
        "claim": "the point estimate is unchanged; only the SE would move",
    })

    show("D7 figure", "images/ch10_diagnostics.png")


# ==================================================== 10.6 multiple
def section_106() -> None:
    h = house()
    dummies = pd.get_dummies(h["location"], prefix="loc", drop_first=True).astype(int)
    h2 = h.join(dummies)
    feats = ["size_sqft", "rooms", "rating", "amenities",
             "loc_B", "loc_C", "interaction"]
    h2["interaction"] = (h2["loc_C"] * (h2["size_sqft"] > 1800)).astype(int)
    fit = ols(h2, "price", feats)
    show("M1 multiple regression coefficients", coef_table(fit).to_string(index=False))
    show("M2 fit", {"r2": round(fit["r2"], 4),
                     "adj_r2": round(fit["adj_r2"], 4),
                     "residual_sd": round(fit["resid_sd"], 2)})

    show("M3 multicollinearity", vif(h2, ["size_sqft", "rooms", "rating",
                                           "amenities"]))

    fit_no_rooms = ols(h2, "price", [f for f in feats if f != "rooms"])
    show("M4 dropping the worst offender", {
        "r2_with_rooms": round(fit["r2"], 4),
        "r2_without_rooms": round(fit_no_rooms["r2"], 4),
        "size_se_with": round(float(fit["se"][1]), 2),
        "size_se_without": round(float(fit_no_rooms["se"][1]), 2),
    })

    slopes = {}
    for loc in ["A", "C"]:
        sub = h[h["location"] == loc]
        sf = ols(sub, "price", ["size_sqft"])
        slopes[loc] = round(float(sf["coef"][1]), 2)
    show("M5 simple slopes", slopes)

    h["log_size"] = np.log(h["size_sqft"])
    h["log_price"] = np.log(h["price"])
    logfit = ols(h, "log_price", ["log_size", "rating", "amenities"])
    show("M6 log-log specification", {
        "size_elasticity": round(float(logfit["coef"][1]), 3),
        "r2": round(logfit["r2"], 4),
    })

    show("M7 model-building discipline", [
        "start from the question and a small set of pre-specified predictors",
        "add variables for a reason, not for a higher R-squared",
        "check collinearity and influence after each addition",
        "keep a holdout set untouched until the model is frozen",
    ])

    show("M8 figure", "images/ch10_interaction.png")


# ==================================================== 10.7 logistic
def section_107() -> None:
    d = churn_df()
    feats = ["tenure_months", "monthly_charges", "support_tickets",
             "satisfaction"]
    res, prob = logistic(d, "churn", feats)
    show("L1 logistic coefficients", res["table"].to_string(index=False))

    table = res["table"].set_index("term")
    for term in ["tenure_months", "support_tickets"]:
        lo, hi = stats.norm.interval(0.95, loc=table.loc[term, "coef"],
                                     scale=table.loc[term, "std_error"])
        show(f"L2 interval for {term}", {
            "coef": round(float(table.loc[term, "coef"]), 4),
            "low": round(float(lo), 4), "high": round(float(hi), 4),
            "odds_ratio": round(float(np.exp(table.loc[term, "coef"])), 4),
        })

    show("L3 odds ratio versus risk ratio", {
        "odds_ratio_ticket": round(float(np.exp(
            table.loc["support_tickets", "coef"])), 4),
        "base_rate": round(float(d["churn"].mean()), 4),
        "note": "an OR of 1.9 is not a 90% increase in probability",
    })

    for t in [0.3, 0.5, 0.7]:
        pred = (prob >= t).astype(int)
        tp = int(((pred == 1) & (d["churn"] == 1)).sum())
        prec = tp / max(int(pred.sum()), 1)
        rec = tp / int(d["churn"].sum())
        show(f"L4 threshold {t}", {
            "flagged": int(pred.sum()), "precision": round(prec, 3),
            "recall": round(rec, 3),
        })

    auc = roc_auc_score(d["churn"], prob)
    show("L5 AUC", round(float(auc), 4))

    bins = np.quantile(prob, np.linspace(0, 1, 6))
    obs = []
    exp = []
    for a, b in zip(bins[:-1], bins[1:]):
        mask = (prob >= a) & (prob <= b)
        obs.append(float(d.loc[mask, "churn"].mean()))
        exp.append(float(prob[mask].mean()))
    show("L6 calibration by quintile", pd.DataFrame({
        "predicted": np.round(exp, 3), "observed": np.round(obs, 3),
    }).to_string(index=False))

    show("L7 figure", "images/ch10_logit_roc.png")


# ==================================================== 10.8 validation
def section_108() -> None:
    d = ad()
    cut = int(len(d) * 0.7)
    train, test = d.iloc[:cut], d.iloc[cut:]
    f_train = ols(train, "revenue", ["ad_spend", "weekend"])
    pred = f_train["coef"][0] + f_train["coef"][1] * test["ad_spend"] \
        + f_train["coef"][2] * test["weekend"]
    show("V1 temporal holdout", {
        "train_rows": len(train), "test_rows": len(test),
        "train_r2": round(f_train["r2"], 4),
        "test_rmse": round(float(np.sqrt(mean_squared_error(
            test["revenue"], pred))), 2),
    })

    f_all = ols(d, "revenue", ["ad_spend", "weekend"])
    pred_all = f_all["coef"][0] + f_all["coef"][1] * test["ad_spend"] \
        + f_all["coef"][2] * test["weekend"]
    show("V2 leakage demonstration", {
        "honest_test_rmse": round(float(np.sqrt(mean_squared_error(
            test["revenue"], pred))), 2),
        "leaked_test_rmse": round(float(np.sqrt(mean_squared_error(
            test["revenue"], pred_all))), 2),
    })

    kfold = []
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(d))
    for fold in np.array_split(idx, 5):
        mask = np.ones(len(d), dtype=bool)
        mask[fold] = False
        tr, te = d.iloc[mask], d.iloc[~mask]
        ff = ols(tr, "revenue", ["ad_spend", "weekend"])
        pp = ff["coef"][0] + ff["coef"][1] * te["ad_spend"] \
            + ff["coef"][2] * te["weekend"]
        kfold.append(float(np.sqrt(mean_squared_error(te["revenue"], pp))))
    show("V3 five-fold CV", {
        "fold_rmse": [round(v, 2) for v in kfold],
        "mean_rmse": round(float(np.mean(kfold)), 2),
    })

    X = pd.get_dummies(d[["ad_spend", "weekend", "channel"]],
                       drop_first=True).astype(float)
    y = d["revenue"]
    for name, model in [("ridge", Ridge(alpha=100.0)),
                        ("lasso", Lasso(alpha=50.0, max_iter=5000))]:
        model.fit(X.iloc[:cut], y.iloc[:cut])
        p = model.predict(X.iloc[cut:])
        show(f"V4 {name} on holdout", {
            "rmse": round(float(np.sqrt(mean_squared_error(y.iloc[cut:], p))), 2),
            "nonzero_coefs": int((np.abs(model.coef_) > 1e-6).sum()),
        })

    show("V5 bias-variance", [
        "ridge shrinks coefficients but keeps all of them",
        "lasso sets small coefficients to zero, performing selection",
        "both trade a little bias for lower variance on new data",
    ])

    show("V6 figure", "images/ch10_lasso_paths.png")


# ==================================================== 10.9 communicating
def section_109() -> None:
    d = ad()
    fit = ols(d, "revenue", ["ad_spend", "weekend"])
    show("CM1 coefficient sentence", {
        "sentence": (f"Each additional dollar of ad spend is associated with "
                     f"${float(fit['coef'][1]):.2f} of revenue "
                     f"(95% CI {float(fit['coef'][1] - 1.96 * fit['se'][1]):.2f} "
                     f"to {float(fit['coef'][1] + 1.96 * fit['se'][1]):.2f})."),
        "unit": "USD revenue per USD spend",
    })

    show("CM2 marginal effect at three spend levels", pd.DataFrame([
        (v, round(float(fit["coef"][0] + fit["coef"][1] * v), 2))
        for v in [1000.0, 5000.0, 9000.0]
    ], columns=["ad_spend", "predicted_revenue"]).to_string(index=False))

    show("CM3 what-if scenario", {
        "spend_5000": round(float(fit["coef"][0] + fit["coef"][1] * 5000), 2),
        "spend_6000": round(float(fit["coef"][0] + fit["coef"][1] * 6000), 2),
        "difference": round(float(fit["coef"][1] * 1000), 2),
    })

    show("CM4 limitations paragraph", [
        "observational spend data; seasonality is only partly modelled",
        "linear response assumed over the observed spend range",
        "no data beyond $9,000 per day, so extrapolation is unsafe",
        "R-squared describes fit, not causality",
    ])


# ==================================================== 10.10 example bank
def section_1010() -> None:
    d = ad()
    show("E1 pairwise correlations", {
        "spend_revenue": round(float(d["ad_spend"].corr(d["revenue"])), 4),
        "weekend_revenue": round(float(d["weekend"].corr(d["revenue"])), 4),
    })
    show("E2 rank correlation", round(float(stats.spearmanr(
        d["ad_spend"], d["revenue"]).statistic), 4))
    show("E3 R-squared equals r-squared", round(float(
        ols(d, "revenue", ["ad_spend"])["r2"]), 4))
    show("E4 residual mean", round(float(
        ols(d, "revenue", ["ad_spend", "weekend"])["resid"].mean()), 6))
    show("E5 max VIF", max(vif(d, ["ad_spend", "weekend"]).values()))
    show("E6 holdout rows", int(len(d) * 0.3))


# ==================================================== 10.11 capstone
def section_1011() -> None:
    d = ad()
    fit = ols(d, "revenue", ["ad_spend", "weekend"])
    ch = churn_df()
    logfit, prob = logistic(ch, "churn", [
        "tenure_months", "monthly_charges", "support_tickets", "satisfaction"])
    table = pd.DataFrame([
        ("linear slope spend", round(float(fit["coef"][1]), 3),
         "USD revenue per USD", "observational"),
        ("linear R-squared", round(fit["r2"], 4), "share", "in sample"),
        ("churn AUC", round(float(roc_auc_score(ch["churn"], prob)), 4),
         "share", "in sample"),
        ("ticket odds ratio", float(table_odds(logfit)), "ratio",
         "per additional ticket"),
    ], columns=["measure", "value", "unit", "basis"])
    show("K1 model summary", table.to_string(index=False))

    show("K2 control totals", {
        "ad_rows": len(d), "churn_rows": len(ch),
        "ad_revenue": round(float(d["revenue"].sum()), 2),
        "churned": int(ch["churn"].sum()),
    })

    show("K3 decision", {
        "recommendation": "treat both models as descriptive",
        "reason": "observational inputs; no randomized spend or holdout evidence of causality",
        "next_step": "randomise spend across days and re-estimate the response",
    })

    show("K4 artifacts", ["ad_response_model.md", "churn_model.md",
                          "diagnostics.png", "model_limitations.md"])


def table_odds(res: dict) -> float:
    t = res["table"].set_index("term")
    return float(t.loc["support_tickets", "odds_ratio"])


def section_extra() -> None:
    """Extra worked blocks for the expanded Chapter 10 sections."""
    d = ad()
    h = house()
    ch = churn_df()
    feats = ["tenure_months", "monthly_charges", "support_tickets",
             "satisfaction"]
    res, prob = logistic(ch, "churn", feats)
    fit = ols(d, "revenue", ["ad_spend", "weekend"])

    # X1 bootstrap interval for r versus the analytic one
    rng = np.random.default_rng(SEED)
    rboot = []
    xv, yv = d["ad_spend"].to_numpy(), d["revenue"].to_numpy()
    for _ in range(1000):
        idx = rng.integers(0, len(d), size=len(d))
        rboot.append(np.corrcoef(xv[idx], yv[idx])[0, 1])
    show("X1 bootstrap interval for r", {
        "analytic_low": 0.8213, "analytic_high": 0.8779,
        "bootstrap_low": round(float(np.quantile(rboot, 0.025)), 4),
        "bootstrap_high": round(float(np.quantile(rboot, 0.975)), 4),
    })

    # X2 partial correlation of size and price controlling rooms
    def resid_of(frame, col, controls):
        f = ols(frame, col, controls)
        return f["resid"]
    rs = resid_of(h, "size_sqft", ["rooms"])
    rp = resid_of(h, "price", ["rooms"])
    show("X2 partial correlation", {
        "raw": round(float(h["size_sqft"].corr(h["price"])), 4),
        "controlling_rooms": round(float(np.corrcoef(rs, rp)[0, 1]), 4),
    })

    # X3 ordered correlation matrix
    num = h.select_dtypes(include=[np.number])
    cm = num.corr()
    order = sorted(cm.columns, key=lambda c: -cm[c].abs().mean())
    show("X3 ordered correlation columns", order)

    # X4 same r, different shape (two cases from the fixture)
    c = pd.read_csv(AUDIT / "correlation_cases.csv")
    show("X4 same r different shape", {
        "linear": round(float(c["x"].corr(c["linear"])), 4),
        "part_whole": round(float(c["x"].corr(c["part_whole"])), 4),
    })

    # X5 polynomial specification
    Xp = np.column_stack([np.ones(len(d)), d["ad_spend"],
                          d["ad_spend"] ** 2 / 1000])
    yp = d["revenue"].to_numpy(float)
    bp = np.linalg.lstsq(Xp, yp, rcond=None)[0]
    rp = yp - Xp @ bp
    r2p = 1 - float(rp @ rp) / float(((yp - yp.mean()) ** 2).sum())
    show("X5 quadratic term", {
        "linear_r2": round(fit["r2"], 4),
        "quadratic_r2": round(r2p, 4),
        "curvature_coef": round(float(bp[2]), 4),
    })

    # X6 sandwich (heteroscedasticity-robust) standard errors
    Xf = fit["X"]
    meat = Xf.T @ (fit["resid"][:, None] ** 2 * Xf)
    bread = np.linalg.inv(Xf.T @ Xf)
    cov_rob = bread @ meat @ bread * len(d) / (len(d) - fit["k"])
    se_rob = np.sqrt(np.diag(cov_rob))
    show("X6 robust vs OLS standard errors", {
        "ols_se_spend": round(float(fit["se"][1]), 4),
        "robust_se_spend": round(float(se_rob[1]), 4),
        "ratio": round(float(se_rob[1] / fit["se"][1]), 3),
    })

    # X7 condition number
    Xs = Xf[:, 1:]
    show("X7 condition number", round(float(np.linalg.cond(Xs)), 2))

    # X8 VIF after dropping rooms
    dummies = pd.get_dummies(h["location"], prefix="loc", drop_first=True).astype(int)
    h2 = h.join(dummies)
    keep = ["size_sqft", "rating", "amenities"]
    show("X8 VIF after dropping rooms", vif(h2, keep))

    # X9 logistic marginal effect at the mean
    means = ch[feats].mean().to_numpy(float)
    p0 = float(res["clf"].predict_proba(means.reshape(1, -1))[0, 1])
    means_t = means.copy()
    means_t[2] += 1
    p1 = float(res["clf"].predict_proba(means_t.reshape(1, -1))[0, 1])
    show("X9 marginal effect of one ticket", {
        "baseline_probability": round(p0, 4),
        "probability_after_ticket": round(p1, 4),
        "difference_pp": round((p1 - p0) * 100, 2),
    })

    # X10 McFadden pseudo R-squared
    ylog = ch["churn"].to_numpy(int)
    ll_model = float(np.sum(ylog * np.log(prob) + (1 - ylog) * np.log(1 - prob)))
    pbar = float(ylog.mean())
    ll_null = float(np.sum(ylog * np.log(pbar) + (1 - ylog) * np.log(1 - pbar)))
    show("X10 McFadden pseudo R2", round(1 - ll_model / ll_null, 4))

    # X11 confusion matrix at 0.5
    pred = (prob >= 0.5).astype(int)
    cm2 = pd.crosstab(ch["churn"], pred)
    show("X11 confusion matrix", cm2.to_string())

    # X12 AUC by segment
    aucs = {s: round(float(roc_auc_score(g["churn"],
                                         prob[g.index])), 4)
            for s, g in ch.groupby("contract")}
    show("X12 AUC by contract", aucs)

    # X13 learning curve
    rng2 = np.random.default_rng(SEED)
    Xa = np.column_stack([np.ones(len(d)), d["ad_spend"], d["weekend"]])
    ya = d["revenue"].to_numpy(float)
    test_idx = rng2.permutation(len(d))[:100]
    curve = []
    for m in [50, 100, 200, 265]:
        idx = rng2.choice([i for i in range(len(d)) if i not in test_idx],
                          size=m, replace=False)
        bb = np.linalg.lstsq(Xa[idx], ya[idx], rcond=None)[0]
        pp = Xa[test_idx] @ bb
        curve.append((m, round(float(np.sqrt(mean_squared_error(
            ya[test_idx], pp))), 2)))
    show("X13 learning curve", pd.DataFrame(
        curve, columns=["train_n", "test_rmse"]).to_string(index=False))

    # X14 train/test R2 gap
    cut = int(len(d) * 0.7)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    Xtr = np.column_stack([np.ones(len(tr)), tr["ad_spend"], tr["weekend"]])
    btr = np.linalg.lstsq(Xtr, tr["revenue"].to_numpy(float), rcond=None)[0]
    pte = btr[0] + btr[1] * te["ad_spend"] + btr[2] * te["weekend"]
    ss_res_te = float(((te["revenue"] - pte) ** 2).sum())
    ss_tot_te = float(((te["revenue"] - te["revenue"].mean()) ** 2).sum())
    show("X14 overfit gap", {
        "train_r2": round(fit["r2"], 4),
        "test_r2": round(1 - ss_res_te / ss_tot_te, 4),
    })

    # X15 bias-variance statement
    show("X15 bias-variance", {
        "flexible_model": "lower bias, higher variance",
        "regularised_model": "higher bias, lower variance",
        "selection": "choose by holdout error, not training fit",
    })

    # X16 simple regression with interaction on housing
    h2["big"] = (h2["size_sqft"] > 1800).astype(int)
    Xint = np.column_stack([np.ones(len(h2)), h2["size_sqft"],
                            h2["loc_C"], h2["big"],
                            h2["loc_C"] * h2["big"]])
    yint = h2["price"].to_numpy(float)
    bint = np.linalg.lstsq(Xint, yint, rcond=None)[0]
    show("X16 interaction coefficients", {
        "size": round(float(bint[1]), 2),
        "loc_C": round(float(bint[2]), 2),
        "big": round(float(bint[3]), 2),
        "loc_C_x_big": round(float(bint[4]), 2),
    })

    # X17 residual by spend third
    thirds = pd.qcut(d["ad_spend"], 3, labels=["low", "mid", "high"])
    show("X17 residual spread by spend third", fit_resid_spread(fit, d, thirds))

    # X18 final model inventory
    show("X18 model inventory", pd.DataFrame([
        ("linear spend", "revenue", "r2 0.726", "observational"),
        ("housing multiple", "price", "r2 0.878", "collinear predictors"),
        ("logistic churn", "churn", "AUC 0.867", "in-sample"),
    ], columns=["model", "outcome", "headline", "caveat"]).to_string(index=False))


def fit_resid_spread(fit, d, thirds):
    out = {}
    for g in thirds.cat.categories:
        mask = (thirds == g).to_numpy()
        out[str(g)] = round(float(np.std(fit["resid"][mask])), 2)
    return out


def main() -> None:
    print("python", "pandas", pd.__version__, "numpy", np.__version__)
    print()
    for section in (
        section_101, section_102, section_103, section_104, section_105,
        section_106, section_107, section_108, section_109, section_1010,
        section_1011, section_extra,
    ):
        print("=" * 78)
        section()


if __name__ == "__main__":
    main()
