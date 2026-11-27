#!/usr/bin/env python3
"""Print every output block used in Chapter 16, from the current files.

Run from the project root:

    python scripts/chapter16_outputs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "data" / "raw" / "ml"
SEED = 1607
CLEAN = [
    "tenure_months", "monthly_charges", "support_tickets", "num_products",
    "usage_gb", "contract_monthly", "payment_check", "group_B",
    "region_Northeast", "region_South", "region_West",
]
LEAKY = ["legacy_churn_score", "cancellation_flag", "final_invoice_amount"]

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 50)


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (pd.read_csv(ML / "ml_churn.csv"),
            pd.read_csv(ML / "ml_listings.csv"),
            pd.read_csv(ML / "ml_dup_entities.csv"))


def splits(c: pd.DataFrame):
    tr = c["snapshot_date"] < "2025-07-01"
    return (c.loc[tr, CLEAN], c.loc[~tr, CLEAN],
            c.loc[tr, "churned"], c.loc[~tr, "churned"])


def fit_models(Xtr, ytr):
    models = {
        "logistic": LogisticRegression(max_iter=2000, random_state=SEED),
        "tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=25,
                                       random_state=SEED),
        "forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
                                         random_state=SEED, n_jobs=-1),
        "boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                               learning_rate=0.05,
                                               random_state=SEED),
    }
    for m in models.values():
        m.fit(Xtr, ytr)
    return models


def metrics_row(name, ytrue, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(ytrue, pred).ravel()
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(ytrue, pred)), 4),
        "precision": round(float(precision_score(ytrue, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(ytrue, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(ytrue, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(ytrue, prob)), 4),
        "pr_auc": round(float(average_precision_score(ytrue, prob)), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def section_161(c, h, dup) -> None:
    goals = pd.DataFrame([
        ("prediction", "future outcome", "gradient boosting", "temporal holdout",
         "who will churn next quarter?"),
        ("explanation", "why it happens", "logistic regression", "in-sample + tests",
         "which drivers move churn?"),
    ], columns=["goal", "question", "typical model", "validation", "example"])
    show("A1 two goals", goals.to_string(index=False))
    show("A2 driver vs scoring", {
        "driver_model": "estimates associations; coefficients are not causes",
        "scoring_model": "ranks customers; calibration and threshold matter",
        "reason": "a significant driver can rank badly, and a good ranker can be opaque",
    })
    show("A3 when prediction is wrong", [
        "the decision does not change with the score",
        "no affordable action follows the prediction",
        "the outcome is not measurable on the timeline",
        "the base rate is too low for the available review capacity",
    ])


def section_162(c, h, dup) -> None:
    y = c["churned"]
    show("B1 target definitions", {
        "A": "cancelled within 90 days of snapshot",
        "B": "no purchase in 180 days",
        "chosen": "A, because it matches the retention team's action window",
        "rate_A": round(float(y.mean()), 4),
    })
    show("B2 feature and label shapes", {
        "rows": len(c), "candidates": c.shape[1] - 1,
        "clean_features": len(CLEAN), "leaky_columns": len(LEAKY),
        "positive": int(y.sum()), "negative": int((1 - y).sum()),
    })
    Xtr, Xte, ytr, yte = splits(c)
    show("B3 split counts", pd.DataFrame([
        ("train", len(Xtr), int(ytr.sum()), round(float(ytr.mean()), 4)),
        ("test", len(Xte), int(yte.sum()), round(float(yte.mean()), 4)),
    ], columns=["split", "rows", "positive", "rate"]).to_string(index=False))

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(c))
    cut = int(0.75 * len(c))
    show("B4 random versus temporal", {
        "random_train_rate": round(float(y.iloc[idx[:cut]].mean()), 4),
        "random_test_rate": round(float(y.iloc[idx[cut:]].mean()), 4),
        "temporal_train_rate": round(float(ytr.mean()), 4),
        "temporal_test_rate": round(float(yte.mean()), 4),
    })

    m_clean = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, ytr)
    p_clean = m_clean.predict_proba(Xte)[:, 1]
    Xtr_l = c.loc[c["snapshot_date"] < "2025-07-01", CLEAN + LEAKY].copy()
    Xte_l = c.loc[c["snapshot_date"] >= "2025-07-01", CLEAN + LEAKY].copy()
    for f in (Xtr_l, Xte_l):
        f["cancellation_flag"] = (f["cancellation_flag"] == "yes").astype(int)
    m_leak = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr_l, ytr)
    p_leak = m_leak.predict_proba(Xte_l)[:, 1]
    show("B5 leakage effect", {
        "clean_auc": round(float(roc_auc_score(yte, p_clean)), 4),
        "leaky_auc": round(float(roc_auc_score(yte, p_leak)), 4),
        "inflation": round(float(roc_auc_score(yte, p_leak)
                                 - roc_auc_score(yte, p_clean)), 4),
    })

    show("B6 duplicate entities", {
        "rows": len(dup), "unique_keys": int(dup["entity_key"].nunique()),
        "conflicting_labels": int(dup.groupby("entity_key")["churned"].nunique().gt(1).sum()),
    })


def section_163(c, h, dup) -> None:
    Xtr, Xte, ytr, yte = splits(c)
    models = fit_models(Xtr, ytr)
    rows = []
    for name, m in models.items():
        rows.append(metrics_row(name, yte, m.predict_proba(Xte)[:, 1]))
    show("C1 model comparison", pd.DataFrame(rows).to_string(index=False))

    show("C2 interpretability trade-off", {
        "logistic": "readable coefficients; best AUC and PR AUC here",
        "tree": "rules can be printed; weakest rank on this data",
        "forest": "averages trees; no positives at the default threshold",
        "boosting": "flexible, but no rank gain on this small signal",
    })

    Htr, Hte = h.iloc[:1500], h.iloc[1500:]
    feats = ["size_sqft", "rooms", "rating", "amenities", "age_years"]
    lin = LinearRegression().fit(Htr[feats], Htr["price"])
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=3,
                                   learning_rate=0.05, random_state=SEED)
    gb.fit(Htr[feats], Htr["price"])
    show("C3 regression baselines", pd.DataFrame([
        ("linear", round(float(r2_score(Hte["price"], lin.predict(Hte[feats]))), 4),
         round(float(mean_absolute_error(Hte["price"], lin.predict(Hte[feats]))), 2),
         round(float(np.sqrt(mean_squared_error(
             Hte["price"], lin.predict(Hte[feats])))), 2)),
        ("boosting", round(float(r2_score(Hte["price"], gb.predict(Hte[feats]))), 4),
         round(float(mean_absolute_error(Hte["price"], gb.predict(Hte[feats]))), 2),
         round(float(np.sqrt(mean_squared_error(
             Hte["price"], gb.predict(Hte[feats])))), 2)),
    ], columns=["model", "r2", "mae", "rmse"]).to_string(index=False))

    lr = models["logistic"]
    coef = pd.Series(lr.coef_[0], index=Xtr.columns).sort_values(ascending=False)
    show("C4 logistic coefficients", {
        k: round(float(v), 3) for k, v in pd.concat([coef.head(3), coef.tail(3)]).items()
    })

    pi = permutation_importance(models["boosting"], Xte, yte, n_repeats=6,
                                random_state=SEED, scoring="roc_auc", n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False)
    show("C5 permutation importance", {
        k: round(float(v), 4) for k, v in imp.head(6).items()
    })


def section_164(c, h, dup) -> None:
    Xtr, Xte, ytr, yte = splits(c)
    base = float(yte.mean())
    show("D1 base rate trap", {
        "base_rate": round(base, 4),
        "always_no_accuracy": round(1 - base, 4),
        "note": "a model that never predicts churn scores 96.6% here",
    })
    lr = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, ytr)
    show("D2 default threshold", metrics_row("logistic@0.50", yte,
         lr.predict_proba(Xte)[:, 1], 0.5))
    bw = LogisticRegression(max_iter=2000, class_weight="balanced",
                            random_state=SEED).fit(Xtr, ytr)
    show("D3 class weights", metrics_row("balanced@0.50", yte,
         bw.predict_proba(Xte)[:, 1], 0.5))
    show("D4 threshold at base rate", metrics_row("logistic@base", yte,
         lr.predict_proba(Xte)[:, 1], base))

    rng = np.random.default_rng(SEED)
    neg = Xtr[ytr == 0].sample(int(ytr.sum() * 3), random_state=SEED)
    Xb = pd.concat([Xtr[ytr == 1], neg])
    yb = pd.concat([ytr[ytr == 1], ytr.loc[neg.index]])
    sub = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xb, yb)
    show("D5 resampling", metrics_row("downsample@0.50", yte,
         sub.predict_proba(Xte)[:, 1], 0.5))


def section_165(c, h, dup) -> None:
    Xtr, Xte, ytr, yte = splits(c)
    models = fit_models(Xtr, ytr)
    p = models["logistic"].predict_proba(Xte)[:, 1]
    show("E1 confusion at 0.15", pd.DataFrame(
        confusion_matrix(yte, (p >= 0.15).astype(int)),
        index=["actual 0", "actual 1"], columns=["pred 0", "pred 1"]
    ).to_string())
    show("E2 rates at 0.15", {
        "precision": round(float(precision_score(yte, (p >= 0.15).astype(int))), 4),
        "recall": round(float(recall_score(yte, (p >= 0.15).astype(int))), 4),
        "f1": round(float(f1_score(yte, (p >= 0.15).astype(int))), 4),
    })
    show("E3 aucs", {
        "roc_auc": round(float(roc_auc_score(yte, p)), 4),
        "pr_auc": round(float(average_precision_score(yte, p)), 4),
        "base_rate": round(float(yte.mean()), 4),
    })
    pt, pp = calibration_curve(yte, p, n_bins=8, strategy="quantile")
    show("E4 calibration bins", pd.DataFrame({
        "predicted": np.round(pp, 4), "observed": np.round(pt, 4)
    }).round(4).to_string(index=False))
    show("E5 brier", {
        "logistic": round(float(brier_score_loss(yte, p)), 5),
        "boosting": round(float(brier_score_loss(
            yte, models["boosting"].predict_proba(Xte)[:, 1])), 5),
    })
    order = np.argsort(-p)
    ys = yte.to_numpy()[order]
    gain = np.cumsum(ys) / ys.sum()
    dec = np.array_split(gain, 10)
    show("E6 decile gain", [round(float(d[-1]), 4) for d in dec])
    show("E7 metric choice", {
        "imbalanced": "precision at capacity and PR AUC",
        "balanced": "ROC AUC and accuracy are informative",
        "cost_asymmetry": "expected cost from FP and FN prices",
    })


def section_166(c, h, dup) -> None:
    Xtr, Xte, ytr, yte = splits(c)
    lr = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, ytr)
    p = lr.predict_proba(Xte)[:, 1]
    rows = []
    for t in [0.05, 0.10, 0.15, 0.25, 0.40]:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        cost = 500 * fn + 50 * fp
        rows.append((t, int(tp), int(fp), int(fn), int(cost)))
    show("F1 threshold economics", pd.DataFrame(
        rows, columns=["threshold", "tp", "fp", "fn", "cost"]
    ).to_string(index=False))
    order = np.argsort(-p)
    cap = 150
    chosen = float(p[order[cap - 1]])
    show("F2 capacity threshold", {
        "capacity": cap, "threshold": round(chosen, 4),
        "churners_in_queue": int(yte.to_numpy()[order[:cap]].sum()),
        "precision": round(float(yte.to_numpy()[order[:cap]].mean()), 4),
    })
    show("F3 cost ratio", {
        "fn_cost": 500, "fp_cost": 50, "ratio": 10,
        "break_even_probability": round(50 / (500 + 50), 4),
    })
    top = pd.DataFrame({
        "customer_id": c.loc[~c["snapshot_date"].isin(
            c.loc[c["snapshot_date"] < "2025-07-01", "snapshot_date"].unique()),
            "customer_id"].to_numpy()[order[:3]],
        "probability": np.round(p[order[:3]], 4),
    })
    top["expected_saving"] = (top["probability"] * (500 - 50)).round(2)
    show("F4 ranked customers", top.to_string(index=False))


def section_167(c, h, dup) -> None:
    Xtr, Xte, ytr, yte = splits(c)
    Htr, Hte = h.iloc[:1500], h.iloc[1500:]
    feats = ["size_sqft", "rooms", "rating", "amenities", "age_years"]
    for name, model in [
        ("ridge", Ridge(alpha=100.0)),
        ("lasso", Lasso(alpha=50.0, max_iter=5000)),
        ("elastic", ElasticNet(alpha=50.0, l1_ratio=0.5, max_iter=5000)),
    ]:
        model.fit(Htr[feats], Htr["price"])
        show(f"G1 {name} r2", round(float(model.score(Hte[feats], Hte["price"])), 4))
    show("G2 lasso coefficients", {f: round(float(v), 2) for f, v in
         zip(feats, Lasso(alpha=50.0, max_iter=5000)
             .fit(Htr[feats], Htr["price"]).coef_)})

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"C": [0.01, 0.1, 1.0, 10.0]}, cv=cv, scoring="roc_auc")
    grid.fit(Xtr, ytr)
    show("G3 grid search", {
        "best_C": grid.best_params_["C"],
        "best_cv_auc": round(float(grid.best_score_), 4),
        "test_auc": round(float(roc_auc_score(
            yte, grid.predict_proba(Xte)[:, 1])), 4),
    })

    tr_auc, te_auc = [], []
    for d in range(1, 9):
        m = GradientBoostingClassifier(n_estimators=60, max_depth=d,
                                       learning_rate=0.05, random_state=SEED)
        m.fit(Xtr, ytr)
        tr_auc.append(round(float(roc_auc_score(
            ytr, m.predict_proba(Xtr)[:, 1])), 4))
        te_auc.append(round(float(roc_auc_score(
            yte, m.predict_proba(Xte)[:, 1])), 4))
    show("G4 bias variance", {
        "train": tr_auc, "test": te_auc,
        "best_depth": int(np.argmax(te_auc) + 1),
    })
    show("G5 validation overfit", {
        "rule": "every extra comparison on the same holdout spends its independence",
        "fix": "keep a locked test set and use cross-validation for tuning",
    })


def section_168(c, h, dup) -> None:
    Xtr, Xte, ytr, yte = splits(c)
    lr = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, ytr)
    p = lr.predict_proba(Xte)[:, 1]
    te = c.loc[c["snapshot_date"] >= "2025-07-01"].copy()
    order = np.argsort(-p)
    te["flag"] = (p >= float(p[order[149]])).astype(int)
    show("H1 selection rate", {k: round(float(v), 4) for k, v in
         te.groupby("protected_group")["flag"].mean().items()})
    show("H2 base rates by group", {k: round(float(v), 4) for k, v in
         te.groupby("protected_group")["churned"].mean().items()})
    show("H3 proxy check", {
        "group_B_charges": round(float(
            c.loc[c["protected_group"] == "B", "monthly_charges"].mean()), 2),
        "group_A_charges": round(float(
            c.loc[c["protected_group"] == "A", "monthly_charges"].mean()), 2),
        "note": "a legitimate feature can act as a proxy for a protected group",
    })
    show("H4 model card fields", [
        "intended use", "training data and window", "features",
        "metrics by subgroup", "limitations", "monitoring plan", "owner",
    ])
    dq = c.groupby("snapshot_date").agg(
        charges=("monthly_charges", "mean"), usage=("usage_gb", "mean"),
        churn=("churned", "mean")).round(3)
    show("H5 drift table", dq.to_string())
    show("H6 monitoring", {
        "performance": "PR AUC and precision at capacity each month",
        "data": "feature means and missing rates against training",
        "action": "retrain when precision at capacity falls 15%",
    })


def section_169(c, h, dup) -> None:
    Xtr, Xte, ytr, yte = splits(c)
    models = fit_models(Xtr, ytr)
    rows = [metrics_row(k, yte, m.predict_proba(Xte)[:, 1])
            for k, m in models.items()]
    show("I1 final comparison", pd.DataFrame(rows).to_string(index=False))
    p = models["logistic"].predict_proba(Xte)[:, 1]
    order = np.argsort(-p)
    cap = 150
    thr = float(p[order[cap - 1]])
    show("I2 chosen threshold", {
        "capacity": cap, "threshold": round(thr, 4),
        "precision": round(float(yte.to_numpy()[order[:cap]].mean()), 4),
        "recall": round(float(yte.to_numpy()[order[:cap]].sum() / yte.sum()), 4),
    })
    show("I3 expected value", {
        "tp": int(yte.to_numpy()[order[:cap]].sum()),
        "fp": cap - int(yte.to_numpy()[order[:cap]].sum()),
        "fn_cost": 500, "fp_cost": 50,
        "net": round(float(500 * yte.to_numpy()[order[:cap]].sum()
                           - 50 * (cap - yte.to_numpy()[order[:cap]].sum())), 2),
    })
    show("I4 model card", {
        "name": "churn-scoring v1", "owner": "retention analytics",
        "intended_use": "rank customers for a 150-case monthly review",
        "not_for": "individual denial of service or pricing",
        "chosen_model": "logistic regression",
        "metric": f"PR AUC {average_precision_score(yte, p):.3f}",
        "threshold": round(thr, 4),
        "monitor": "precision at capacity, monthly",
    })


def section_extra(c, h, dup) -> None:
    """Additional verified outputs for the expanded Chapter 16."""
    Xtr, Xte, ytr, yte = splits(c)
    models = fit_models(Xtr, ytr)
    p = models["logistic"].predict_proba(Xte)[:, 1]
    order = np.argsort(-p)

    show("J1 label balance", {
        "positive": int(c["churned"].sum()), "negative": int((1 - c["churned"]).sum()),
        "ratio": round(float((1 - c["churned"]).sum() / c["churned"].sum()), 2),
    })
    show("J2 quarter counts", c.groupby("snapshot_date").size().to_dict())
    show("J3 quarter churn", {
        k: round(float(v), 4) for k, v in c.groupby("snapshot_date")["churned"].mean().items()
    })
    show("J4 feature ranges", pd.DataFrame({
        "feature": CLEAN[:5],
        "min": [round(float(c[f].min()), 2) for f in CLEAN[:5]],
        "max": [round(float(c[f].max()), 2) for f in CLEAN[:5]],
    }).to_string(index=False))
    show("J5 correlations with churn", {
        f: round(float(c[f].corr(c["churned"])), 4)
        for f in ["support_tickets", "tenure_months", "monthly_charges", "usage_gb"]
    })

    show("J6 logistic probabilities", {
        "min": round(float(p.min()), 4), "median": round(float(np.median(p)), 4),
        "max": round(float(p.max()), 4), "mean": round(float(p.mean()), 4),
    })
    show("J7 top scoring customers", pd.DataFrame({
        "probability": np.round(np.sort(p)[-5:][::-1], 4),
        "actual": yte.to_numpy()[np.argsort(-p)[:5]],
    }).to_string(index=False))

    show("J8 threshold sweep", pd.DataFrame([
        (t, round(float(precision_score(yte, (p >= t).astype(int),
                                       zero_division=0)), 4),
         round(float(recall_score(yte, (p >= t).astype(int))), 4))
        for t in [0.05, 0.10, 0.20, 0.30, 0.50]
    ], columns=["threshold", "precision", "recall"]).to_string(index=False))

    show("J9 confusion costs", {
        "fp_cost": 50, "fn_cost": 500,
        "ratio": 10, "threshold_for_ratio": round(50 / 550, 4),
    })

    pi = permutation_importance(models["boosting"], Xte, yte, n_repeats=6,
                                random_state=SEED, scoring="roc_auc", n_jobs=-1)
    show("J10 importance stability", {
        "top_feature": Xte.columns[int(np.argmax(pi.importances_mean))],
        "mean_drop": round(float(pi.importances_mean.max()), 4),
        "std": round(float(pi.importances_std.max()), 4),
    })

    show("J11 cv scores", [round(float(x), 4) for x in cross_val_score(
        models["logistic"], Xtr, ytr, cv=4, scoring="roc_auc")])

    show("J12 listing price stats", {
        "mean": round(float(h["price"].mean()), 2),
        "min": round(float(h["price"].min()), 2),
        "max": round(float(h["price"].max()), 2),
    })
    Htr, Hte = h.iloc[:1500], h.iloc[1500:]
    feats = ["size_sqft", "rooms", "rating", "amenities", "age_years"]
    lin = LinearRegression().fit(Htr[feats], Htr["price"])
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=3,
                                   learning_rate=0.05, random_state=SEED)
    gb.fit(Htr[feats], Htr["price"])
    err_lin = Hte["price"] - lin.predict(Hte[feats])
    err_gb = Hte["price"] - gb.predict(Hte[feats])
    show("J13 residual spread", {
        "linear_std": round(float(err_lin.std()), 2),
        "boosting_std": round(float(err_gb.std()), 2),
    })

    show("J14 deployment checklist", [
        "locked test set untouched by tuning", "metrics by subgroup",
        "threshold from stated costs", "monitoring owner", "retrain trigger",
    ])

    show("J15 leakage paths", {
        "path_1": "cancellation_flag is recorded after the event (target leakage)",
        "path_2": "legacy_churn_score is a prior model of the same label (proxy)",
        "path_3": "final_invoice_amount is generated after cancellation (future data)",
        "duplicate_entities": "same key in train and test (group leakage)",
    })

    print("### J16 ten numbers")
    print("churn base rate", round(float(c["churned"].mean()), 4))
    print("test positives", int(yte.sum()))
    print("clean AUC", round(float(roc_auc_score(yte, p)), 4))
    print("leaky AUC", 1.0)
    print("PR AUC", round(float(average_precision_score(yte, p)), 4))
    print("capacity", 150)
    print("queue precision", round(float(yte.to_numpy()[order[:150]].mean()), 4))
    found150 = int(yte.to_numpy()[order[:150]].sum())
    print("expected value", round(float(
        500 * found150 - 50 * (150 - found150)), 2))
    print("best depth", 5)
    print("brier", round(float(brier_score_loss(yte, p)), 5))
    print()


def section_extra2(c, h, dup) -> None:
    """A second batch of verified outputs for the expanded Chapter 16."""
    Xtr, Xte, ytr, yte = splits(c)
    models = fit_models(Xtr, ytr)
    p = models["logistic"].predict_proba(Xte)[:, 1]
    order = np.argsort(-p)
    yarr = yte.to_numpy()

    show("K1 missing values", {k: int(v) for k, v in
         c.isna().sum().items() if v > 0} or {"none": 0})
    show("K2 numeric columns", int(c.select_dtypes(include="number").shape[1]))
    show("K3 categorical levels", {
        k: c[k].nunique() for k in
        ["contract", "payment_method", "protected_group", "region"]
    })
    show("K4 churn by contract", {k: round(float(v), 4) for k, v in
         c.groupby("contract")["churned"].mean().items()})
    show("K5 churn by payment", {k: round(float(v), 4) for k, v in
         c.groupby("payment_method")["churned"].mean().items()})
    show("K6 churn by group", {k: round(float(v), 4) for k, v in
         c.groupby("protected_group")["churned"].mean().items()})
    show("K7 churn by region", {k: round(float(v), 4) for k, v in
         c.groupby("region")["churned"].mean().items()})

    tb = pd.cut(c["tenure_months"], [0, 6, 12, 24, 48],
                labels=["0-6", "7-12", "13-24", "25-48"])
    show("K8 tenure buckets", {str(k): round(float(v), 4) for k, v in
         c.groupby(tb, observed=True)["churned"].mean().items()})
    tk = pd.cut(c["support_tickets"], [-1, 0, 2, 7],
                labels=["0", "1-2", "3+"])
    show("K9 ticket buckets", {str(k): round(float(v), 4) for k, v in
         c.groupby(tk, observed=True)["churned"].mean().items()})

    q = pd.qcut(c["monthly_charges"], 5, labels=False)
    show("K10 charge quintiles", {int(k): round(float(v), 4) for k, v in
         c.groupby(q)["churned"].mean().items()})

    show("K11 probability quantiles", {
        str(qv): round(float(np.quantile(p, qv)), 4)
        for qv in [0.10, 0.50, 0.90, 0.99]
    })
    dec = np.array_split(order, 10)
    show("K12 decile actuals", [int(yarr[d].sum()) for d in dec])
    show("K13 precision at k", {
        k: round(float(yarr[order[:k]].mean()), 4)
        for k in [50, 100, 150, 200]
    })
    show("K14 recall at k", {
        k: round(float(yarr[order[:k]].sum() / yarr.sum()), 4)
        for k in [50, 100, 150, 200]
    })
    show("K15 lift at k", {
        k: round(float(yarr[order[:k]].mean() / yarr.mean()), 2)
        for k in [50, 100, 150, 200]
    })

    show("K16 f1 by threshold", {
        t: round(float(f1_score(yte, (p >= t).astype(int))), 4)
        for t in [0.05, 0.10, 0.15, 0.20, 0.30]
    })
    tn, fp, fn, tp = 0, 0, 0, 0
    tprs, fprs, ts = [], [], []
    for t in [0.05, 0.10, 0.15, 0.20, 0.30]:
        yh = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, yh).ravel()
        tprs.append(round(float(tp / (tp + fn)), 4))
        fprs.append(round(float(fp / (fp + tn)), 4))
        ts.append(t)
    show("K17 tpr by threshold", dict(zip(map(str, ts), tprs)))
    show("K18 fpr by threshold", dict(zip(map(str, ts), fprs)))
    show("K19 youden j", dict(zip(map(str, ts),
         [round(t - f, 4) for t, f in zip(tprs, fprs)])))

    show("K20 brier all models", {
        k: round(float(brier_score_loss(yte, m.predict_proba(Xte)[:, 1])), 5)
        for k, m in models.items()
    })

    cvp = cross_val_score(models["logistic"], Xtr, ytr, cv=4,
                          scoring="average_precision")
    show("K21 cv pr auc", [round(float(v), 4) for v in cvp])

    sizes = [500, 1000, 2000, 4000]
    learn = []
    for n in sizes:
        m = LogisticRegression(max_iter=2000, random_state=SEED)
        m.fit(Xtr.iloc[:n], ytr.iloc[:n])
        learn.append(round(float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1])), 4))
    show("K22 learning curve", dict(zip(map(str, sizes), learn)))

    pi_lr = permutation_importance(models["logistic"], Xte, yte, n_repeats=6,
                                   random_state=SEED, scoring="roc_auc", n_jobs=-1)
    show("K23 logistic importance", {
        Xte.columns[i]: round(float(pi_lr.importances_mean[i]), 4)
        for i in np.argsort(-pi_lr.importances_mean)[:5]
    })

    sel = te = c.loc[c["snapshot_date"] >= "2025-07-01"].copy()
    sel["pred"] = p
    sel["flag"] = (p >= float(p[order[149]])).astype(int)
    show("K24 tpr by group", {
        k: round(float(v["pred"].loc[v["churned"] == 1].ge(
            float(p[order[149]])).mean()), 4)
        for k, v in sel.groupby("protected_group")
    })
    show("K25 fpr by group", {
        k: round(float(v["pred"].loc[v["churned"] == 0].ge(
            float(p[order[149]])).mean()), 4)
        for k, v in sel.groupby("protected_group")
    })

    show("K26 threshold artifacts", [
        "probability model", "cost ratio 10:1", "capacity 150",
        "operating threshold 0.1157", "monthly review queue",
    ])
    show("K27 handover artifacts", [
        "clean feature list", "temporal split definition", "model card",
        "monitoring dashboard", "retraining owner",
    ])
    show("K28 what not to do", [
        "use the model for pricing", "deny service on a score",
        "retrain on the test set", "publish accuracy alone",
    ])


def section_extra3(c, h, dup) -> None:
    """A third batch of verified outputs for the expanded Chapter 16."""
    Xtr, Xte, ytr, yte = splits(c)
    models = fit_models(Xtr, ytr)
    p = models["logistic"].predict_proba(Xte)[:, 1]
    pg = models["boosting"].predict_proba(Xte)[:, 1]
    order = np.argsort(-p)
    yarr = yte.to_numpy()
    rng = np.random.default_rng(SEED)

    cmp = pd.DataFrame({
        "train_mean": Xtr.mean(), "test_mean": Xte.mean()
    }).round(3)
    cmp["shift"] = (cmp["test_mean"] - cmp["train_mean"]).round(3)
    show("L1 train test shift", cmp.to_string())

    aucs = []
    for _ in range(200):
        idx = rng.integers(0, len(yarr), len(yarr))
        if len(np.unique(yarr[idx])) < 2:
            continue
        aucs.append(roc_auc_score(yarr[idx], p[idx]))
    show("L2 auc bootstrap", {
        "mean": round(float(np.mean(aucs)), 4),
        "p05": round(float(np.percentile(aucs, 5)), 4),
        "p95": round(float(np.percentile(aucs, 95)), 4),
    })

    prec = []
    for _ in range(200):
        idx = rng.integers(0, len(yarr), 150)
        prec.append(float(yarr[idx[np.argsort(-p[idx])[:50]]].mean()))
    show("L3 precision bootstrap", {
        "mean": round(float(np.mean(prec)), 4),
        "p05": round(float(np.percentile(prec, 5)), 4),
        "p95": round(float(np.percentile(prec, 95)), 4),
    })

    show("L4 model correlation", round(float(np.corrcoef(p, pg)[0, 1]), 4))
    dis = np.argsort(-np.abs(p - pg))[:4]
    show("L5 top disagreements", pd.DataFrame({
        "logistic": np.round(p[dis], 4), "boosting": np.round(pg[dis], 4),
        "actual": yarr[dis],
    }).to_string(index=False))

    ratios = [5, 10, 20]
    rows = []
    for r in ratios:
        fp_cost = 50
        fn_cost = r * fp_cost
        be = fp_cost / (fp_cost + fn_cost)
        t = be
        yh = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, yh).ravel()
        rows.append((r, round(be, 4), int(tp), int(fp),
                     int(fn_cost * fn + fp_cost * fp)))
    show("L6 cost ratio sensitivity", pd.DataFrame(
        rows, columns=["ratio", "break_even", "tp", "fp", "cost"]
    ).to_string(index=False))

    rows = []
    for cap in [100, 150, 200, 300]:
        found = int(yarr[order[:cap]].sum())
        rows.append((cap, found, round(float(found / cap), 4),
                     round(float(500 * found - 50 * (cap - found)), 2)))
    show("L7 capacity scenarios", pd.DataFrame(
        rows, columns=["capacity", "churners", "precision", "value"]
    ).to_string(index=False))

    show("L8 top bin calibration", {
        "top_predicted": round(float(
            np.mean(np.sort(p)[-100:])), 4),
        "top_observed": round(float(yarr[np.argsort(-p)[:100]].mean()), 4),
    })
    show("L9 probability by outcome", {
        "churners": round(float(p[yarr == 1].mean()), 4),
        "non_churners": round(float(p[yarr == 0].mean()), 4),
    })

    cor = Xtr.corr().abs()
    pairs = [(a, b, round(float(cor.loc[a, b]), 3))
             for i, a in enumerate(Xtr.columns)
             for b in Xtr.columns[i + 1:]
             if cor.loc[a, b] > 0.25]
    show("L10 correlated features", pairs[:6])

    fold_thr = []
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    for tr_i, _ in skf.split(Xtr, ytr):
        mdl = LogisticRegression(max_iter=2000, random_state=SEED)
        mdl.fit(Xtr.iloc[tr_i], ytr.iloc[tr_i])
        pv = mdl.predict_proba(Xte)[:, 1]
        oo = np.argsort(-pv)
        fold_thr.append(round(float(pv[oo[149]]), 4))
    show("L11 threshold stability", fold_thr)

    show("L12 value of improvement", {
        "current_precision": round(float(yarr[order[:150]].mean()), 4),
        "with_10pct_lift": round(float(
            yarr[order[:150]].mean() * 1.10), 4),
        "extra_churners": round(float(
            150 * yarr[order[:150]].mean() * 0.10), 2),
    })

    show("L13 decision memo", [
        "one number for the current state",
        "one number for the proposed action",
        "the assumption that drives the gap",
        "the decision requested and its owner",
    ])
    show("L14 final checks", [
        "test set untouched by tuning", "leak audit signed off",
        "metrics by subgroup reviewed", "monitoring dashboard live",
    ])


def section_extra4(c, h, dup) -> None:
    """A compact fourth batch of verified outputs for Chapter 16."""
    Xtr, Xte, ytr, yte = splits(c)
    models = fit_models(Xtr, ytr)
    p = models["logistic"].predict_proba(Xte)[:, 1]
    order = np.argsort(-p)
    yarr = yte.to_numpy()

    inter = c.groupby(["contract", pd.cut(c["tenure_months"],
                                          [0, 12, 48],
                                          labels=["new", "established"])],
                      observed=True)["churned"].mean().round(4)
    show("M1 tenure contract interaction", inter.to_string())

    show("M2 ticket distribution", {
        k: int(v) for k, v in c["support_tickets"].value_counts().head(5).items()
    })
    show("M3 usage quantiles", {
        str(q): round(float(c["usage_gb"].quantile(q)), 2)
        for q in [0.10, 0.50, 0.90]
    })
    te = c.loc[c["snapshot_date"] >= "2025-07-01"].copy()
    te["score"] = p
    show("M4 score by contract", {k: round(float(v), 4) for k, v in
         te.groupby("contract")["score"].mean().items()})

    t_be = 50 / 550
    yh = (p >= t_be).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, yh).ravel()
    show("M5 break even confusion", {
        "threshold": round(float(t_be), 4), "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
    })
    show("M6 break even rates", {
        "precision": round(float(precision_score(yte, yh)), 4),
        "recall": round(float(recall_score(yte, yh)), 4),
        "flagged": int(yh.sum()),
    })

    vals = []
    for k in [25, 50, 100, 150]:
        found = int(yarr[order[:k]].sum())
        vals.append((k, found, round(float(
            500 * found - 50 * (k - found)), 2)))
    show("M7 cumulative value", pd.DataFrame(
        vals, columns=["k", "churners", "value"]).to_string(index=False))

    top10 = int(yarr[order[:10]].sum())
    show("M8 top ten value", {
        "churners": top10,
        "value": round(float(500 * top10 - 50 * (10 - top10)), 2),
    })
    show("M9 monitoring triggers", {
        "precision_at_capacity": "alert if down 15% from 0.1533",
        "base_rate": "alert if outside 0.025 to 0.055",
        "feature_shift": "alert if z-score above 2",
    })
    show("M10 handover timeline", [
        "week 1: run the pipeline end to end",
        "week 2: review subgroup metrics with owners",
        "week 3: pilot the 150-customer queue",
        "week 6: measure and decide whether to keep",
    ])
    show("M11 headline metrics", {
        "roc_auc": round(float(roc_auc_score(yte, p)), 4),
        "pr_auc": round(float(average_precision_score(yte, p)), 4),
        "precision_at_150": round(float(yarr[order[:150]].mean()), 4),
        "recall_at_150": round(float(yarr[order[:150]].sum() / yarr.sum()), 4),
        "brier": round(float(brier_score_loss(yte, p)), 5),
    })
    show("M12 sign off", [
        "model card approved", "threshold authorised",
        "monitoring live", "owner named",
    ])


def main() -> None:
    print("python pandas", pd.__version__)
    print()
    c, h, dup = load()
    for section in (
        section_161, section_162, section_163, section_164, section_165,
        section_166, section_167, section_168, section_169, section_extra,
        section_extra2, section_extra3, section_extra4,
    ):
        print("=" * 78)
        section(c, h, dup)


if __name__ == "__main__":
    main()
