#!/usr/bin/env python3
"""Generate the predictive-analytics fixtures and figures Chapter 16 uses.

Run from the project root:

    python scripts/generate_chapter16_sources.py

Outputs:

    data/raw/ml/ml_churn.csv          6,000 customers, ~4% churn, with leaky columns
    data/raw/ml/ml_listings.csv       2,000 property listings with a nonlinear price
    data/raw/ml/ml_dup_entities.csv   12 rows for six duplicated entities
    images/ch16_*.png                 nine figures used in the chapter

Everything is deterministic, including the figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "data" / "raw" / "ml"
IMAGES = ROOT / "images"
SEED = 1607

CLEAN = [
    "tenure_months", "monthly_charges", "support_tickets", "num_products",
    "usage_gb", "contract_monthly", "payment_check", "group_B",
    "region_Northeast", "region_South", "region_West",
]
LEAKY = ["legacy_churn_score", "cancellation_flag", "final_invoice_amount"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def churn() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = 6000
    quarters = pd.date_range("2024-01-01", periods=8, freq="QS")
    snapshot = rng.choice(quarters.strftime("%Y-%m-%d"), size=n)

    tenure = rng.integers(1, 49, size=n)
    charges = np.round(rng.normal(72, 24, size=n).clip(18, 160), 2)
    tickets = rng.poisson(0.9, size=n)
    products = rng.integers(1, 5, size=n)
    usage = np.round(rng.gamma(3.0, 12.0, size=n).clip(0, 260), 1)
    contract = rng.choice(["monthly", "annual", "two_year"], size=n,
                          p=[0.60, 0.28, 0.12])
    payment = rng.choice(["card", "bank", "check"], size=n,
                         p=[0.60, 0.30, 0.10])
    group = rng.choice(["A", "B"], size=n, p=[0.72, 0.28])
    region = rng.choice(["Midwest", "Northeast", "South", "West"], size=n)

    # Group B buys richer plans, so monthly_charges acts as a proxy for group.
    charges = np.round(charges + 6.0 * (group == "B"), 2)

    logit = (
        -4.55
        + 0.55 * tickets
        - 0.045 * tenure
        + 0.011 * charges
        + 1.05 * (contract == "monthly")
        + 0.55 * (payment == "check")
        + 0.30 * (group == "B")
        - 0.006 * usage
        + 0.25 * (products == 1)
        + 0.80 * ((tickets >= 3) & (contract == "monthly"))
        + 0.70 * (usage < 10)
    )
    p = sigmoid(logit)
    churned = rng.binomial(1, p, size=n)

    df = pd.DataFrame({
        "customer_id": [f"C-{i + 1:05d}" for i in range(n)],
        "snapshot_date": snapshot,
        "tenure_months": tenure,
        "monthly_charges": charges,
        "support_tickets": tickets,
        "num_products": products,
        "usage_gb": usage,
        "contract": contract,
        "payment_method": payment,
        "protected_group": group,
        "region": region,
        "churned": churned,
    })
    # encoded convenience columns
    df["contract_monthly"] = (df["contract"] == "monthly").astype(int)
    df["payment_check"] = (df["payment_method"] == "check").astype(int)
    df["group_B"] = (df["protected_group"] == "B").astype(int)
    for reg in ["Northeast", "South", "West"]:
        df[f"region_{reg}"] = (df["region"] == reg).astype(int)

    # three deliberate leakage paths
    df["legacy_churn_score"] = np.round(np.where(
        churned == 1, rng.uniform(0.70, 0.99, size=n),
        rng.uniform(0.01, 0.30, size=n)), 3)
    df["cancellation_flag"] = np.where(churned == 1, "yes", "no")
    df["final_invoice_amount"] = np.round(np.where(
        churned == 1, rng.uniform(180, 900, size=n),
        rng.uniform(0, 120, size=n)), 2)

    df.to_csv(ML / "ml_churn.csv", index=False)
    print(f"  ml_churn.csv: {len(df)} rows, churn rate {df['churned'].mean():.4f}")
    return df


def listings() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    n = 2000
    rooms = rng.integers(1, 6, size=n)
    size = np.round(rooms * 300 + rng.normal(200, 75, size=n), 1).clip(260)
    location = rng.choice(["A", "B", "C"], size=n, p=[0.45, 0.35, 0.20])
    rating = np.round(rng.uniform(3.0, 5.0, size=n), 1)
    amenities = rng.integers(0, 6, size=n)
    age = rng.integers(0, 60, size=n)
    loc = np.select([location == "A", location == "B"],
                    [0.0, 28000.0], default=62000.0)
    price = (
        48000
        + 92 * size
        + 5900 * rooms
        + loc
        + 8800 * rating
        + 3900 * amenities
        - 520 * age
        + 26000 * ((location == "C") & (size > 1800))
        + 14000 * ((rating >= 4.5) & (amenities >= 4))
        + rng.normal(0, 21000, size=n)
    )
    df = pd.DataFrame({
        "listing_id": [f"L-{i + 1:05d}" for i in range(n)],
        "size_sqft": size,
        "rooms": rooms,
        "location": location,
        "rating": rating,
        "amenities": amenities,
        "age_years": age,
        "price": np.round(price, 2),
    })
    df.to_csv(ML / "ml_listings.csv", index=False)
    print(f"  ml_listings.csv: {len(df)} listings")
    return df


def duplicates() -> pd.DataFrame:
    rows = []
    for i in range(1, 7):
        rows.append((f"A{i:03d}", f"Customer {i}", "2025-01-01", 0))
        rows.append((f"A{i:03d}", f"Cust. {i}", "2025-01-02", 1))
    df = pd.DataFrame(rows, columns=[
        "entity_key", "name_variant", "snapshot_date", "churned"])
    df.to_csv(ML / "ml_dup_entities.csv", index=False)
    print(f"  ml_dup_entities.csv: {len(df)} rows / 6 entities")
    return df


def figures(c: pd.DataFrame, listings: pd.DataFrame) -> None:
    """Train a few models and save the chapter's figures."""
    tr = c["snapshot_date"] < "2025-07-01"
    te = ~tr
    Xtr, Xte = c.loc[tr, CLEAN], c.loc[te, CLEAN]
    ytr, yte = c.loc[tr, "churned"], c.loc[te, "churned"]

    lr = LogisticRegression(max_iter=2000, random_state=SEED)
    lr.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                    learning_rate=0.05, random_state=SEED)
    gb.fit(Xtr, ytr)
    p_gb = gb.predict_proba(Xte)[:, 1]
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
                                random_state=SEED, n_jobs=-1)
    rf.fit(Xtr, ytr)
    p_rf = rf.predict_proba(Xte)[:, 1]

    Xtr_l = c.loc[tr, CLEAN + LEAKY].copy()
    Xte_l = c.loc[te, CLEAN + LEAKY].copy()
    for frame in (Xtr_l, Xte_l):
        frame["cancellation_flag"] = (frame["cancellation_flag"] == "yes").astype(int)
    leak = LogisticRegression(max_iter=2000, random_state=SEED)
    leak.fit(Xtr_l, ytr)
    p_leak = leak.predict_proba(Xte_l)[:, 1]

    # 1. leakage ROC comparison
    f, ax = plt.subplots(figsize=(5.2, 4.2))
    for label, yscore, colour in [
        ("leaky model", p_leak, "#c0392b"),
        ("clean logistic", p_lr, "#2c3e50"),
        ("clean boosting", p_gb, "#2980b9"),
    ]:
        fpr, tpr, _ = roc_curve(yte, yscore)
        ax.plot(fpr, tpr, label=f"{label} (AUC {roc_auc_score(yte, yscore):.3f})",
                color=colour, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Leakage inflates apparent performance")
    ax.legend(loc="lower right", fontsize=8)
    f.tight_layout(); f.savefig(IMAGES / "ch16_leakage.png", dpi=130); plt.close(f)

    # 2. ROC and PR curves
    f, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for label, yscore in [("logistic", p_lr), ("boosting", p_gb), ("forest", p_rf)]:
        fpr, tpr, _ = roc_curve(yte, yscore)
        axes[0].plot(fpr, tpr, lw=2, label=label)
        pr, rc, _ = precision_recall_curve(yte, yscore)
        axes[1].plot(rc, pr, lw=2, label=label)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="ROC")
    axes[1].axhline(yte.mean(), color="k", ls="--", lw=1,
                    label=f"base rate {yte.mean():.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    for a in axes:
        a.legend(fontsize=8)
    f.tight_layout(); f.savefig(IMAGES / "ch16_roc_pr.png", dpi=130); plt.close(f)

    # 3. confusion matrix
    yhat = (p_lr >= 0.15).astype(int)
    cm = confusion_matrix(yte, yhat)
    f, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["pred 0", "pred 1"], yticklabels=["actual 0", "actual 1"],
           title="Confusion matrix at p = 0.15", xlabel="", ylabel="")
    f.colorbar(im, ax=ax, shrink=0.8)
    f.tight_layout(); f.savefig(IMAGES / "ch16_confusion.png", dpi=130); plt.close(f)

    # 4. calibration / reliability
    f, ax = plt.subplots(figsize=(5.2, 4.2))
    for label, yscore, colour in [("logistic", p_lr, "#2c3e50"),
                                  ("boosting", p_gb, "#2980b9")]:
        prob_true, prob_pred = calibration_curve(
            yte, yscore, n_bins=8, strategy="quantile")
        ax.plot(prob_pred, prob_true, "o-", color=colour, label=label)
    ax.plot([0, max(p_gb.max(), 0.2)], [0, max(p_gb.max(), 0.2)],
            "k--", lw=1, label="perfect")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction",
           title="Reliability diagram")
    ax.legend(fontsize=8)
    f.tight_layout(); f.savefig(IMAGES / "ch16_calibration.png", dpi=130); plt.close(f)

    # 5. lift and cumulative gain
    order = np.argsort(-p_lr)
    ys = yte.to_numpy()[order]
    gain = np.cumsum(ys) / ys.sum()
    lift = gain / (np.arange(1, len(ys) + 1) / len(ys))
    f, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(np.arange(1, len(ys) + 1) / len(ys), gain, lw=2, label="model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set(xlabel="Fraction of customers reviewed", ylabel="Fraction of churners found",
           title="Cumulative gain")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(ys) + 1) / len(ys), lift, color="#c0392b", lw=1, alpha=0.6)
    ax2.set_ylabel("Lift", color="#c0392b")
    f.tight_layout(); f.savefig(IMAGES / "ch16_lift_gain.png", dpi=130); plt.close(f)

    # 6. permutation importance
    pi = permutation_importance(lr, Xte, yte, n_repeats=6,
                                random_state=SEED, scoring="roc_auc", n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values()
    f, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.barh(imp.index, imp.values, xerr=pi.importances_std[
        imp.index.map({k: i for i, k in enumerate(Xte.columns)})], color="#2980b9")
    ax.set(xlabel="Drop in ROC AUC", title="Permutation importance (test set)")
    f.tight_layout(); f.savefig(IMAGES / "ch16_importance.png", dpi=130); plt.close(f)

    # 7. threshold economics
    thresholds = np.linspace(0.02, 0.60, 60)
    costs = []
    for t in thresholds:
        yh = (p_lr >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, yh).ravel()
        costs.append(500 * fn + 50 * fp)
    f, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(thresholds, costs, lw=2, color="#c0392b")
    best = int(np.argmin(costs))
    ax.axvline(thresholds[best], color="k", ls="--", lw=1)
    ax.set(xlabel="Decision threshold", ylabel="Expected cost ($)",
           title="Cost of errors by threshold")
    ax.annotate(f"min near {thresholds[best]:.2f}",
                (thresholds[best], costs[best]), fontsize=8)
    f.tight_layout(); f.savefig(IMAGES / "ch16_threshold.png", dpi=130); plt.close(f)

    # 8. bias-variance / regularisation curve
    depths = range(1, 11)
    train_auc, test_auc = [], []
    for d in depths:
        m = GradientBoostingClassifier(n_estimators=60, max_depth=int(d),
                                       learning_rate=0.05, random_state=SEED)
        m.fit(Xtr, ytr)
        train_auc.append(roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1]))
        test_auc.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    f, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(list(depths), train_auc, "o-", label="train")
    ax.plot(list(depths), test_auc, "o-", label="test")
    ax.set(xlabel="Tree depth", ylabel="ROC AUC", title="Bias–variance trade-off")
    ax.legend(fontsize=8)
    f.tight_layout(); f.savefig(IMAGES / "ch16_bias_variance.png", dpi=130); plt.close(f)

    # 9. drift across quarters
    dr = c.groupby("snapshot_date").agg(
        charges=("monthly_charges", "mean"), usage=("usage_gb", "mean"),
        tickets=("support_tickets", "mean"), churn=("churned", "mean"))
    f, ax = plt.subplots(figsize=(6.2, 4.2))
    for col in ["charges", "usage", "tickets"]:
        z = (dr[col] - dr[col].mean()) / dr[col].std()
        ax.plot(dr.index, z, "o-", label=col)
    ax.axhline(0, color="k", lw=1)
    ax.set(title="Standardised feature means by quarter", ylabel="z-score")
    ax.legend(fontsize=8)
    f.tight_layout(); f.savefig(IMAGES / "ch16_drift.png", dpi=130); plt.close(f)


def main() -> None:
    ML.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    print("generating Chapter 16 predictive-analytics fixtures ...")
    c = churn()
    h = listings()
    duplicates()
    print("drawing figures ...")
    figures(c, h)
    for p in sorted(IMAGES.glob("ch16_*.png")):
        print(f"  {p.name}: {p.stat().st_size:,} bytes")
    print("done.")


if __name__ == "__main__":
    main()
