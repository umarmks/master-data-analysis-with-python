#!/usr/bin/env python3
"""Print every output block used in Chapter 6, from the current files.

Run from the project root:

    python scripts/chapter06_outputs.py

Reads the fixtures written by generate_chapter06_sources.py, under
data/raw/audit/.
"""

from __future__ import annotations

import difflib
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "raw" / "audit"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 80)


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def read_messy() -> pd.DataFrame:
    return pd.read_csv(
        AUDIT / "orders_messy_40k.csv",
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )


def read_customers() -> pd.DataFrame:
    return pd.read_csv(
        AUDIT / "customers_messy.csv", dtype=str, keep_default_na=False, na_values=[""]
    )


# ============================================================ 6.1 mechanisms
def section_61() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.normal(50, 10, n)

    mcar = rng.random(n) < 0.20
    y = rng.normal(0, 1, n)
    mar = rng.random(n) < 1.0 / (1.0 + np.exp(-(0.9 * y)))
    mnar = x > np.quantile(x, 0.80)

    rows = [
        ("complete data", n, round(float(x.mean()), 3)),
        ("MCAR (dropped at random)", int((~mcar).sum()), round(float(x[~mcar].mean()), 3)),
        ("MAR (dropped via y)", int((~mar).sum()), round(float(x[~mar].mean()), 3)),
        ("MNAR (dropped high x)", int((~mnar).sum()), round(float(x[~mnar].mean()), 3)),
    ]
    show(
        "three mechanisms, same data",
        pd.DataFrame(rows, columns=["scenario", "rows_kept", "observed_mean"]).to_string(index=False),
    )
    show("true mean", round(float(x.mean()), 3))

    ic = pd.read_csv(AUDIT / "income_churn.csv")
    ic["income_missing"] = ic["income"].isna()
    by_seg = ic.groupby("segment").agg(
        n=("income", "size"),
        missing=("income_missing", "mean"),
        churn=("churn", "mean"),
    ).round(4)
    show("income missingness and churn by segment", by_seg.to_string())

    within = ic.groupby(["segment", "income_missing"])["churn"].mean().round(4)
    show("churn rate by missing status within segment", within.to_dict())
    show("overall churn by missing status", ic.groupby("income_missing")["churn"].mean().round(4).to_dict())

    # is missingness associated with churn once we condition on segment?
    rng2 = np.random.default_rng(7)
    strata = [g["churn"].to_numpy() for _, g in ic.groupby("segment")]
    flags = [g["income_missing"].to_numpy() for _, g in ic.groupby("segment")]

    def pooled_gap(strata_flags: list) -> float:
        num, den = 0.0, 0
        for y, f in zip(strata, strata_flags):
            if f.sum() == 0 or (~f).sum() == 0:
                continue
            num += (y[f].mean() - y[~f].mean()) * len(y)
            den += len(y)
        return num / den

    obs = pooled_gap(flags)
    perm = [pooled_gap([rng2.permutation(f) for f in flags]) for _ in range(2000)]
    p_perm = (1 + sum(abs(v) >= abs(obs) for v in perm)) / (2000 + 1)
    show("stratified test: churn vs missingness, within segment",
         {"observed_gap": round(float(obs), 4), "permutation_p": round(float(p_perm), 3)})


# ======================================================= 6.2 quantifying missing
def section_62() -> None:
    df = read_messy()
    miss = pd.DataFrame(
        {
            "missing": df.isna().sum(),
            "pct": (df.isna().mean() * 100).round(3),
        }
    )
    miss = miss[miss["missing"] > 0].sort_values("missing", ascending=False)
    show("missing counts and percentages", miss.to_string())
    show("cells missing", {"total_cells": int(df.size), "missing_cells": int(df.isna().sum().sum()),
                            "pct": round(float(100 * df.isna().sum().sum() / df.size), 4)})

    # fake nulls to real nulls, first
    tokens = {"", "N/A", "n/a", "-1", "unknown", "Unknown", "9999", "not available"}
    fake = {}
    for col in df.columns:
        hits = int(df[col].astype(str).isin(tokens).sum())
        if hits:
            fake[col] = hits
    show("look-alike nulls before conversion", fake)

    conv = df.replace({"quantity": {"9999": np.nan}, "discount_pct": {"9999": np.nan},
                       "customer_id": {"-1": np.nan}, "region": {"unknown": np.nan, "N/A": np.nan}})
    show("missing after conversion", {"customer_id": int(conv["customer_id"].isna().sum()),
                                      "region": int(conv["region"].isna().sum()),
                                      "quantity": int(conv["quantity"].isna().sum()),
                                      "discount_pct": int(conv["discount_pct"].isna().sum())})

    cols = ["customer_id", "region", "discount_pct", "quantity"]
    per_row = conv[cols].isna().sum(axis=1)
    show("missing columns per row (of 4)", per_row.value_counts().sort_index().to_dict())

    co = conv[cols].isna()
    co = co.T.dot(co)
    show("column co-occurrence of missingness", co.to_string())

    parsed = pd.to_datetime(conv["order_date"], errors="coerce", format="%Y-%m-%d")
    by_month = (
        conv.assign(month=parsed.dt.strftime("%Y-%m"),
                    region_missing=conv["region"].isna())
        .dropna(subset=["month"])
        .groupby("month")["region_missing"].agg(["size", "mean"])
    )
    by_month.columns = ["rows", "region_missing"]
    show("region missingness by month (head/tail)", pd.concat([by_month.head(3), by_month.tail(3)]).round(4).to_string())

    show("region missingness by channel",
         conv.assign(region_missing=conv["region"].isna()).groupby("channel")["region_missing"].mean().round(4).to_dict())


# ============================================================ 6.3 imputation
def section_63() -> None:
    ic = pd.read_csv(AUDIT / "income_churn.csv")
    n = len(ic)
    show("listwise vs columnwise deletion", {
        "rows": n,
        "listwise (dropna)": int(ic.dropna().shape[0]),
        "lost": int(n - ic.dropna().shape[0]),
        "lost_pct": round(100 * (n - ic.dropna().shape[0]) / n, 2),
    })

    inc = ic["income"]
    row = pd.DataFrame(
        [
            ("complete case", round(float(inc.mean()), 2), 0),
            ("mean", round(float(inc.fillna(inc.mean()).mean()), 2), int(inc.isna().sum())),
            ("median", round(float(inc.fillna(inc.median()).mean()), 2), int(inc.isna().sum())),
            ("constant 0", round(float(inc.fillna(0).mean()), 2), int(inc.isna().sum())),
        ],
        columns=["method", "mean_income", "cells_filled"],
    )
    show("imputation changes the mean", row.to_string(index=False))

    # group-based: median within segment
    grp = ic.copy()
    grp["income"] = grp.groupby("segment")["income"].transform(lambda s: s.fillna(s.median()))
    show("group median by segment", {
        "mean_income": round(float(grp["income"].mean()), 2),
        "by_segment_median": ic.groupby("segment")["income"].median().round(2).to_dict(),
    })

    # forward / backward fill and interpolation on a small monthly series
    ts = pd.DataFrame(
        {"month": ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"],
         "revenue": [120.0, np.nan, 138.0, np.nan, np.nan, 175.0]}
    )
    ts["ffill"] = ts["revenue"].ffill()
    ts["bfill"] = ts["revenue"].bfill()
    ts["linear"] = ts["revenue"].interpolate(method="linear")
    show("forward / backward / linear fill", ts.round(2).to_string(index=False))

    # regression imputation
    reg_data = ic.dropna(subset=["income"]).copy()
    reg_data["segment_plus"] = (reg_data["segment"] == "Plus").astype(int)
    reg_data["segment_premium"] = (reg_data["segment"] == "Premium").astype(int)
    feats = ["tenure_months", "segment_plus", "segment_premium"]
    model = LinearRegression().fit(reg_data[feats], reg_data["income"])
    imp = ic.copy()
    imp["segment_plus"] = (imp["segment"] == "Plus").astype(int)
    imp["segment_premium"] = (imp["segment"] == "Premium").astype(int)
    pred = model.predict(imp[feats])
    imp["income_reg"] = imp["income"].fillna(pd.Series(pred, index=imp.index))
    show("regression imputation", {"r2_on_complete": round(float(model.score(reg_data[feats], reg_data["income"])), 3),
                                    "mean_income_after": round(float(imp["income_reg"].mean()), 2)})

    # k-NN imputation
    kn = ic[["tenure_months", "income"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(kn[["tenure_months", "income"]])
    filled = KNNImputer(n_neighbors=5).fit_transform(scaled)
    kn["income_knn"] = scaler.inverse_transform(filled)[:, 1]
    show("k-NN imputation", {"mean_income_after": round(float(kn["income_knn"].mean()), 2)})

    # missingness indicator
    ic2 = ic.assign(income_missing=ic["income"].isna()).copy()
    ic2["income_med"] = ic2["income"].fillna(ic2["income"].median())
    show("missingness indicator as a feature", {
        "missing_indicator_mean": round(float(ic2["income_missing"].mean()), 4),
        "corr_indicator_churn": round(float(ic2["income_missing"].astype(int).corr(ic2["churn"])), 4),
    })

    # sensitivity check: does the income-churn story depend on the method?
    def churn_high_vs_low(frame: pd.DataFrame, col: str) -> dict:
        med = frame[col].median()
        hi = frame.loc[frame[col] >= med, "churn"].mean()
        lo = frame.loc[frame[col] < med, "churn"].mean()
        return {"high_income_churn": round(float(hi), 3), "low_income_churn": round(float(lo), 3),
                "gap": round(float(lo - hi), 3)}

    complete = ic.dropna(subset=["income"])
    med_fill = ic.assign(income=ic["income"].fillna(ic["income"].median()))
    grp_fill = grp
    reg_fill = ic.assign(income=pd.Series(pred, index=ic.index).where(ic["income"].isna(), ic["income"]))
    sens = pd.DataFrame(
        [
            ("complete case", len(complete), churn_high_vs_low(complete, "income")["gap"]),
            ("median", len(med_fill), churn_high_vs_low(med_fill, "income")["gap"]),
            ("group median", len(grp_fill), churn_high_vs_low(grp_fill, "income")["gap"]),
            ("regression", len(reg_fill), churn_high_vs_low(reg_fill, "income")["gap"]),
        ],
        columns=["method", "rows", "churn_gap_lo_minus_hi"],
    )
    show("sensitivity of the churn gap to the imputation method", sens.to_string(index=False))


# ============================================================= 6.4 duplicates
def section_64() -> None:
    df = read_messy()
    show("exact duplicates", {"rows": len(df), "exact_duplicate_rows": int(df.duplicated().sum()),
                              "distinct_rows": int(df.drop_duplicates().shape[0])})

    key_cols = ["order_id", "product_id"]
    dup_key = df.duplicated(subset=key_cols, keep=False)
    show("business-key duplicates", {"key": key_cols, "rows_in_groups": int(dup_key.sum()),
                                     "groups": int(df.loc[dup_key].groupby(key_cols).ngroups)})

    rev = pd.to_numeric(df["revenue"], errors="coerce")
    spread = df.loc[dup_key].groupby(key_cols)["revenue"].nunique()
    show("duplicate keys with conflicting revenue", int((spread > 1).sum()))

    # keep-first / keep-last / keep-most-complete
    def most_complete(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(_miss=frame.isna().sum(axis=1)).sort_values(
            ["order_id", "product_id", "_miss"], kind="stable"
        ).drop_duplicates(subset=key_cols, keep="first").drop(columns="_miss")

    first = df.drop_duplicates(subset=key_cols, keep="first")
    last = df.drop_duplicates(subset=key_cols, keep="last")
    mc = most_complete(df)
    show("dedup policy row counts", {"keep_first": len(first), "keep_last": len(last),
                                     "keep_most_complete": len(mc)})
    show("gross revenue under each policy",
         {"keep_first": round(float(pd.to_numeric(first["revenue"], errors="coerce").sum()), 2),
          "keep_last": round(float(pd.to_numeric(last["revenue"], errors="coerce").sum()), 2),
          "keep_most_complete": round(float(pd.to_numeric(mc["revenue"], errors="coerce").sum()), 2)})

    # near-duplicate detection inside a key: values that differ only by case/space
    def near_groups(values: pd.Series) -> int:
        seen = {}
        for v in values.dropna().astype(str):
            seen.setdefault(v.strip().lower(), set()).add(v)
        return sum(1 for s in seen.values() if len(s) > 1)

    show("near-duplicate category strings", near_groups(df["category"]))
    show("near-duplicate region strings", near_groups(df["region"]))

    # fuzzy match on customer names (difflib + blocking by zip)
    cust = read_customers()
    truth = pd.read_csv(AUDIT / "households_truth.csv", dtype=str)

    def strip_accents(text: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFKD", text)
                       if not unicodedata.combining(c))

    def norm_name(raw: str) -> str:
        raw = strip_accents(str(raw)).upper().replace(".", "").replace(",", " , ")
        return " ".join(raw.split())

    def surname(raw: str) -> str:
        raw = str(raw).strip()
        if "," in raw:
            return norm_name(raw.split(",", 1)[0])
        parts = norm_name(raw).split()
        return parts[-1] if len(parts) > 1 else ""

    def norm_addr(raw: str) -> str:
        raw = strip_accents(str(raw)).upper()
        raw = raw.replace(".", "").replace(",", "")
        raw = raw.replace(" STREET", " ST").replace(" AVENUE", " AVE").replace(" DRIVE", " DR")
        raw = raw.replace(" LANE", " LN").replace(" ROAD", " RD").replace(" COURT", " CT")
        raw = raw.replace(" BOULEVARD", " BLVD").replace(" APT", "")
        return " ".join(raw.split())

    work = cust.copy()
    work["sur"] = work["full_name"].map(surname)
    work["addr"] = work["address_line"].map(norm_addr)
    work["zip5"] = work["postal_code"].str.slice(0, 5)

    parent = list(range(len(work)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for _, grp in work.groupby("zip5"):
        rows = list(grp.index)
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                i, j = rows[a], rows[b]
                ns = difflib.SequenceMatcher(None, work.at[i, "sur"], work.at[j, "sur"]).ratio()
                as_ = difflib.SequenceMatcher(None, work.at[i, "addr"], work.at[j, "addr"]).ratio()
                if ns >= 0.82 and as_ >= 0.80:
                    union(i, j)

    work["pred_household"] = [find(i) for i in range(len(work))]
    work["pred_household"] = work["pred_household"].map(lambda v: f"P-{v:05d}")
    pairs = work[["record_id", "pred_household"]].merge(truth, on="record_id")
    # pairwise precision/recall against the truth household id
    from itertools import combinations

    def build_pairs(frame: pd.DataFrame, col: str) -> set:
        out = set()
        for _, grp in frame.groupby(col):
            for a, b in combinations(sorted(grp["record_id"]), 2):
                out.add((a, b))
        return out

    pred_pairs = build_pairs(pairs, "pred_household")
    true_pairs = build_pairs(pairs, "household_id")
    tp = len(pred_pairs & true_pairs)
    prec = tp / len(pred_pairs) if pred_pairs else 0.0
    rec = tp / len(true_pairs) if true_pairs else 0.0
    show("fuzzy household merge vs truth", {
        "predicted_households": int(pairs["pred_household"].nunique()),
        "true_households": int(pairs["household_id"].nunique()),
        "pair_precision": round(prec, 3),
        "pair_recall": round(rec, 3),
        "pair_f1": round(2 * prec * rec / (prec + rec), 3),
    })

    # audit trail of removed rows
    removed = df.loc[df.duplicated(keep="first"), ["order_id", "product_id", "quantity", "revenue"]]
    removed = removed.assign(reason="exact duplicate row")
    show("audit trail (first rows removed)", removed.head(4).to_string(index=False))
    removed.to_csv(REPORTS / "ch06_duplicate_audit.csv", index=False)
    show("audit trail written", str((REPORTS / "ch06_duplicate_audit.csv").relative_to(ROOT)))

    # legitimate repeats: same order, different product
    multi = df.groupby("order_id")["product_id"].nunique()
    show("legitimate repeats (same order, several products)", int((multi > 1).sum()))


# ================================================== 6.5 incorrect / impossible
def section_65() -> None:
    df = read_messy()
    price = pd.to_numeric(df["unit_price"].str.replace("$", "", regex=False), errors="coerce")
    qty = pd.to_numeric(df["quantity"], errors="coerce")
    disc = pd.to_numeric(df["discount_pct"], errors="coerce")
    ret = pd.to_numeric(df["returned"], errors="coerce")
    rev = pd.to_numeric(df["revenue"], errors="coerce")
    parsed = pd.to_datetime(df["order_date"], errors="coerce")

    checks = pd.DataFrame(
        [
            ("negative quantity", int((qty < 0).sum())),
            ("zero unit_price", int((price == 0).sum())),
            ("discount above 100", int((disc > 100).sum())),
            ("discount below 0", int((disc < 0).sum())),
            ("returned outside {0,1}", int((~ret.isin([0, 1])).sum())),
            ("order_date in the future", int((parsed > pd.Timestamp("2026-04-02")).sum())),
            ("unknown product_id", int((df["product_id"] == "P-9999").sum())),
            ("unknown store_id", int((df["store_id"] == "S-99").sum())),
        ],
        columns=["check", "failures"],
    )
    show("impossible-value sweep", checks.to_string(index=False))

    # impossible combination: discount > 0 but revenue equals the undiscounted price
    full = (qty * price).round(2)
    impossible = (disc > 0) & (rev.round(2) == full)
    show("impossible combination (discount set, price undiscounted)", int(impossible.sum()))

    # revenue that does not recompute
    implied = (qty * price * (1 - disc.fillna(0) / 100)).round(2)
    mismatch = (rev - implied).abs() > 0.01
    show("revenue does not recompute", {"rows": int(mismatch.sum()),
                                        "share": round(float(mismatch.mean()), 4)})

    # validation vs plausibility
    show("validation vs plausibility", pd.DataFrame(
        [
            ("quantity must be an integer 1-8", "validation", "structural rule, always enforceable"),
            ("unit_price should be above $0.01", "validation", "a zero price is impossible for a sale"),
            ("a $9,000 order is unusual", "plausibility", "possible for a trade buyer, needs a look"),
            ("signup before 2000 is implausible", "plausibility", "likely a sentinel, not a truth"),
        ],
        columns=["statement", "type", "why"],
    ).to_string(index=False))

    # typographic / entry errors: a small corrected pair
    entry = pd.DataFrame(
        [("P-1O01", "P-1001", "letter O mistaken for zero"),
         ("S-1", "S-01", "missing leading zero"),
         ("C-00l234", "C-001234", "letter l mistaken for one")],
        columns=["as_entered", "corrected", "cause"],
    )
    show("typographic errors and their corrections", entry.to_string(index=False))


# ======================================================= 6.6 inconsistent data
def section_66() -> None:
    df = read_messy()
    cat = df["category"].dropna()
    show("category levels before normalization", sorted(cat.unique().tolist()))
    normalized = sorted(cat.str.strip().str.title().unique().tolist())
    show("category levels after trim + title case", normalized)

    region = df["region"].dropna()
    show("region contains non-breaking space", int(region.str.contains("\u00a0").sum()))
    fixed = region.str.replace("\u00a0", " ").str.strip()
    show("region levels after NBSP cleanup", sorted(fixed.unique().tolist()))

    # Unicode normalization
    sample = ["García", "Garcia", "Müller", "Muller", "José", "Jose"]
    forms = pd.DataFrame({"raw": sample})
    forms["nfc_len"] = forms["raw"].map(lambda s: len(unicodedata.normalize("NFC", s)))
    forms["nfd_len"] = forms["raw"].map(lambda s: len(unicodedata.normalize("NFD", s)))
    forms["folded"] = forms["raw"].map(
        lambda s: "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).upper()
    )
    show("Unicode normalization and accent folding", forms.to_string(index=False))
    show("accents fold to the same key",
         {"García vs Garcia": unicodedata.normalize("NFC", "García") == "García",
          "folded equal": (forms.loc[0, "folded"] == forms.loc[1, "folded"])})

    # currency parsing
    currency = df["unit_price"].fillna("").str.startswith("$")
    before = pd.to_numeric(df["unit_price"], errors="coerce")
    after = pd.to_numeric(df["unit_price"].str.replace(r"[$,]", "", regex=True), errors="coerce")
    show("currency-formatted prices", {"rows_with_$": int(currency.sum()),
                                       "nan_before_strip": int(before.isna().sum()),
                                       "nan_after_strip": int(after.isna().sum())})

    # convert a currency column using dated exchange rates and an as-of join
    orders = pd.read_csv(AUDIT / "eu_orders_fx.csv")
    rates = pd.read_csv(AUDIT / "fx_rates.csv")
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    rates["rate_date"] = pd.to_datetime(rates["rate_date"])
    orders = orders.sort_values("order_date")
    rates = rates.sort_values("rate_date")
    merged = pd.merge_asof(orders, rates, left_on="order_date", right_on="rate_date",
                           by="currency", direction="backward")
    merged["usd"] = (merged["amount"] * merged["usd_per_unit"]).round(2)
    show("as-of FX join (head)",
         merged.dropna(subset=["usd_per_unit"])[["order_id", "order_date", "currency", "amount", "rate_date", "usd_per_unit", "usd"]].head(4).to_string(index=False))
    show("as-of FX totals", {"orders": len(merged),
                             "without_a_rate": int(merged["usd_per_unit"].isna().sum()),
                             "usd_total": round(float(merged["usd"].sum()), 2)})

    # mixed date formats and Excel serials
    dates = pd.DataFrame({"mixed": ["2026-01-15", "01/15/2026", "15/01/2026", "45735", "unknown"]})
    parsed_d = pd.to_datetime(dates["mixed"], errors="coerce", format="mixed")
    dates["parsed"] = parsed_d.dt.strftime("%Y-%m-%d")
    serial = (pd.Timestamp("2026-01-15") - pd.Timestamp("1899-12-30")).days
    back = (datetime(1899, 12, 30) + timedelta(days=int(serial))).strftime("%Y-%m-%d")
    show("mixed date formats parsed with format='mixed'", dates.to_string(index=False))
    show("Excel serial round-trip", {"date": "2026-01-15", "serial": int(serial), "back": back})

    # a versioned mapping table for category merges
    mapping = pd.DataFrame(
        [("camping", "Camping", "v1", "2026-01-01"),
         ("footwear", "Footwear", "v1", "2026-01-01"),
         ("apparel", "Apparel", "v1", "2026-01-01"),
         ("accessories", "Accessories", "v1", "2026-01-01"),
         ("outdoor gear", "Camping", "v2", "2026-02-01"),
         ("sportswear", "Apparel", "v2", "2026-02-01")],
        columns=["raw_level", "canonical", "version", "effective_from"],
    )
    show("versioned mapping table", mapping.to_string(index=False))

    key = df["category"].dropna().str.strip().str.lower()
    mapped = key.map(mapping.set_index("raw_level")["canonical"])
    unmapped = sorted(mapped[mapped.isna()].dropna().unique().tolist())
    show("mapping coverage", {"levels": int(key.nunique()),
                              "mapped": int(mapped.notna().sum()),
                              "unmapped_rows": int(mapped.isna().sum()),
                              "unmapped_levels": unmapped})
    mapping.to_csv(REPORTS / "ch06_category_map.csv", index=False)
    show("mapping written", str((REPORTS / "ch06_category_map.csv").relative_to(ROOT)))


# ============================================================== 6.7 outliers
def section_67() -> None:
    df = read_messy()
    rev = pd.to_numeric(df["revenue"], errors="coerce").dropna()

    q1, q3 = [float(v) for v in rev.quantile([0.25, 0.75])]
    iqr = q3 - q1
    hi = q3 + 1.5 * iqr
    iqr_flag = rev > hi

    z = (rev - rev.mean()) / rev.std()
    z_flag = z.abs() > 3

    med = float(rev.median())
    mad = float((rev - med).abs().median())
    mz = 0.6745 * (rev - med) / mad
    mz_flag = mz.abs() > 3.5

    show("IQR fence", {"q1": round(q1, 2), "q3": round(q3, 2), "iqr": round(iqr, 2),
                       "upper_fence": round(hi, 2), "flagged": int(iqr_flag.sum()),
                       "share": round(float(iqr_flag.mean()), 4)})
    show("z-score", {"mean": round(float(rev.mean()), 2), "sd": round(float(rev.std()), 2),
                     "flagged_gt_3": int(z_flag.sum())})
    show("modified z-score (MAD)", {"median": round(med, 2), "mad": round(mad, 2),
                                     "flagged_gt_3.5": int(mz_flag.sum())})
    show("flag overlap", {"iqr_and_z": int((iqr_flag & z_flag).sum()),
                          "iqr_and_mz": int((iqr_flag & mz_flag).sum()),
                          "z_and_mz": int((z_flag & mz_flag).sum())})

    # Mahalanobis on revenue and quantity
    pair = df.assign(
        revenue=pd.to_numeric(df["revenue"], errors="coerce"),
        quantity=pd.to_numeric(df["quantity"], errors="coerce"),
    ).dropna(subset=["revenue", "quantity"])
    X = pair[["revenue", "quantity"]].to_numpy(dtype=float)
    cov = np.cov(X, rowvar=False)
    inv = np.linalg.inv(cov)
    diff = X - X.mean(axis=0)
    md = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
    show("Mahalanobis distance", {"mean": round(float(md.mean()), 3),
                                   "threshold_chi2_0.999": round(float(stats.chi2.ppf(0.999, df=2)), 3),
                                   "flagged": int((md > stats.chi2.ppf(0.999, df=2)).sum())})

    # treatment effects on mean and median
    cap99 = float(rev.quantile(0.99))
    capped = rev.clip(upper=cap99)
    lo1, hi1 = float(rev.quantile(0.01)), float(rev.quantile(0.99))
    winsor = rev.clip(lower=lo1, upper=hi1)
    trimmed = rev[(rev >= lo1) & (rev <= hi1)]
    treat = pd.DataFrame(
        [
            ("keep", len(rev), round(float(rev.mean()), 2), round(float(rev.median()), 2)),
            ("cap at p99", len(capped), round(float(capped.mean()), 2), round(float(capped.median()), 2)),
            ("winsorize 1%/1%", len(winsor), round(float(winsor.mean()), 2), round(float(winsor.median()), 2)),
            ("trim p1-p99", len(trimmed), round(float(trimmed.mean()), 2), round(float(trimmed.median()), 2)),
        ],
        columns=["treatment", "rows", "mean", "median"],
    )
    show("effect of each treatment", treat.to_string(index=False))
    show("p99 cap value", round(cap99, 2))

    # contextual outlier: a Web order shipped express
    show("contextual check", {
        "express_shipping_rows": int((pd.to_numeric(df["shipping_cost"], errors="coerce") >= 40).sum()),
        "note": "49.95 express is rare but valid; investigate, do not delete",
    })


# ====================================================== 6.8 feature engineering
def section_68() -> None:
    df = read_messy()
    clean_qty = pd.to_numeric(df["quantity"], errors="coerce")
    clean_price = pd.to_numeric(df["unit_price"].str.replace("$", "", regex=False), errors="coerce")
    rev = pd.to_numeric(df["revenue"], errors="coerce")
    frame = pd.DataFrame({"revenue": rev}).dropna()

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    scaled = pd.DataFrame({"revenue": frame["revenue"]})
    for name, sc in scalers.items():
        scaled[name] = sc.fit_transform(frame[["revenue"]])
    show("scaling the same column three ways", scaled.head(4).round(3).to_string(index=False))
    show("scaler summaries", {
        "standard_mean": round(float(scaled["standard"].mean()), 6),
        "minmax_range": [round(float(scaled["minmax"].min()), 3), round(float(scaled["minmax"].max()), 3)],
    })

    positive = frame.loc[frame["revenue"] > 0, "revenue"]
    log_rev = np.log(positive)
    bc, lam = stats.boxcox(positive.to_numpy())
    show("log and Box-Cox", {"log_skew_before": round(float(stats.skew(positive)), 3),
                              "log_skew_after": round(float(stats.skew(log_rev)), 3),
                              "boxcox_lambda": round(float(lam), 3)})

    disc = pd.to_numeric(df["discount_pct"], errors="coerce")
    band = pd.cut(disc, bins=[-1, 0, 10, 25, 100], labels=["none", "light", "medium", "deep"])
    quart = pd.qcut(rev.dropna(), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    show("cut (fixed edges) on discount", band.value_counts().sort_index().to_dict())
    show("qcut (equal counts) on revenue", quart.value_counts().sort_index().to_dict())

    features = pd.DataFrame({"revenue": rev, "quantity": clean_qty, "unit_price": clean_price})
    features["line_value"] = features["quantity"] * features["unit_price"]
    features["avg_price"] = (features["revenue"] / features["quantity"]).round(2)
    features["discount_share"] = ((1 - features["revenue"] / features["line_value"]) * 100).round(2)
    show("ratio features (head)", features.head(4).to_string(index=False))

    cat = df["category"].dropna().str.strip().str.title()
    counts = cat.value_counts()
    rare = counts[counts < counts.sum() * 0.05].index.tolist()
    grouped = cat.where(~cat.isin(rare), "Other")
    show("rare-level grouping", {"levels": int(cat.nunique()), "rare_levels": rare,
                                  "after_grouping": int(grouped.nunique())})
    show("one-hot preview", pd.get_dummies(grouped.head(3), prefix="cat").to_string(index=False))

    parsed = pd.to_datetime(df["order_date"], errors="coerce", format="mixed")
    date_feats = pd.DataFrame({
        "order_date": parsed,
        "year": parsed.dt.year,
        "month": parsed.dt.month,
        "dow": parsed.dt.dayofweek,
        "is_weekend": parsed.dt.dayofweek.isin([5, 6]),
        "days_since_start": (parsed - pd.Timestamp("2024-01-01")).dt.days,
    }).dropna()
    for col in ["year", "month", "dow", "days_since_start"]:
        date_feats[col] = date_feats[col].astype(int)
    show("date-derived features (head)", date_feats.head(4).to_string(index=False))

    text = df["order_id"].astype(str)
    show("text features on order_id", {"length": int(text.str.len().iloc[0]),
                                       "digit_count": int(text.str.count(r"\d").iloc[0]),
                                       "has_hyphen": bool(text.str.contains("-").iloc[0])},
         )


# ====================================================== 6.9 reusable pipeline
CONFIG = {
    "category_map": {"camping": "Camping", "footwear": "Footwear",
                     "apparel": "Apparel", "accessories": "Accessories"},
    "qty_min": 1,
    "qty_max": 8,
    "discount_min": 0.0,
    "discount_max": 100.0,
    "as_of": "2026-04-02",
}


class CleaningError(Exception):
    """Raised when a control check fails and the pipeline must stop."""


def clean_orders(df: pd.DataFrame, config: dict, log: list,
                 exc_path: Path = REPORTS / "ch06_exceptions.csv") -> pd.DataFrame:
    """Apply the chapter's rules; return a cleaned frame and append to `log`."""
    out = df.copy()
    rows_in = len(out)

    # 1. exact duplicates
    out = out.drop_duplicates().reset_index(drop=True)
    log.append(("drop_exact_duplicates", rows_in - len(out)))

    # 2. normalize text columns
    for col in ["channel", "region", "category"]:
        out[col] = (out[col].astype("string")
                    .str.replace("\u00a0", " ", regex=False)
                    .str.strip())
    out["category"] = out["category"].str.title()
    out["region"] = out["region"].str.title()
    out["channel"] = out["channel"].str.title()

    # 3. map categories through the versioned table
    unmapped = out.loc[~out["category"].isin(config["category_map"].values()) & out["category"].notna(), "category"]
    log.append(("unmapped_categories", int(unmapped.nunique())))

    # 4. parse mixed date formats; flag the rest
    parsed = pd.to_datetime(out["order_date"], errors="coerce", format="mixed")
    bad_dates = parsed.isna()
    out["order_date"] = parsed
    log.append(("unparseable_dates", int(bad_dates.sum())))

    # 5. coerce numbers
    out["unit_price"] = pd.to_numeric(out["unit_price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    for col in ["quantity", "discount_pct", "shipping_cost", "returned", "revenue"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # 6. replace sentinels that survived numeric coercion
    out["quantity"] = out["quantity"].replace(9999, np.nan)
    out["discount_pct"] = out["discount_pct"].replace(9999, np.nan)
    out["customer_id"] = out["customer_id"].replace("-1", pd.NA)

    # 7. flag impossible values, do not silently drop
    exceptions = []
    for idx, r in out.iterrows():
        if pd.notna(r["quantity"]) and (r["quantity"] < config["qty_min"] or r["quantity"] > config["qty_max"]):
            exceptions.append((idx, "quantity", "out_of_range", r["quantity"]))
        if pd.notna(r["discount_pct"]) and not (config["discount_min"] <= r["discount_pct"] <= config["discount_max"]):
            exceptions.append((idx, "discount_pct", "out_of_range", r["discount_pct"]))
        if pd.notna(r["order_date"]) and r["order_date"] > pd.Timestamp(config["as_of"]):
            exceptions.append((idx, "order_date", "future", str(r["order_date"])[:10]))
        if pd.notna(r["unit_price"]) and r["unit_price"] <= 0:
            exceptions.append((idx, "unit_price", "non_positive", r["unit_price"]))
    exc = pd.DataFrame(exceptions, columns=["row", "column", "issue", "value"])
    log.append(("exceptions_logged", len(exc)))
    exc.to_csv(exc_path, index=False)

    # 8. rebuild revenue from the formula (do not trust the stored value)
    out["revenue_calc"] = (out["quantity"] * out["unit_price"] * (1 - out["discount_pct"].fillna(0) / 100)).round(2)
    mismatch = (out["revenue"] - out["revenue_calc"]).abs() > 0.01
    log.append(("revenue_recomputed_mismatch", int(mismatch.sum())))
    out["revenue"] = out["revenue_calc"]
    out = out.drop(columns="revenue_calc")
    return out


def section_69() -> None:
    df = read_messy()
    log: list = []
    cleaned = clean_orders(df, CONFIG, log)
    show("cleaning log", pd.DataFrame(log, columns=["step", "count"]).to_string(index=False))

    # control totals
    def control_total(frame: pd.DataFrame) -> dict:
        return {"rows": len(frame), "orders": int(frame["order_id"].nunique()),
                "gross": round(float(frame["revenue"].sum()), 2)}

    show("control totals before / after", {"before": control_total(df.assign(revenue=pd.to_numeric(df["revenue"], errors="coerce"))),
                                           "after": control_total(cleaned)})

    # pure-function unit tests on 20 hand-built rows
    hand_rows = [
        ("O-900000", "2026-01-04", "C-000101", "Web", "Midwest", "camping", "P-1001",
         "2", "58.53", "0", "3.95", "0", "117.06", "S-01"),
        ("O-900001", "01/05/2026", "C-000102", "app", "South\u00a0", " Camping ", "P-1001",
         "1", "$129.00", "10", "0", "0", "116.10", "S-02"),
        ("O-900002", "2026-02-30", "-1", "Phone", "West", "FOOTWEAR", "P-1001",
         "9999", "0.00", "150.0", "0", "0", "0.00", "S-03"),
        ("O-900003", "unknown", "C-000104", "Web", "", "Apparel", "P-1001",
         "-2", "39.95", "0", "7.95", "0", "79.90", "S-04"),
        ("O-900004", "2026-01-08", "C-000105", "WEB", "Northeast", "Apparel", "P-1001",
         "3", "44.63", "0", "0", "1", "133.89", "S-05"),
        ("O-900005", "2030-05-01", "C-000106", "App", "South", "accessories", "P-1001",
         "1", "24.50", "0", "0", "0", "24.50", "S-06"),
    ]
    for i in range(6, 20):
        hand_rows.append((f"O-{900000 + i}", "2026-01-10", f"C-{100 + i:06d}", "Web",
                          "Midwest", "Camping", "P-1002", "1", "50.00", "0", "3.95",
                          "0", "50.00", "S-01"))
    hand = pd.DataFrame(hand_rows, columns=[
        "order_id", "order_date", "customer_id", "channel", "region", "category",
        "product_id", "quantity", "unit_price", "discount_pct", "shipping_cost",
        "returned", "revenue", "store_id"])
    tests: list = []
    hand_log: list = []
    hand_clean = clean_orders(hand, CONFIG, hand_log,
                              exc_path=REPORTS / "ch06_hand_exceptions.csv")

    def check(name: str, cond: bool) -> None:
        tests.append({"test": name, "result": "PASS" if cond else "FAIL"})

    mask = hand_clean[["quantity", "unit_price", "discount_pct"]].notna().all(axis=1)
    recomputed = (
        hand_clean.loc[mask, "quantity"] * hand_clean.loc[mask, "unit_price"]
        * (1 - hand_clean.loc[mask, "discount_pct"].fillna(0) / 100)
    ).round(2)
    check("duplicate policy is idempotent", len(hand_clean.drop_duplicates()) == len(hand_clean))
    check("category is canonical", set(hand_clean["category"].dropna()) <= set(CONFIG["category_map"].values()))
    check("NBSP removed from region", not hand_clean["region"].astype("string").str.contains("\u00a0", na=False).any())
    check("invalid date becomes NaT", bool(hand_clean["order_date"].isna().any()))
    check("currency parsed to number", hand_clean.loc[1, "unit_price"] == 129.00)
    check("sentinel quantity is null", bool(pd.isna(hand_clean.loc[2, "quantity"])))
    check("revenue recomputes", bool((hand_clean.loc[mask, "revenue"].round(2) == recomputed).all()))
    show("unit tests on 20 hand-built rows", pd.DataFrame(tests).to_string(index=False))
    show("hand-built rows cleaned (first 8)",
         hand_clean[["order_id", "order_date", "category", "region", "quantity", "unit_price", "revenue"]].head(8).to_string(index=False))
    show("hand-built exceptions", pd.read_csv(REPORTS / "ch06_hand_exceptions.csv").to_string(index=False))

    # exceptions file shape (the 40,000-row run)
    exc = pd.read_csv(REPORTS / "ch06_exceptions.csv")
    show("exceptions file", {"rows": len(exc), "issues": exc["issue"].value_counts().to_dict()})


# ========================================================= 6.10 example bank
def section_610() -> None:
    df = read_messy()

    # small helpers
    price = pd.to_numeric(df["unit_price"].str.replace("$", "", regex=False), errors="coerce")
    qty = pd.to_numeric(df["quantity"], errors="coerce")
    disc = pd.to_numeric(df["discount_pct"], errors="coerce")
    rev = pd.to_numeric(df["revenue"], errors="coerce")

    show("E1 blank vs sentinel vs lookalike", {
        "truly_blank_region": int(df["region"].isna().sum()),
        "region_unknown_token": int((df["region"] == "unknown").sum()),
        "region_na_token": int((df["region"] == "N/A").sum()),
    })
    show("E2 missingness ranking (top)", (df.isna().mean().sort_values(ascending=False).head(4) * 100).round(3).to_dict())
    show("E3 rows with more than one missing field", int((df.isna().sum(axis=1) > 1).sum()))
    show("E4 sentinel sweep", {c: int(df[c].astype(str).isin(["-1", "9999", "unknown", "N/A", "1900-01-01"]).sum())
                                for c in df.columns if df[c].astype(str).isin(["-1", "9999", "unknown", "N/A", "1900-01-01"]).any()})
    show("E5 exact vs key duplicates", {"exact": int(df.duplicated().sum()),
                                        "on (order_id, product_id)": int(df.duplicated(subset=["order_id", "product_id"], keep=False).sum())})
    show("E6 negative and zero sweep", {"negative_quantity": int((qty < 0).sum()),
                                        "zero_price": int((price == 0).sum()),
                                        "discount_gt_100": int((disc > 100).sum())})
    parsed = pd.to_datetime(df["order_date"], errors="coerce", format="mixed")
    show("E7 unparseable and future dates", {"unparseable": int(parsed.isna().sum()),
                                             "future": int((parsed > pd.Timestamp("2026-04-02")).sum())})
    show("E8 category normalization", {"raw_levels": sorted(df["category"].dropna().unique().tolist()),
                                        "normalized": sorted(df["category"].dropna().str.strip().str.title().unique().tolist())})
    show("E9 whitespace and encoding flags", {"padded_category": int(df["category"].fillna("").str.startswith(" ").sum()),
                                              "nbsp_region": int(df["region"].fillna("").str.contains("\u00a0").sum())})
    show("E10 IQR fence on revenue", {"upper_fence": round(float(rev.quantile(0.75) + 1.5 * (rev.quantile(0.75) - rev.quantile(0.25))), 2),
                                      "flagged": int((rev > rev.quantile(0.75) + 1.5 * (rev.quantile(0.75) - rev.quantile(0.25))).sum())})
    show("E11 cap effect", {"mean_uncapped": round(float(rev.mean()), 2),
                            "mean_capped_p99": round(float(rev.clip(upper=rev.quantile(0.99)).mean()), 2)})
    show("E12 control total", {"rows": len(df), "orders": int(df["order_id"].nunique()),
                               "gross": round(float(rev.sum()), 2)})
    show("E13 broken foreign keys", {"P-9999": int((df["product_id"] == "P-9999").sum()),
                                     "S-99": int((df["store_id"] == "S-99").sum())})
    show("E14 currency-format parse", {"nan_plain": int(pd.to_numeric(df["unit_price"], errors="coerce").isna().sum()),
                                       "nan_after_strip": int(price.isna().sum())})
    variants: dict = {}
    for value in df["category"].dropna().astype(str):
        variants.setdefault(value.strip().title(), set()).add(value)
    show("E15 raw variants per canonical category",
         {k: sorted(v) for k, v in variants.items()})
    show("E16 column-name hygiene", {"unnamed": int(sum(str(c).startswith("Unnamed") for c in df.columns)),
                                     "with_spaces": int(sum(" " in str(c) for c in df.columns))})


# ============================================================= 6.11 capstone
def section_611() -> None:
    df = read_messy()
    log: list = []
    cleaned = clean_orders(df, CONFIG, log)
    exc = pd.read_csv(REPORTS / "ch06_exceptions.csv")

    before_gross = round(float(pd.to_numeric(df["revenue"], errors="coerce").sum()), 2)
    after_gross = round(float(cleaned["revenue"].sum()), 2)
    show("capstone: rows and gross", {"rows_before": len(df), "rows_after": len(cleaned),
                                      "gross_before": before_gross, "gross_after": after_gross,
                                      "difference": round(before_gross - after_gross, 2)})

    defect_log = pd.DataFrame(log, columns=["defect", "count"])
    show("capstone: defect log", defect_log.to_string(index=False))

    exc_summary = exc.groupby("issue").size().to_dict()
    show("capstone: exceptions by issue", exc_summary)

    # completeness after cleaning
    completeness = (1 - cleaned.isna().mean().mean()) * 100
    show("capstone: completeness", {"cell_completeness_pct": round(float(completeness), 3),
                                    "columns_with_missing": int(cleaned.isna().any().sum())})

    clean_path = ROOT / "data" / "processed" / "orders_clean_ch06.csv"
    cleaned.to_csv(clean_path, index=False)
    exc.to_csv(ROOT / "data" / "processed" / "orders_exceptions_ch06.csv", index=False)
    show("capstone: artifacts written", {
        "clean": str(clean_path.relative_to(ROOT)),
        "exceptions": "data/processed/orders_exceptions_ch06.csv",
        "defect_log_rows": len(defect_log),
    })

    # a one-line verdict
    show("capstone: verdict", {
        "usable": bool(completeness > 95),
        "blocking": [r for r in exc_summary],
        "note": "revenue recomputed from the formula, so the stored-value defect cannot leak",
    })


# ============================================================ extra material
def section_extra() -> None:
    """Additional worked blocks used in the expanded chapter sections."""
    df = read_messy()
    ic0 = pd.read_csv(AUDIT / "income_churn.csv")
    qty0 = pd.to_numeric(df["quantity"], errors="coerce")
    price0 = pd.to_numeric(df["unit_price"].str.replace("$", "", regex=False), errors="coerce")
    disc0 = pd.to_numeric(df["discount_pct"], errors="coerce")
    rev0 = pd.to_numeric(df["revenue"], errors="coerce")
    parsed0 = pd.to_datetime(df["order_date"], errors="coerce", format="mixed")

    # X1 bias direction under MAR with a strong composition effect
    rng = np.random.default_rng(11)
    n = 4000
    seg = rng.choice(["Basic", "Plus"], size=n, p=[0.5, 0.5])
    inc = np.where(seg == "Basic", rng.normal(45000, 9000, n), rng.normal(90000, 14000, n))
    ch = rng.binomial(1, np.where(seg == "Basic", 0.70, 0.40))
    miss = rng.random(n) < np.where(seg == "Basic", 0.30, 0.05)
    sim = pd.DataFrame({"segment": seg, "income": inc, "churn": ch, "missing": miss})
    show("X1 bias direction under MAR", {
        "population_churn": round(float(sim["churn"].mean()), 4),
        "complete_case_churn": round(float(sim.loc[~sim["missing"], "churn"].mean()), 4),
        "bias": round(float(sim.loc[~sim["missing"], "churn"].mean() - sim["churn"].mean()), 4),
    })

    # X2 fake-null catalog
    tokenset = {"", "N/A", "n/a", "-1", "unknown", "Unknown", "9999", "not available"}
    rows = []
    for col in df.columns:
        vals = df[col].dropna().astype(str)
        found = sorted(set(vals) & tokenset)
        if found:
            rows.append((col, ", ".join(found), int(vals.isin(found).sum())))
    show("X2 fake-null catalog", pd.DataFrame(rows, columns=["column", "tokens", "cells"]).to_string(index=False))

    # X3 missingness by quarter, as a compact text heatmap
    bucket = parsed0.dt.to_period("Q").astype(str)
    grid = pd.DataFrame({"bucket": bucket, "region": df["region"].isna(),
                         "discount": df["discount_pct"].isna(),
                         "customer": df["customer_id"].isna()}).dropna()
    show("X3 missingness by quarter", grid.groupby("bucket").mean().round(4).head(6).to_string())

    # X4 categorical fill: mode vs an honest Unknown bucket
    conv_reg = (df["region"].replace({"unknown": np.nan, "N/A": np.nan})
                .str.replace("\u00a0", " ", regex=False).str.strip().str.title())
    mode = conv_reg.mode(dropna=True).iloc[0]
    before = conv_reg.value_counts(dropna=False)
    after = conv_reg.fillna("Unknown").value_counts()
    show("X4 mode and Unknown bucket", pd.DataFrame({"before": before, "after": after}).fillna(0).astype(int).to_string())

    # X5 variance shrinkage under each fill
    inc = ic0["income"]
    fills = [
        ("complete case", inc.dropna()),
        ("mean", inc.fillna(inc.mean())),
        ("median", inc.fillna(inc.median())),
        ("constant 0", inc.fillna(0)),
    ]
    show("X5 mean and sd under each fill", pd.DataFrame(
        [(name, round(float(s.mean()), 2), round(float(s.std()), 2)) for name, s in fills],
        columns=["method", "mean", "sd"]).to_string(index=False))

    # X6 small-group fallback
    small = ic0[ic0["segment"] == "Premium"].head(3).copy()
    small["segment"] = "Other"
    small["income"] = np.nan
    combined = pd.concat([ic0, small], ignore_index=True)
    med = combined.groupby("segment")["income"].transform("median")
    filled = combined["income"].fillna(med).fillna(combined["income"].median())
    show("X6 minimum-group-size fallback", {
        "small_group_median": float(med[combined["segment"] == "Other"].iloc[0]),
        "fallback_value": round(float(filled[combined["segment"] == "Other"].iloc[0]), 2),
        "global_median": float(combined["income"].median()),
    })

    # X7 scikit-learn SimpleImputer
    imp = __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="median")
    arr = imp.fit_transform(ic0[["income"]])
    show("X7 SimpleImputer", {"statistics": [round(float(v), 2) for v in imp.statistics_],
                              "rows": len(arr), "nan_after": int(np.isnan(arr).sum())})

    # X8 conflicting-key amounts
    key = ["order_id", "product_id"]
    dup = df[df.duplicated(subset=key, keep=False)].copy()
    dup["rn"] = pd.to_numeric(dup["revenue"], errors="coerce")
    spread = dup.groupby(key)["rn"].agg(lambda s: float(s.max() - s.min()))
    show("X8 conflicting-key amounts", {
        "conflicting_groups": int((spread > 0.01).sum()),
        "max_spread": round(float(spread.max()), 2),
        "sum_of_spreads": round(float(spread.sum()), 2),
    })

    # X9 blocking quality
    cust = read_customers()
    cust["zip5"] = cust["postal_code"].str.slice(0, 5)
    counts = cust.groupby("zip5").size()
    show("X9 blocking quality", {
        "blocks": int(len(counts)), "largest_block": int(counts.max()),
        "candidate_pairs": int(sum(int(v) * (int(v) - 1) // 2 for v in counts)),
        "all_pairs": int(len(cust) * (len(cust) - 1) // 2),
    })

    # X10 completeness score inside a conflict group
    example = dup.groupby(key).size().sort_values(ascending=False).index[0]
    top = dup[(dup["order_id"] == example[0]) & (dup["product_id"] == example[1])].copy()
    top["non_null"] = top.notna().sum(axis=1)
    show("X10 most-complete row in a conflict", top[["order_id", "product_id", "region", "discount_pct", "quantity", "revenue", "non_null"]].head(5).to_string(index=False))

    # X11 named rule runner
    norm_cat = df["category"].fillna("").str.strip().str.title()
    norm_reg = df["region"].replace({"unknown": np.nan, "N/A": np.nan}).str.replace("\u00a0", " ", regex=False).str.strip().str.title()
    rules = {
        "order_id format": df["order_id"].str.fullmatch(r"O-\d{6}").eq(True),
        "customer_id format or null": df["customer_id"].isna() | df["customer_id"].str.fullmatch(r"C-\d{6}").eq(True),
        "quantity 1-8": qty0.between(1, 8),
        "unit_price > 0": price0 > 0,
        "discount 0-100": disc0.between(0, 100),
        "returned in {0,1}": pd.to_numeric(df["returned"], errors="coerce").isin([0, 1]),
        "date parses": parsed0.notna(),
        "date not future": parsed0 <= pd.Timestamp("2026-04-02"),
        "category in domain": norm_cat.isin(["Camping", "Footwear", "Apparel", "Accessories"]),
        "region in domain or null": norm_reg.isna() | norm_reg.isin(["Midwest", "Northeast", "South", "West"]),
    }
    res = pd.DataFrame([{"rule": k, "checks": int(len(v)), "failures": int((~v).sum())} for k, v in rules.items()])
    res["status"] = np.where(res["failures"] == 0, "PASS", "FAIL")
    show("X11 named rule runner", res.to_string(index=False))

    # X12 plausibility review queue
    review = df.loc[(qty0 > 8) | (rev0 > rev0.quantile(0.999)),
                    ["order_id", "quantity", "unit_price", "discount_pct", "revenue"]]
    show("X12 plausibility review queue", review.head(6).to_string(index=False))

    # X13 case-fold collisions across text columns
    collisions = {}
    for col in ["channel", "region", "category"]:
        groups = {}
        for value in df[col].dropna().astype(str):
            groups.setdefault(value.strip().casefold(), set()).add(value)
        collisions[col] = sum(1 for v in groups.values() if len(v) > 1)
    show("X13 case-fold collisions", collisions)

    # X14 mapping table with a new level
    mapping = {"camping": "Camping", "footwear": "Footwear", "apparel": "Apparel",
               "accessories": "Accessories", "outdoor gear": "Camping", "sportswear": "Apparel"}
    new_levels = pd.Series(["Camping", "camping", "Outdoor Gear", "Sportswear", "Mystery"])
    show("X14 mapping a new source level", pd.DataFrame({
        "raw": new_levels,
        "canonical": new_levels.str.strip().str.lower().map(mapping),
    }).to_string(index=False))

    # X15 double-conversion control
    orders = pd.read_csv(AUDIT / "eu_orders_fx.csv")
    rates = pd.read_csv(AUDIT / "fx_rates.csv")
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    rates["rate_date"] = pd.to_datetime(rates["rate_date"])
    m = pd.merge_asof(orders.sort_values("order_date"), rates.sort_values("rate_date"),
                      left_on="order_date", right_on="rate_date", by="currency",
                      direction="backward").dropna(subset=["usd_per_unit"])
    m["usd"] = (m["amount"] * m["usd_per_unit"]).round(2)
    second = (m["usd"] * m["usd_per_unit"]).round(2)
    show("X15 double-conversion control", {
        "first_conversion_valid": bool(np.allclose(m["usd"], (m["amount"] * m["usd_per_unit"]).round(2))),
        "second_conversion_error": round(float((second - m["usd"]).abs().sum()), 2),
    })

    # X16 top extreme orders
    X = df.assign(revenue=rev0, quantity=qty0).dropna(subset=["revenue", "quantity"])[["revenue", "quantity"]].to_numpy(float)
    cov = np.cov(X, rowvar=False)
    diff = X - X.mean(axis=0)
    md = np.sqrt(np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff))
    pos = df.assign(revenue=rev0, quantity=qty0).dropna(subset=["revenue", "quantity"]).index
    top = pd.DataFrame({"row": pos, "revenue": X[:, 0].round(2), "mahalanobis": md.round(3)})
    show("X16 ten most extreme orders", top.sort_values("mahalanobis", ascending=False).head(10).to_string(index=False))

    # X17 threshold sensitivity
    q1v, q3v = [float(v) for v in rev0.dropna().quantile([0.25, 0.75])]
    iqrv = q3v - q1v
    med = float(rev0.dropna().median())
    madv = float((rev0.dropna() - med).abs().median())
    mzv = 0.6745 * (rev0.dropna() - med) / madv
    rows = []
    for k in [1.0, 1.5, 2.0, 3.0]:
        f = rev0 > q3v + k * iqrv
        rows.append((f"iqr k={k}", int(f.sum()), round(float(f.mean()), 4)))
    for t in [2.5, 3.0, 3.5, 4.0]:
        f = mzv.abs() > t
        rows.append((f"mz>{t}", int(f.sum()), round(float(f.mean()), 4)))
    show("X17 threshold sensitivity", pd.DataFrame(rows, columns=["rule", "flagged", "share"]).to_string(index=False))

    # X18 transform effects
    s = rev0.dropna()
    s = s[s > 0]
    bc, lam = stats.boxcox(s.to_numpy())
    show("X18 transform effects on shape", pd.DataFrame([
        ("raw", stats.skew(s), stats.kurtosis(s)),
        ("log", stats.skew(np.log(s)), stats.kurtosis(np.log(s))),
        ("sqrt", stats.skew(np.sqrt(s)), stats.kurtosis(np.sqrt(s))),
        ("boxcox", stats.skew(bc), stats.kurtosis(bc)),
    ], columns=["transform", "skew", "kurtosis"]).round(3).to_string(index=False))

    # X19 a wider feature frame and its profile
    feat = pd.DataFrame({"revenue": rev0, "quantity": qty0,
                         "unit_price": price0, "discount": disc0}).dropna()
    feat["line_value"] = feat["quantity"] * feat["unit_price"]
    feat["avg_price"] = (feat["revenue"] / feat["quantity"]).round(2)
    feat["discount_share"] = ((1 - feat["revenue"] / feat["line_value"]) * 100).round(2)
    feat["log_revenue"] = np.log1p(feat["revenue"]).round(3)
    feat["high_value"] = (feat["revenue"] > feat["revenue"].median()).astype(int)
    show("X19 feature frame (head)", feat.head(5).round(2).to_string(index=False))
    show("X19 feature frame profile", feat.describe().round(2).T.to_string())

    # X20 rare-level threshold sensitivity
    cat = df["category"].dropna().str.strip().str.title()
    counts = cat.value_counts()
    show("X20 rare-level threshold sensitivity", pd.DataFrame([
        (t, int((counts < counts.sum() * t).sum()),
         int(cat.isin(counts[counts < counts.sum() * t].index).sum()))
        for t in [0.05, 0.10, 0.20, 0.30]
    ], columns=["threshold", "rare_levels", "rows_grouped"]).to_string(index=False))

    # X21 quarter / week / rolling features
    dd = pd.DataFrame({"date": parsed0, "revenue": rev0}).dropna().sort_values("date").reset_index(drop=True)
    dd["quarter"] = dd["date"].dt.to_period("Q").astype(str)
    dd["week"] = dd["date"].dt.isocalendar().week.astype(int)
    dd["rolling7"] = dd["revenue"].rolling(7).mean().round(2)
    show("X21 quarter, week and rolling features", dd.head(4).to_string(index=False))

    # X23 log schema
    show("X23 cleaning log schema", pd.DataFrame([
        ("step", "string", "name of the rule that produced the count"),
        ("count", "integer", "rows or cells affected; zero proves the rule ran"),
    ], columns=["column", "type", "meaning"]).to_string(index=False))

    # X24 blocking control checks
    cleaned = clean_orders(df, CONFIG, [])
    try:
        if len(cleaned) > len(df):
            raise CleaningError("row count grew")
        if cleaned["order_id"].nunique() != df["order_id"].nunique():
            raise CleaningError("distinct order count changed")
        print("control checks passed: no blocking failure")
    except CleaningError as exc:
        print("blocked:", exc)

    # E17-E28 example-bank sweeps
    show("E17 constant and high-cardinality columns", {
        "constant": int((df.nunique(dropna=True) <= 1).sum()),
        "high_cardinality_gt_1000": int((df.nunique(dropna=True) > 1000).sum()),
    })
    show("E18 index sanity", {"is_range_index": isinstance(df.index, pd.RangeIndex),
                              "is_monotonic": bool(df.index.is_monotonic_increasing),
                              "is_unique": bool(df.index.is_unique)})
    days = parsed0.dropna().dt.normalize().drop_duplicates().sort_values()
    show("E19 date span and gaps", {"span_days": int((days.max() - days.min()).days),
                                    "distinct_days": int(days.nunique()),
                                    "largest_gap_days": int(days.diff().dt.days.max())})
    show("E20 duplicate key spread", {"groups": int(dup.groupby(key).ngroups),
                                      "conflicting": int((spread > 0.01).sum())})
    show("E21 discount missing by channel",
         df.assign(m=df["discount_pct"].isna()).groupby("channel")["m"].mean().round(4).to_dict())
    show("E22 sentinel taxonomy", {
        "dash_one": int((df == "-1").sum().sum()),
        "nines": int((df == "9999").sum().sum()),
        "unknown_tokens": int(df.isin(["unknown", "N/A", "n/a"]).sum().sum()),
    })
    masks = df[["region", "discount_pct", "customer_id"]].replace(
        {"region": {"unknown": np.nan, "N/A": np.nan},
         "discount_pct": {"9999": np.nan},
         "customer_id": {"-1": np.nan}}).isna()
    pairs = {}
    for a in masks.columns:
        for b in masks.columns:
            if a < b:
                pairs[f"{a}+{b}"] = int((masks[a] & masks[b]).sum())
    show("E23 co-occurrence top pairs", {k: v for k, v in sorted(pairs.items(), key=lambda x: -x[1])[:3]})
    show("E24 skew and kurtosis", {"skew": round(float(stats.skew(rev0.dropna())), 3),
                                   "kurtosis": round(float(stats.kurtosis(rev0.dropna())), 3)})
    show("E25 IQR flags by channel", rev0.groupby(df["channel"]).apply(
        lambda s: int((s > s.quantile(0.75) + 1.5 * (s.quantile(0.75) - s.quantile(0.25))).sum()),
        include_groups=False).to_dict())
    show("E26 median-fill effect on gross", {
        "gross_complete": round(float(rev0.sum()), 2),
        "gross_after_recompute": round(float(cleaned["revenue"].sum()), 2),
    })
    line_value = qty0 * price0
    show("E27 zero denominators", int((line_value == 0).sum()))
    show("E28 column-name casing", {"all_lower": int(sum(str(c) == str(c).lower() for c in df.columns)),
                                    "total": int(len(df.columns))})

    # capstone defect impact
    work = df.assign(rn=rev0)
    impact = pd.DataFrame([
        ("exact duplicate rows", int(df.duplicated().sum()),
         round(float(work.loc[df.duplicated(), "rn"].sum()), 2)),
        ("blank region", int(df["region"].isna().sum()),
         round(float(work.loc[df["region"].isna(), "rn"].sum()), 2)),
        ("unparseable dates", int(parsed0.isna().sum()),
         round(float(work.loc[parsed0.isna(), "rn"].sum()), 2)),
        ("zero prices", int((price0 == 0).sum()),
         round(float(work.loc[price0 == 0, "rn"].sum()), 2)),
    ], columns=["issue", "rows", "revenue_at_stake"])
    show("X37 capstone defect impact", impact.to_string(index=False))


def section_extra2() -> None:
    """Final worked blocks: clean-file profile, revenue bridge, more sweeps."""
    df = read_messy()
    qty = pd.to_numeric(df["quantity"], errors="coerce")
    price = pd.to_numeric(df["unit_price"].str.replace("$", "", regex=False), errors="coerce")
    disc = pd.to_numeric(df["discount_pct"], errors="coerce")
    rev = pd.to_numeric(df["revenue"], errors="coerce")
    parsed = pd.to_datetime(df["order_date"], errors="coerce", format="mixed")
    cleaned = clean_orders(df, CONFIG, [])

    # Y1 sensitivity across outcomes
    complete = ic0_complete = pd.read_csv(AUDIT / "income_churn.csv")
    inc = complete["income"]
    med_fill = complete.assign(income=inc.fillna(inc.median()))
    grp = complete.copy()
    grp["income"] = grp.groupby("segment")["income"].transform(lambda s: s.fillna(s.median()))
    def gap(frame):
        m = frame["income"].median()
        return round(float(frame.loc[frame["income"] < m, "churn"].mean()
                           - frame.loc[frame["income"] >= m, "churn"].mean()), 3)
    sens = pd.DataFrame([
        ("complete case", len(complete.dropna()), round(float(inc.mean()), 2),
         gap(complete.dropna())),
        ("median", len(med_fill), round(float(med_fill["income"].mean()), 2),
         gap(med_fill)),
        ("group median", len(grp), round(float(grp["income"].mean()), 2), gap(grp)),
    ], columns=["method", "rows", "mean_income", "churn_gap"])
    show("Y1 sensitivity across outcomes", sens.to_string(index=False))

    # Y2 treatment effects by channel
    rows = []
    for ch, g in df.groupby("channel"):
        r = pd.to_numeric(g["revenue"], errors="coerce").dropna()
        rows.append((ch, len(r), round(float(r.mean()), 2),
                     round(float(r.clip(upper=r.quantile(0.99)).mean()), 2),
                     round(float(r.median()), 2)))
    show("Y2 treatment effects by channel", pd.DataFrame(
        rows, columns=["channel", "rows", "mean", "mean_capped", "median"]
    ).to_string(index=False))

    # Y3 clean-file profile
    prof = pd.DataFrame([{
        "column": c, "dtype": str(cleaned[c].dtype),
        "non_null": int(cleaned[c].notna().sum()),
        "missing": int(cleaned[c].isna().sum()),
        "unique": int(cleaned[c].nunique(dropna=True)),
        "example": str(cleaned[c].dropna().iloc[0])[:18]
        if cleaned[c].notna().any() else "",
    } for c in cleaned.columns])
    show("Y3 clean-file profile", prof.to_string(index=False))

    # Y4 post-clean integrity
    show("Y4 post-clean integrity", {
        "rows": len(cleaned),
        "exact_duplicates": int(cleaned.duplicated().sum()),
        "total_missing_cells": int(cleaned.isna().sum().sum()),
        "columns_with_missing": int(cleaned.isna().any().sum()),
        "orders": int(cleaned["order_id"].nunique()),
    })

    # Y5 revenue bridge
    before = round(float(rev.sum()), 2)
    dup_mask = df.duplicated()
    dup_rev = round(float(rev[dup_mask].sum()), 2)
    after_no_dupes = round(float(rev[~dup_mask].sum()), 2)
    after = round(float(cleaned["revenue"].sum()), 2)
    show("Y5 revenue bridge", pd.DataFrame([
        ("gross as stored (40,000 rows)", before),
        ("less stored revenue on duplicate rows", -dup_rev),
        ("= stored gross after dedupe", after_no_dupes),
        ("effect of rebuilding revenue from components", round(after - after_no_dupes, 2)),
        ("= gross in clean file", after),
    ], columns=["line", "amount"]).to_string(index=False))

    # Y6 headline before / after
    show("Y6 headline metrics", {
        "mean_before": round(float(rev.mean()), 2),
        "mean_after": round(float(cleaned["revenue"].mean()), 2),
        "median_before": round(float(rev.median()), 2),
        "median_after": round(float(cleaned["revenue"].median()), 2),
    })

    # E29-E40: a third dozen of short sweeps
    show("E29 lines per order", df.groupby("order_id").size().value_counts().sort_index().to_dict())
    show("E30 duplicate rate by channel", df.assign(d=df.duplicated()).groupby("channel")["d"].mean().round(4).to_dict())
    show("E31 price quantiles", {str(q): round(float(price.quantile(q)), 2)
                                 for q in [0.01, 0.25, 0.5, 0.75, 0.99]})
    show("E32 revenue quantiles", {str(q): round(float(rev.quantile(q)), 2)
                                   for q in [0.01, 0.25, 0.5, 0.75, 0.99]})
    show("E33 discount levels", disc.round(-1).value_counts().sort_index().head(6).to_dict())
    show("E34 return rate by channel", pd.to_numeric(df["returned"], errors="coerce").groupby(df["channel"]).mean().round(4).to_dict())
    show("E35 shipping levels", pd.to_numeric(df["shipping_cost"], errors="coerce").value_counts().sort_index().head(6).to_dict())
    show("E36 key lengths", {c: int(df[c].dropna().str.len().mode().iloc[0])
                             for c in ["order_id", "customer_id", "product_id", "store_id"]})
    show("E37 date token shapes", {
        "iso": int(df["order_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").sum()),
        "us_style": int(df["order_date"].str.fullmatch(r"\d{2}/\d{2}/\d{4}").sum()),
        "junk": int((~df["order_date"].str.fullmatch(r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})")).sum()),
    })
    implied = (qty * price * (1 - disc.fillna(0) / 100)).round(2)
    ratio = (rev / implied).round(2)
    show("E38 cross-field ratio tails", {
        "ratio_gt_1.1": int((ratio > 1.1).sum()),
        "ratio_lt_0.9": int((ratio < 0.9).sum()),
    })
    show("E39 missing after conversion", {
        "region": int(df["region"].replace({"unknown": np.nan, "N/A": np.nan}).isna().sum()),
        "discount": int(df["discount_pct"].replace({"9999": np.nan}).isna().sum()),
        "customer": int(df["customer_id"].replace({"-1": np.nan}).isna().sum()),
    })
    show("E40 completeness after cleaning", round(
        float((1 - cleaned.isna().mean().mean()) * 100), 3))


def section_extra3() -> None:
    """Final decision tables and run manifest for Chapter 6."""
    df = read_messy()
    key = ["order_id", "product_id"]
    cleaned = clean_orders(df, CONFIG, [])

    # Z1 normalization step table
    cmap = CONFIG["category_map"]
    steps = [("raw", len(df), int(df["category"].nunique()))]
    d = df.drop_duplicates()
    steps.append(("drop exact duplicates", len(d), int(d["category"].nunique())))
    d2 = d.assign(category=d["category"].astype("string")
                  .str.replace("\u00a0", " ", regex=False).str.strip().str.title())
    steps.append(("trim + title case", len(d2), int(d2["category"].nunique())))
    d3 = d2.assign(category=d2["category"].map(
        lambda x: cmap.get(str(x).lower(), x) if pd.notna(x) else x))
    steps.append(("map canonical levels", len(d3), int(d3["category"].nunique())))
    show("Z1 normalization steps", pd.DataFrame(
        steps, columns=["step", "rows", "category_levels"]).to_string(index=False))

    # Z2 duplicate policy comparison
    def most_complete(frame):
        return (frame.assign(_m=frame.isna().sum(axis=1))
                .sort_values(key + ["_m"], kind="stable")
                .drop_duplicates(subset=key, keep="first"))
    pol = []
    for name, frame in [("keep first", df.drop_duplicates(subset=key, keep="first")),
                        ("keep last", df.drop_duplicates(subset=key, keep="last")),
                        ("keep most complete", most_complete(df))]:
        r = pd.to_numeric(frame["revenue"], errors="coerce")
        pol.append((name, len(frame), round(float(r.sum()), 2), round(float(r.max()), 2)))
    show("Z2 duplicate policy comparison", pd.DataFrame(
        pol, columns=["policy", "rows", "gross", "largest_line"]).to_string(index=False))

    # Z3 run manifest
    import hashlib
    p = AUDIT / "orders_messy_40k.csv"
    show("Z3 run manifest", {
        "input": "data/raw/audit/orders_messy_40k.csv",
        "input_bytes": int(p.stat().st_size),
        "input_sha256_12": hashlib.sha256(p.read_bytes()).hexdigest()[:12],
        "pandas": pd.__version__, "numpy": np.__version__,
    })

    # Z4 final control checklist
    rev0 = pd.to_numeric(df["revenue"], errors="coerce")
    dupe = df.duplicated()
    before = round(float(rev0.sum()), 2)
    dup_rev = round(float(rev0[dupe].sum()), 2)
    after = round(float(cleaned["revenue"].sum()), 2)
    checks = [
        ("row count did not grow", len(cleaned) <= len(df)),
        ("distinct orders unchanged", cleaned["order_id"].nunique() == df["order_id"].nunique()),
        ("no exact duplicates remain", int(cleaned.duplicated().sum()) == 0),
        ("gross bridge closes", abs((before - dup_rev) - round(
            float(rev0[~dupe].sum()), 2)) < 0.01),
    ]
    show("Z4 final control checklist", pd.DataFrame(
        [(n, "PASS" if ok else "FAIL") for n, ok in checks],
        columns=["check", "result"]).to_string(index=False))

    # Z5 exception severity
    exc = pd.read_csv(REPORTS / "ch06_exceptions.csv")
    sev = exc.assign(severity=exc["issue"].map(
        {"future": "high", "out_of_range": "high", "non_positive": "medium"}))
    show("Z5 exceptions by issue and severity",
         sev.groupby(["issue", "severity"]).size().to_string())

    # Z6 artifact inventory
    map_df = pd.DataFrame([
        ("camping", "Camping"), ("footwear", "Footwear"),
        ("apparel", "Apparel"), ("accessories", "Accessories"),
        ("outdoor gear", "Camping"), ("sportswear", "Apparel"),
    ], columns=["raw_level", "canonical"])
    inv = pd.DataFrame([
        ("data/processed/orders_clean_ch06.csv", len(cleaned), "clean order lines"),
        ("data/processed/orders_exceptions_ch06.csv", len(exc), "cell-level exceptions"),
        ("reports/ch06_duplicate_audit.csv", int(df.duplicated().sum()),
         "deduplication audit trail"),
        ("reports/ch06_category_map.csv", len(map_df), "versioned category map"),
    ], columns=["path", "rows", "purpose"])
    show("Z6 artifact inventory", inv.to_string(index=False))


def main() -> None:
    print("python", "pandas", pd.__version__, "numpy", np.__version__)
    print()
    for section in (
        section_61, section_62, section_63, section_64, section_65, section_66,
        section_67, section_68, section_69, section_610, section_611,
        section_extra, section_extra2, section_extra3,
    ):
        print("=" * 78)
        section()


if __name__ == "__main__":
    main()
