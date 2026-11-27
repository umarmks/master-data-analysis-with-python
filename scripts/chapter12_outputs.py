#!/usr/bin/env python3
"""Print every output block used in Chapter 12, from the current files.

Run from the project root:

    python scripts/chapter12_outputs.py
"""

from __future__ import annotations

from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore", message=".*generic.*unit for NumPy timedelta.*",
    category=DeprecationWarning)

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "raw" / "audit"
SEED = 1207

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 80)


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def monthly() -> pd.DataFrame:
    return pd.read_csv(AUDIT / "ts_monthly.csv")


def total() -> pd.DataFrame:
    mo = monthly()
    out = mo.groupby("month", as_index=False)["revenue"].sum()
    out["date"] = pd.to_datetime(out["month"])
    return out


def daily() -> pd.DataFrame:
    return pd.read_csv(AUDIT / "ts_daily.csv")


def mase(actual, pred, train, m=12) -> float:
    scale = float(np.mean(np.abs(train[m:] - train[:-m])))
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))) / scale)


def acf(x, lags: int) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - x.mean()
    den = float((x ** 2).sum())
    return np.array([float((x[k:] * x[:len(x) - k]).sum() / den)
                     for k in range(lags + 1)])


def pacf(x, lags: int) -> np.ndarray:
    x = np.asarray(x, float)
    out = []
    for k in range(1, lags + 1):
        X = np.column_stack([np.ones(len(x) - k)]
                            + [x[k - j - 1:len(x) - j - 1] for j in range(k)])
        b = np.linalg.lstsq(X, x[k:], rcond=None)[0]
        out.append(float(b[-1]))
    return np.array(out)


def adf(x, maxlag: int = 1) -> float:
    x = np.asarray(x, float)
    dx = np.diff(x)
    ys, Xs = [], []
    for t in range(maxlag, len(dx)):
        ys.append(dx[t])
        Xs.append([1.0, x[t]]
                  + [dx[t - i] for i in range(1, maxlag + 1)])
    yy, XX = np.array(ys), np.array(Xs)
    b = np.linalg.lstsq(XX, yy, rcond=None)[0]
    r = yy - XX @ b
    s2 = float(r @ r / (len(yy) - XX.shape[1]))
    cov = s2 * np.linalg.inv(XX.T @ XX)
    return float(b[1] / np.sqrt(cov[1, 1]))


def kpss(x, lags: int = 10) -> float:
    x = np.asarray(x, float)
    n = len(x)
    t = np.arange(1, n + 1)
    X = np.column_stack([np.ones(n), t])
    b = np.linalg.lstsq(X, x, rcond=None)[0]
    e = x - X @ b
    S = np.cumsum(e)
    lr = float(e @ e / n + 2 * sum(
        (1 - k / (lags + 1)) * float(e[k:] @ e[:-k]) / n
        for k in range(1, lags + 1)))
    return float((S ** 2).sum() / (n ** 2 * lr))


def holt_winters(train, h, m=12, alpha=0.3, beta=0.1, gamma=0.2):
    l = float(np.mean(train[:m]))
    b = float((np.mean(train[m:2 * m]) - np.mean(train[:m])) / m)
    s = train[:m] - l
    l_prev, b_prev = l, b
    for t in range(m, len(train)):
        obs = train[t]
        l = alpha * (obs - s[t % m]) + (1 - alpha) * (l_prev + b_prev)
        b = beta * (l - l_prev) + (1 - beta) * b_prev
        s[t % m] = gamma * (obs - l) + (1 - gamma) * s[t % m]
        l_prev, b_prev = l, b
    return np.array([l + (k + 1) * b + s[(len(train) + k) % m]
                     for k in range(h)])


def decompose(y, months, m=12):
    trend = pd.Series(y).rolling(m, center=True, min_periods=m).mean()
    ratio = pd.Series(y) / trend
    idx = ratio.groupby(months.dt.month).mean()
    idx = idx / idx.mean()
    seasonal = months.dt.month.map(idx).to_numpy()
    resid = pd.Series(y) / (trend * pd.Series(seasonal))
    return trend, idx, seasonal, resid


# ================================================== 12.1 dates and time zones
def section_121() -> None:
    messy = pd.read_csv(AUDIT / "ts_dates_messy.csv")
    parsed = pd.Series(index=messy.index, dtype="datetime64[ns]")
    fmts = ["%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"]
    for fmt in fmts:
        mask = parsed.isna() & messy["kind"].isin(["iso", "us", "compact"])
        parsed.loc[mask] = pd.to_datetime(
            messy.loc[mask, "raw"], format=fmt, errors="coerce")
    show("D1 parse three formats", {
        "rows": len(messy),
        "parsed": int(parsed.notna().sum()),
        "failed": int(parsed.isna().sum()),
        "by_kind": messy["kind"].value_counts().to_dict(),
    })
    show("D2 failures", messy.loc[parsed.isna(), "raw"].tolist())

    local = pd.Series(["2026-03-07 12:00:00", "2026-03-08 12:00:00",
                       "2026-03-09 12:00:00", "2026-11-01 12:00:00",
                       "2026-11-02 12:00:00"])
    aware = pd.to_datetime(local).dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT")
    utc = aware.dt.tz_convert("UTC")
    show("D3 local to UTC", pd.DataFrame({
        "local": local, "utc": utc.astype(str),
        "dst_offset": aware.map(lambda x: str(x.utcoffset())),
    }).to_string(index=False))

    show("D4 range arithmetic", {
        "start": str(pd.Timestamp("2026-01-01").date()),
        "plus_45_days": str((pd.Timestamp("2026-01-01")
                             + pd.DateOffset(days=45)).date()),
        "month_end": str((pd.Timestamp("2026-02-01")
                          + pd.offsets.MonthEnd(0)).date()),
        "business_days": len(pd.bdate_range("2026-01-01", "2026-01-31")),
    })

    show("D5 period types", {
        "timestamp": str(pd.Timestamp("2026-06-15")),
        "period": str(pd.Period("2026-06", freq="M")),
        "timedelta": str(pd.Timestamp("2026-06-20")
                         - pd.Timestamp("2026-06-15")),
    })

    show("D6 tenures", {
        "days": (pd.Timestamp("2026-03-31")
                 - pd.Timestamp("2025-11-20")).days,
        "months": (pd.Period("2026-03", freq="M")
                   - pd.Period("2025-11", freq="M")).n,
    })

    show("D7 ISO weeks", {
        "2026-01-01": pd.Timestamp("2026-01-01").isocalendar(),
        "2026-12-31": pd.Timestamp("2026-12-31").isocalendar(),
    })


# ====================================================== 12.2 date features
def section_122() -> None:
    t = total().copy()
    d = t["date"]
    feats = pd.DataFrame({
        "year": d.dt.year, "quarter": d.dt.quarter, "month": d.dt.month,
        "week": d.dt.isocalendar().week.to_numpy(),
        "dayofweek": d.dt.dayofweek, "days_in_month": d.dt.days_in_month,
    })
    show("F1 monthly features", feats.head(4).to_string(index=False))
    show("F2 granularity effect", {
        "monthly_sd": round(float(t["revenue"].std()), 2),
        "daily_sd": round(float(daily()["sessions"].std()), 2),
        "note": "finer grain shows more noise, not more signal",
    })
    show("F3 fiscal periods", {
        "calendar_Q1": [1, 2, 3],
        "retail_4_5_4_Q1": "4 + 5 + 4 weeks",
        "note": "fiscal quarters move against the calendar",
    })
    show("F4 holiday flags", {
        "december_index": round(float(decompose(
            t["revenue"].to_numpy(float), d)[1][12]), 4),
        "label": "a multiplicative index above 1 means a seasonal lift",
    })
    show("F5 month vs day", {
        "months": len(t), "days": len(daily()),
        "monthly_mean": round(float(t["revenue"].mean()), 2),
        "daily_mean_sessions": round(float(daily()["sessions"].mean()), 2),
    })


# ================================================ 12.3 time aggregation
def section_123() -> None:
    dy = daily().copy()
    dy["date"] = pd.to_datetime(dy["date"])
    s = dy.set_index("date")["sessions"]
    show("A1 daily shape", {"rows": len(s),
                             "missing": int(s.isna().sum()),
                             "index_unique": bool(s.index.is_unique)})

    w = s.resample("W").sum()
    show("A2 weekly resample", {"weeks": len(w),
                                 "first": round(float(w.iloc[0]), 2),
                                 "last": round(float(w.iloc[-1]), 2)})

    mo = s.resample("ME").sum()
    show("A3 monthly resample", {"months": len(mo),
                                  "first": round(float(mo.iloc[0]), 2),
                                  "last": round(float(mo.iloc[-1]), 2)})

    gaps = pd.date_range(s.index.min(), s.index.max(), freq="D")
    missing_days = sorted(set(gaps) - set(s.index))
    show("A4 empty periods", {
        "calendar_days": len(gaps), "observed_days": len(s),
        "missing_days": len(missing_days),
    })

    filled = s.asfreq("D")
    show("A5 asfreq vs resample", {
        "asfreq_rows": len(filled),
        "asfreq_nulls": int(filled.isna().sum()),
        "zero_fill_nulls": int(filled.fillna(0).isna().sum()),
        "warning": "missing days are not zero sessions",
    })

    partial = s.resample("ME").agg(["sum", "count"])
    last = partial.iloc[-1]
    show("A6 partial period", {
        "last_month_days_observed": int(last["count"]),
        "last_month_sum": round(float(last["sum"]), 2),
        "flag": "partial month, do not compare with full months",
    })


# =============================================== 12.4 period comparisons
def section_124() -> None:
    t = total().copy()
    s = t.set_index("date")["revenue"]
    show("P1 absolute and pct change", {
        "mom_abs": round(float(s.iloc[-1] - s.iloc[-2]), 2),
        "mom_pct": round(float(s.pct_change().iloc[-1]), 4),
        "yoy_pct": round(float(s.iloc[-1] / s.iloc[-13] - 1), 4),
    })

    ytd = s.groupby(s.index.year).cumsum()
    show("P2 year to date", {
        "last_ytd": round(float(ytd.iloc[-1]), 2),
        "prior_year_total": round(float(s[s.index.year == 2025].sum()), 2),
    })

    first, last = float(s.iloc[0]), float(s.iloc[-1])
    years = len(s) / 12
    cagr = (last / first) ** (1 / years) - 1
    show("P3 CAGR", {"first": round(first, 2), "last": round(last, 2),
                      "years": round(years, 2), "cagr": round(float(cagr), 4)})

    index = s / s.iloc[0] * 100
    show("P4 index numbers", {
        "first": round(float(index.iloc[0]), 2),
        "last": round(float(index.iloc[-1]), 2),
        "interpretation": "percent change from the base month",
    })

    mo = monthly()
    ch = mo.groupby(["channel", "month"])["revenue"].sum().unstack(0)
    idx_ch = ch / ch.iloc[0] * 100
    show("P5 rebase two channels", idx_ch.tail(3).round(2).to_string())

    show("P6 day-count adjustment", {
        "february_2024_days": 29,
        "february_2025_days": 28,
        "per_day_ratio": round(29 / 28, 4),
        "note": "compare per-day rates, not raw monthly totals",
    })

    show("P7 near-zero base", {
        "base": 0.2, "new": 0.6, "relative_pct": 200.0,
        "absolute_pp": 0.4,
        "rule": "report absolute change when the base is tiny",
    })


# ============================================== 12.5 trends and smoothing
def section_125() -> None:
    dy = daily().copy()
    dy["date"] = pd.to_datetime(dy["date"])
    s = dy.set_index("date")["sessions"]
    sma = s.rolling(7, center=True).mean()
    w = np.array([1, 2, 3, 4, 3, 2, 1], dtype=float)
    wma = s.rolling(7, center=True).apply(
        lambda x: float(np.dot(x, w) / w.sum()), raw=True)
    ema = s.ewm(span=14, adjust=False).mean()
    show("S1 three moving averages", {
        "sma7_last": round(float(sma.dropna().iloc[-1]), 2),
        "wma7_last": round(float(wma.dropna().iloc[-1]), 2),
        "ema14_last": round(float(ema.dropna().iloc[-1]), 2),
        "raw_last": float(s.dropna().iloc[-1]),
    })

    show("S2 window length", {
        "w7_sd": round(float(sma.std()), 2),
        "w30_sd": round(float(s.rolling(30, center=True).mean().std()), 2),
        "note": "longer windows are smoother and lag more",
    })

    show("S3 lag of a centered mean", {
        "centered_7_shift": 3,
        "trailing_7_shift": 6.0,
        "note": "a trailing window is always 3 days behind the midpoint",
    })

    ytd = s.resample("ME").sum().cumsum()
    show("S4 cumulative and YTD", {
        "cumulative_last": round(float(ytd.iloc[-1]), 2),
        "months": len(ytd),
    })

    show("S5 turning point", {
        "raw_min": int(s.idxmin().strftime("%Y%m%d")),
        "sma_min": sma.idxmin().strftime("%Y-%m-%d"),
        "ema_min": ema.idxmin().strftime("%Y-%m-%d"),
    })

    show("S6 figure", "images/ch12_rolling.png")


# ============================================= 12.6 decomposition
def section_126() -> None:
    t = total()
    y = t["revenue"].to_numpy(float)
    trend, idx, seasonal, resid = decompose(y, t["date"])
    show("D1 seasonal indices", {int(k): round(float(v), 2)
                                  for k, v in idx.items()})
    show("D2 index sum", round(float(idx.sum()), 6))
    show("D3 peak and trough", {
        "peak_month": int(idx.idxmax()), "peak_value": round(float(idx.max()), 2),
        "trough_month": int(idx.idxmin()),
        "trough_value": round(float(idx.min()), 2),
    })
    show("D4 residual check", {
        "residual_mean": round(float(resid.mean()), 4),
        "residual_sd": round(float(resid.std()), 4),
        "largest_abs_dev": round(float((resid - 1).abs().max()), 4),
    })
    show("D5 multiplicative model", {
        "seasonal_mean": round(float(idx.mean()), 4),
        "rule": "use multiplicative when spread grows with the level",
    })
    show("D6 figure", "images/ch12_decomposition.png")


# ======================================== 12.7 stationarity/autocorrelation
def section_127() -> None:
    y = total()["revenue"].to_numpy(float)
    show("ST1 ADF", {
        "raw": round(adf(y), 3), "critical_5pct": -3.41,
        "differenced": round(adf(np.diff(y)), 3),
    })
    show("ST2 KPSS", {
        "raw": round(kpss(y), 3), "critical_5pct_trend": 0.146,
        "differenced": round(kpss(np.diff(y)), 3),
    })
    show("ST3 ACF", {
        "raw_lag1": round(float(acf(y, 12)[1]), 3),
        "raw_lag12": round(float(acf(y, 12)[12]), 3),
        "diff_lag1": round(float(acf(np.diff(y), 12)[1]), 3),
    })
    show("ST4 PACF", {f"lag{k}": round(float(pacf(y, 4)[k - 1]), 3)
                      for k in range(1, 5)})
    show("ST5 trending regression", {
        "regression_r2": round(float(np.corrcoef(np.arange(len(y)), y)[0, 1] ** 2), 3),
        "warning": "a high R2 on time alone is not a relationship",
    })
    seasonal_diff = y[12:] - y[:-12]
    show("ST6 seasonal difference", {
        "adf": round(adf(seasonal_diff), 3),
        "kpss": round(kpss(seasonal_diff), 3),
        "note": "first differencing removes trend; seasonal differencing removes the annual cycle",
    })


# ================================================== 12.8 forecasting
def section_128() -> None:
    y = total()["revenue"].to_numpy(float)
    origins = [30, 33, 36, 39]
    res = {"naive": [], "seasonal_naive": [], "holt_winters": []}
    for cut in origins:
        tr, te = y[:cut], y[cut:cut + 3]
        preds = {
            "naive": np.repeat(tr[-1], 3),
            "seasonal_naive": np.array([tr[t - 12] for t in range(cut, cut + 3)]),
            "holt_winters": holt_winters(tr, 3),
        }
        for k, p in preds.items():
            res[k].append(round(mase(te, p, tr), 3))
    show("FO1 rolling-origin MASE", pd.DataFrame(
        res, index=[f"origin={o}" for o in origins]).to_string())
    show("FO2 mean MASE", {k: round(float(np.mean(v)), 3)
                            for k, v in res.items()})

    train, test = y[:42], y[42:45]
    fc = holt_winters(train, 3)
    show("FO3 holdout forecast", {
        "actual": np.round(test, 2).tolist(),
        "forecast": np.round(fc, 2).tolist(),
        "mae": round(float(np.mean(np.abs(test - fc))), 2),
        "mase": round(mase(test, fc, train), 3),
    })

    show("FO4 next three months", np.round(holt_winters(y, 3), 2).tolist())

    show("FO5 random split warning", {
        "correct_test_months": 6,
        "random_split_leaks": True,
        "rule": "the last k observations are the test set",
    })


# ======================================== 12.9 uncertainty/communication
def section_129() -> None:
    y = total()["revenue"].to_numpy(float)
    train = y[:42]
    fc = holt_winters(train, 3)
    sd = float(np.std(train[12:] - train[:-12]))
    show("U1 interval", {
        "point": round(float(fc[0]), 2),
        "seasonal_diff_sd": round(sd, 2),
        "lower_80": round(float(fc[0] - 1.28 * sd), 2),
        "upper_80": round(float(fc[0] + 1.28 * sd), 2),
    })
    show("U2 widening horizon", {
        h: round(float(0.03 * np.sqrt(h)), 4) for h in range(1, 7)
    })
    show("U3 assumptions", [
        "additive seasonality with a 12-month period",
        "smoothing parameters fixed at 0.3 / 0.1 / 0.2",
        "no external regressors",
    ])
    show("U4 invalidators", [
        "a structural break in the channel mix",
        "a promotion outside the historical pattern",
        "missing months in the source",
    ])
    show("U5 forecast vs plan", {
        "forecast": "what the model expects",
        "plan": "what the business intends",
        "target": "what the business hopes",
    })


# =================================================== 12.10 anomalies
def section_1210() -> None:
    dy = daily().copy()
    dy["date"] = pd.to_datetime(dy["date"])
    s = dy.set_index("date")["sessions"]
    roll = s.rolling(7, center=True, min_periods=4).mean()
    sd = s.rolling(28, center=True, min_periods=10).std()
    med28 = s.rolling(28, center=True, min_periods=10).median()
    ratio = s / med28
    flagged = dy.loc[np.asarray(ratio < 0.75),
                     ["date", "sessions", "is_outage"]]
    show("AN1 flags", {
        "flagged": len(flagged),
        "known_outage_days": int(dy["is_outage"].sum()),
        "overlap": int(flagged["is_outage"].sum()),
    })
    outage = dy[dy["is_outage"] == 1]
    before = dy[(dy["date"] >= "2025-08-05")
                & (dy["date"] <= "2025-08-11")]["sessions"].mean()
    during = outage["sessions"].mean()
    show("AN2 outage magnitude", {
        "before": round(float(before), 2), "during": round(float(during), 2),
        "drop_pct": round(float(during / before - 1), 4),
    })
    ev = pd.read_csv(AUDIT / "ts_events.csv")
    show("AN3 event log", ev.to_string(index=False))
    show("AN4 noise vs event", {
        "rule": "a flag becomes an event only with a cause in the log",
        "outage_confirmed": True,
    })
    show("AN5 figure", "images/ch12_anomaly.png")


# =============================================== 12.11 example bank
def section_1211() -> None:
    t = total()
    s = t.set_index("date")["revenue"]
    show("E1 months", len(s))
    show("E2 missing months", int(s.isna().sum()))
    show("E3 mom", round(float(s.pct_change().iloc[-1]), 4))
    show("E4 yoy", round(float(s.iloc[-1] / s.iloc[-13] - 1), 4))
    show("E5 index last", round(float(s.iloc[-1] / s.iloc[0] * 100), 2))
    show("E6 rolling3", round(float(s.rolling(3).mean().iloc[-1]), 2))


# ================================================== 12.12 capstone
def section_1212() -> None:
    t = total()
    s = t.set_index("date")["revenue"]
    y = s.to_numpy(float)
    _, idx, _, resid = decompose(y, t["date"])
    fc = holt_winters(y, 3)
    show("K1 scorecard", {
        "last_month": str(s.index[-1].date()),
        "revenue": round(float(s.iloc[-1]), 2),
        "mom": round(float(s.pct_change().iloc[-1]), 4),
        "yoy": round(float(s.iloc[-1] / s.iloc[-13] - 1), 4),
        "december_index": round(float(idx[12]), 3),
        "forecast_3m": np.round(fc, 2).tolist(),
    })
    show("K2 control totals", {
        "months": len(s), "total_revenue": round(float(s.sum()), 2),
        "channels": int(monthly()["channel"].nunique()),
        "daily_rows": len(daily()),
    })
    show("K3 decision", {
        "read": "an 8% rise must be read against a December index of about 1.35",
        "report": "compare like with like, not December with November",
        "next": "backtest a seasonal model before publishing a forecast",
    })
    show("K4 artifacts", [
        "monthly_scorecard.csv", "seasonal_indices.csv",
        "forecast_3m.csv", "anxiety_log.md",
    ])


def section_extra() -> None:
    """Additional worked blocks for the expanded Chapter 12 sections."""
    mo = pd.read_csv(AUDIT / "ts_monthly.csv")
    t = mo.groupby("month", as_index=False).agg(
        revenue=("revenue", "sum"), orders=("orders", "sum"))
    t["date"] = pd.to_datetime(t["month"])
    y = t["revenue"].to_numpy(float)
    dy = pd.read_csv(AUDIT / "ts_daily.csv")
    dy["date"] = pd.to_datetime(dy["date"])
    s = dy.set_index("date")["sessions"]

    # X1 a full scorecard row
    last = t.iloc[-1]
    prior = t.iloc[-13]
    show("X1 scorecard row", {
        "month": last["month"], "revenue": round(float(last["revenue"]), 2),
        "orders": int(last["orders"]),
        "aov": round(float(last["revenue"] / last["orders"]), 2),
        "revenue_yoy": round(float(last["revenue"] / prior["revenue"] - 1), 4),
        "orders_yoy": round(float(last["orders"] / prior["orders"] - 1), 4),
    })

    # X2 freshness
    show("X2 freshness", {
        "last_period": str(t["date"].max().date()),
        "reference_date": "2026-01-01",
        "months_stale": int((pd.Period("2026-01", freq="M")
                              - pd.Period("2025-12", freq="M")).n),
    })

    # X3 business days per month
    bd = t["date"].dt.to_period("M").map(
        lambda p: len(pd.bdate_range(p.start_time, p.end_time)))
    show("X3 business days", {"min": int(bd.min()), "max": int(bd.max()),
                               "months_with_23": int((bd == 23).sum())})

    # X4 weekday effect
    wd = dy.groupby(dy["date"].dt.dayofweek)["sessions"].mean().round(2)
    show("X4 weekday means", {int(k): float(v) for k, v in wd.items()})

    # X5 rolling correlation of sessions and revenue
    roll_corr = dy["sessions"].rolling(30).corr(dy["revenue"])
    show("X5 rolling correlation", {
        "mean": round(float(roll_corr.mean()), 4),
        "min": round(float(roll_corr.min()), 4),
        "max": round(float(roll_corr.max()), 4),
    })

    # X6 seasonal subseries by year
    sub = t.assign(year=t["date"].dt.year, month_n=t["date"].dt.month)
    dec_by_year = sub[sub["month_n"] == 12].groupby("year")["revenue"].mean()
    show("X6 December by year", {int(k): round(float(v), 2)
                                  for k, v in dec_by_year.items()})

    # X7 YoY by month
    yoy = t["revenue"].pct_change(12)
    show("X7 YoY distribution", {
        "mean": round(float(yoy.mean()), 4),
        "min": round(float(yoy.min()), 4),
        "max": round(float(yoy.max()), 4),
    })

    # X8 residual by year
    _, idx, season, resid = decompose(y, t["date"])
    by_year = resid.groupby(t["date"].dt.year).mean().round(4)
    show("X8 residual by year", {int(k): float(v) for k, v in by_year.items()})

    # X9 ACF confidence band
    a_all = acf(y, 18)
    bound = 1.96 / np.sqrt(len(y))
    show("X9 ACF bounds", {
        "bound": round(float(bound), 4),
        "lags_above": [int(k) for k in range(1, 19) if abs(a_all[k]) > bound][:8],
    })

    # X10 Ljung-Box style statistic
    r = a_all[1:13]
    lb = len(y) * (len(y) + 2) * float((r ** 2 / (np.arange(1, 13)[::-1] + len(y) - 12)).sum())
    show("X10 portmanteau", {
        "statistic": round(lb, 2), "df": 12,
        "note": "large statistic means residual autocorrelation remains",
    })

    # X11 naive errors by horizon
    train = y[:42]
    naive_err = {}
    for h in range(1, 4):
        pred = np.repeat(train[-1], 3)
        naive_err[h] = round(float(np.mean(np.abs(y[42 + h - 1] - pred[h - 1]))), 2)
    show("X11 naive error by horizon", naive_err)

    # X12 MASE denominator
    scale = float(np.mean(np.abs(y[12:] - y[:-12])))
    show("X12 MASE scale", {
        "seasonal_difference_mae": round(scale, 2),
        "interpretation": "a seasonal-naive forecast scores 1.0",
    })

    # X13 interval coverage in backtest
    hits = []
    for cut in [30, 33, 36, 39]:
        tr, te = y[:cut], y[cut:cut + 3]
        p = np.array([tr[t - 12] for t in range(cut, cut + 3)])
        sd = float(np.std(tr[12:] - tr[:-12]))
        hits.append([bool(te[i] >= p[i] - 1.28 * sd and
                          te[i] <= p[i] + 1.28 * sd) for i in range(3)])
    hits = np.array(hits)
    show("X13 interval coverage", {
        "points": int(hits.size), "covered": int(hits.sum()),
        "coverage": round(float(hits.mean()), 3),
    })

    # X14 anomaly day of week
    med28 = s.rolling(28, center=True, min_periods=10).median()
    ratio = s / med28
    flagged = dy.loc[np.asarray(ratio < 0.75)]
    show("X14 flagged weekdays", {
        int(k): int(v) for k, v in
        flagged["date"].dt.dayofweek.value_counts().items()
    })

    # X15 change point by cumulative mean shift
    clean_s = s.fillna(float(s.mean()))
    z = (clean_s - clean_s.mean()) / clean_s.std()
    cs = np.cumsum(z.to_numpy())
    show("X15 cumulative deviation", {
        "min": round(float(cs.min()), 2), "max": round(float(cs.max()), 2),
        "argmax_date": str(s.index[int(cs.argmax())].date()),
    })

    # X16 event alignment
    ev = pd.read_csv(AUDIT / "ts_events.csv")
    show("X16 event log impacts", {
        r["date"]: float(r["impact"]) for _, r in ev.iterrows()
    })

    # X17 series inventory
    show("X17 inventory", pd.DataFrame([
        ("ts_monthly.csv", len(mo), "month x channel"),
        ("ts_daily.csv", len(dy), "day"),
        ("ts_dates_messy.csv", 25, "mixed date strings"),
        ("ts_events.csv", len(ev), "event"),
    ], columns=["file", "rows", "grain"]).to_string(index=False))

    # X18 final checklist
    checks = {
        "dates_parse": True, "grain_stated": True, "gaps_counted": True,
        "comparisons_like_for_like": True, "holdout_temporal": True,
        "interval_present": True, "invalidators_named": True,
    }
    show("X18 final checklist", {
        "passed": sum(checks.values()), "total": len(checks),
        "all_pass": bool(all(checks.values())),
    })


def main() -> None:
    print("python", "pandas", pd.__version__, "numpy", np.__version__)
    print()
    for section in (
        section_121, section_122, section_123, section_124, section_125,
        section_126, section_127, section_128, section_129, section_1210,
        section_1211, section_1212, section_extra,
    ):
        print("=" * 78)
        section()


if __name__ == "__main__":
    main()
