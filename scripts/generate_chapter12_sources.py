#!/usr/bin/env python3
"""Generate the time-series fixtures and figures that Chapter 12 uses.

Run from the project root:

    python scripts/generate_chapter12_sources.py

Outputs (all under data/raw/audit/ except the figures):

    ts_monthly.csv       48 months x 3 channels, with trend and seasonality
    ts_daily.csv         730 daily sessions with weekly seasonality and an outage
    ts_dates_messy.csv   three date formats plus timezone/DST cases
    ts_events.csv        a small event log for anomaly investigation
    images/ch12_*.png    six figures used in the chapter

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
AUDIT = ROOT / "data" / "raw" / "audit"
IMAGES = ROOT / "images"
SEED = 1207


def monthly() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    months = pd.date_range("2022-01-01", periods=48, freq="MS")
    rows = []
    for ch, base, share in [("Web", 120000, 0.55),
                            ("App", 80000, 0.30),
                            ("Phone", 40000, 0.15)]:
        for t, m in enumerate(months):
            trend = 1 + 0.006 * t
            season = 1 + 0.18 * np.sin(2 * np.pi * (m.month - 9) / 12)
            dec = 1.15 if m.month == 12 else 1.0
            rev = base * share * 2.4 * trend * season * dec * float(
                rng.normal(1, 0.035))
            orders = int(rev / float(rng.uniform(160, 210)))
            rows.append((m.strftime("%Y-%m-01"), ch, round(rev, 2), orders))
    df = pd.DataFrame(rows, columns=["month", "channel", "revenue", "orders"])
    df.to_csv(AUDIT / "ts_monthly.csv", index=False)
    print(f"  ts_monthly.csv: {len(df)} rows")
    return df


def daily() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2024-01-01", periods=730, freq="D")
    trend = 1200 + 0.8 * np.arange(len(dates))
    weekly = np.where(np.asarray(dates.dayofweek) % 7 >= 5, -180, 90)
    season = 120 * np.sin(2 * np.pi * np.asarray(dates.dayofyear) / 365)
    sessions = np.round(trend + weekly + season
                        + rng.normal(0, 90, size=len(dates))).astype(int)
    sessions = np.maximum(sessions, 100)
    outage = (dates >= "2025-08-12") & (dates <= "2025-08-14")
    sessions[outage] = np.round(sessions[outage] * 0.62).astype(int)
    revenue = np.round(sessions * rng.uniform(1.8, 2.4, size=len(dates)), 2)
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"),
                       "sessions": sessions, "revenue": revenue,
                       "is_outage": outage.astype(int)})
    missing = rng.choice(len(df), size=5, replace=False)
    df.loc[missing, "sessions"] = np.nan
    df.to_csv(AUDIT / "ts_daily.csv", index=False)
    print(f"  ts_daily.csv: {len(df)} rows, "
          f"{int(df['sessions'].isna().sum())} missing")
    return df


def dates_messy() -> pd.DataFrame:
    rows = []
    iso = pd.date_range("2026-01-01", periods=6, freq="D")
    us = pd.date_range("2026-02-01", periods=6, freq="D")
    compact = pd.date_range("2026-03-01", periods=6, freq="D")
    for d in iso:
        rows.append((d.strftime("%Y-%m-%d"), "iso"))
    for d in us:
        rows.append((d.strftime("%m/%d/%Y"), "us"))
    for d in compact:
        rows.append((d.strftime("%Y%m%d"), "compact"))
    rows.append(("not a date", "junk"))
    rows.append(("2026-02-30", "junk"))
    local = ["2026-03-07 12:00:00", "2026-03-08 12:00:00",
             "2026-03-09 12:00:00", "2026-11-01 12:00:00",
             "2026-11-02 12:00:00"]
    for v in local:
        rows.append((v, "local_timestamp"))
    df = pd.DataFrame(rows, columns=["raw", "kind"])
    df.to_csv(AUDIT / "ts_dates_messy.csv", index=False)
    print(f"  ts_dates_messy.csv: {len(df)} rows")
    return df


def events() -> pd.DataFrame:
    df = pd.DataFrame([
        ("2025-08-12", "outage", "tracking pipeline down", -0.38),
        ("2025-08-15", "recovery", "tracking restored", 0.29),
        ("2025-03-01", "campaign", "spring promotion", 0.11),
        ("2025-11-28", "campaign", "black Friday", 0.34),
    ], columns=["date", "kind", "note", "impact"])
    df.to_csv(AUDIT / "ts_events.csv", index=False)
    print(f"  ts_events.csv: {len(df)} events")
    return df


def figures(mo: pd.DataFrame, dy: pd.DataFrame) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 110})

    total = mo.groupby("month", as_index=False)["revenue"].sum()
    y = total["revenue"].to_numpy(float)
    months = pd.to_datetime(total["month"])

    # classical multiplicative decomposition components
    m = 12
    trend = pd.Series(y).rolling(window=m, center=True, min_periods=m).mean()
    ratio = pd.Series(y) / trend
    seasonal_idx = ratio.groupby(months.dt.month).mean()
    seasonal_idx = seasonal_idx / seasonal_idx.mean()
    seasonal = months.dt.month.map(seasonal_idx).to_numpy()
    resid = pd.Series(y) / (trend * pd.Series(seasonal))

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(months, y, label="observed")
    axes[0].plot(months, trend, color="#cc6677", label="trend")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Observed and trend")
    axes[1].bar(range(1, 13), [seasonal_idx[i] for i in range(1, 13)],
                color="#4477aa")
    axes[1].set_title("Monthly seasonal indices")
    axes[2].plot(months, seasonal, color="#228833")
    axes[2].set_title("Seasonal component")
    axes[3].plot(months, resid, color="#aa3377")
    axes[3].axhline(0, color="black", lw=0.8)
    axes[3].set_title("Residual")
    fig.suptitle("Classical additive decomposition of monthly revenue")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch12_decomposition.png")
    plt.close(fig)

    # ACF and PACF before and after differencing
    def acf(x, lags):
        x = np.asarray(x, float)
        x = x - x.mean()
        den = (x ** 2).sum()
        return np.array([float((x[k:] * x[:-k]).sum() / den)
                         for k in range(1, lags + 1)])

    def pacf(x, lags):
        x = np.asarray(x, float)
        out = []
        for k in range(1, lags + 1):
            X = np.column_stack([np.ones(len(x) - k)]
                                + [x[k - j - 1:len(x) - j - 1]
                                   for j in range(k)])
            b = np.linalg.lstsq(X, x[k:], rcond=None)[0]
            out.append(float(b[-1]))
        return np.array(out)

    d1 = np.diff(y)
    lags = np.arange(1, 19)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for row, (series, label) in enumerate([(y, "raw"), (d1, "differenced")]):
        axes[row, 0].bar(lags, acf(series, 18), color="#4477aa")
        axes[row, 0].axhline(0, color="black", lw=0.8)
        axes[row, 0].set_title(f"ACF {label}")
        axes[row, 1].bar(lags, pacf(series, 18), color="#ee7733")
        axes[row, 1].axhline(0, color="black", lw=0.8)
        axes[row, 1].set_title(f"PACF {label}")
    fig.suptitle("Autocorrelation before and after first differencing")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch12_acf_pacf.png")
    plt.close(fig)

    # rolling averages on daily sessions
    s = dy.set_index("date")["sessions"]
    s.index = pd.to_datetime(s.index)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s.index, s, alpha=0.35, color="#bbbbbb", label="daily")
    ax.plot(s.index, s.rolling(7, center=True).mean(), color="#4477aa",
            label="7-day")
    ax.plot(s.index, s.rolling(30, center=True).mean(), color="#cc6677",
            label="30-day")
    ax.legend()
    ax.set_title("Daily sessions with 7- and 30-day moving averages")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(IMAGES / "ch12_rolling.png")
    plt.close(fig)

    # forecast: seasonal naive with naive interval
    train_n = 42
    train, test = y[:train_n], y[train_n:]
    fc = np.array([y[t - 12] for t in range(train_n, len(y))])
    errors = test - fc
    sd = float(np.std(errors[:3])) if len(errors) > 3 else float(np.std(train))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(months, y, color="#4477aa", label="actual")
    ax.plot(months[train_n:], fc, "o--", color="#cc6677",
            label="seasonal naive")
    ax.fill_between(months[train_n:], fc - 1.28 * sd, fc + 1.28 * sd,
                    alpha=0.2, color="#cc6677")
    ax.axvline(months[train_n], color="grey", linestyle=":")
    ax.legend()
    ax.set_title("Seasonal naive forecast with an 80% band")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch12_forecast.png")
    plt.close(fig)

    # anomaly: outage window
    dd = dy.set_index("date")["sessions"].rolling(7, center=True).mean()
    dd.index = pd.to_datetime(dd.index)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dd.index, dd, color="#4477aa")
    ax.axvspan(pd.Timestamp("2025-08-12"), pd.Timestamp("2025-08-14"),
               color="#cc6677", alpha=0.3)
    ax.text(pd.Timestamp("2025-08-12"), dd.max(), " outage", fontsize=8)
    ax.set_title("Smoothed sessions with the tracking outage marked")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch12_anomaly.png")
    plt.close(fig)

    # fan chart of three scenarios
    last = float(y[-1])
    horizon = np.arange(1, 7)
    center = last * (1 + 0.006) ** horizon
    widths = 0.03 * np.sqrt(horizon)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(horizon, center, color="#4477aa", label="base")
    ax.fill_between(horizon, center * (1 - widths), center * (1 + widths),
                    alpha=0.3, color="#4477aa", label="80%")
    ax.fill_between(horizon, center * (1 - 2 * widths),
                    center * (1 + 2 * widths), alpha=0.15,
                    color="#4477aa", label="95%")
    ax.legend()
    ax.set_title("Six-month scenario fan from the last observation")
    fig.tight_layout()
    fig.savefig(IMAGES / "ch12_fan.png")
    plt.close(fig)

    print(f"  images: {len(list(IMAGES.glob('ch12_*.png')))} figures")


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    print("generating Chapter 12 time-series fixtures and figures ...")
    mo = monthly()
    dy = daily()
    dates_messy()
    events()
    figures(mo, dy)
    print("done.")


if __name__ == "__main__":
    main()
