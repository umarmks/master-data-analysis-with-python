#!/usr/bin/env python3
"""Generate the Chapter 21 marketing and A/B test project data and figures.

Run from the project root:

    python scripts/generate_chapter21_sources.py

Fixtures under data/raw/project21/:

    kestrel_marketing.db   ad_daily, orders, touches, campaigns, experiment
    finance_assumptions.csv  planning constants (AOV, margin, FX, annual users)
    tracking_incidents.csv   planted tracking defects used in the reconciliation

Figures under images/ch21_*.png.  Everything is deterministic (SEED = 2109).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "data" / "raw" / "project21"
IMAGES = ROOT / "images"
SEED = 2109

CHANNELS = ["Search", "Social", "Display"]
CAMPAIGNS = {
    "Search": ["Search_Brand", "Search_Generic"],
    "Social": ["Social_Retarget", "Social_Prospect"],
    "Display": ["Display_AlwaysOn", "Display_Holiday"],
}
SHARES = {"Search": [0.55, 0.45], "Social": [0.50, 0.50],
          "Display": [0.70, 0.30]}
BASE_SPEND = {"Search": 3200.0, "Social": 2500.0, "Display": 1800.0}
EFFICIENCY = {"Search": 3.2, "Social": 2.2, "Display": 1.2}
ELASTICITY = {"Search": 0.75, "Social": 0.85, "Display": 0.60}
OVERCOUNT = {"Search": 0.04, "Social": 0.20, "Display": 0.50}
CTR = {"Search": 0.045, "Social": 0.012, "Display": 0.0035}

AOV = 185.0
FX_EUR_USD = 1.08
Q4_START = pd.Timestamp("2025-10-01")
Q1_START = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-03-31")
DATES = pd.date_range(Q4_START, END, freq="D")


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(PROJ / "kestrel_marketing.db")


def main() -> None:
    PROJ.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("generating Chapter 21 marketing project data ...")

    # ------------------------------------------------------------------ spend
    ads_rows = []
    for d in DATES:
        # 1.1964 yields a realised +18% quarter-on-quarter spend once the
        # lognormal daily noise is drawn; order volume is unaffected because
        # the response model normalises spend by this same multiplier.
        qmult = 1.0 if d < Q1_START else 1.1964
        for ch in CHANNELS:
            noise = np.exp(rng.normal(-0.5 * 0.15**2, 0.15))
            usd_total = BASE_SPEND[ch] * qmult * noise
            for k, camp in enumerate(CAMPAIGNS[ch]):
                part = usd_total * SHARES[ch][k]
                clicks = int(round(part / {"Search": 1.10, "Social": 0.65,
                                           "Display": 0.32}[ch]))
                impressions = int(round(clicks / CTR[ch]))
                eur = camp == "Display_Holiday"
                reported = part / FX_EUR_USD if eur else part
                ads_rows.append((
                    d.strftime("%Y-%m-%d"), ch, camp, impressions, clicks,
                    round(float(reported), 2), "EUR" if eur else "USD"))
    ads = pd.DataFrame(ads_rows, columns=[
        "date", "channel", "campaign", "impressions", "clicks",
        "spend", "currency"])
    ads["spend_usd"] = np.round(
        ads["spend"] * np.where(ads["currency"] == "EUR", FX_EUR_USD, 1.0), 2)

    # ----------------------------------------------------------------- orders
    # Paid order volume follows a saturating response to daily spend; the
    # quarter multiplier is chosen so revenue is roughly flat while spend rises.
    daily_paid_spend = ads.groupby(["date", "channel"])["spend_usd"].sum()
    lam_scale = {ch: BASE_SPEND[ch] * EFFICIENCY[ch] / AOV for ch in CHANNELS}
    order_rows, touch_rows = [], []
    oid = 0
    for d in DATES:
        ds = d.strftime("%Y-%m-%d")
        # 1.1964 yields a realised +18% quarter-on-quarter spend once the
        # lognormal daily noise is drawn; order volume is unaffected because
        # the response model normalises spend by this same multiplier.
        qmult = 1.0 if d < Q1_START else 1.1964
        qfactor = 1.0 if d < Q1_START else 92.0 / 90.0
        for ch in CHANNELS:
            s = float(daily_paid_spend.loc[(ds, ch)])
            lam = (lam_scale[ch] * qfactor
                   * (s / (BASE_SPEND[ch] * qmult)) ** ELASTICITY[ch])
            for _ in range(int(rng.poisson(lam))):
                oid += 1
                pay_channel = ch
                hour = int(rng.integers(0, 24))
                utc = d + pd.DateOffset(days=1) if hour >= 20 else d
                rev = float(np.exp(np.log(AOV) - 0.35**2 / 2
                                   + 0.35 * rng.normal()))
                rev = round(rev, 2)
                cid = f"C-{int(rng.integers(1, 9001)):05d}"
                order_rows.append((
                    f"O-{oid:06d}", ds, utc.strftime("%Y-%m-%d"), hour,
                    cid, pay_channel,
                    str(rng.choice(CAMPAIGNS[ch], p=SHARES[ch])),
                    rev, round(rev * {"Search": 0.36, "Social": 0.34,
                                      "Display": 0.30}[ch], 2), "USD"))
        # organic / direct orders get no paid credit
        for _ in range(int(rng.poisson(17.0 * qfactor))):
            oid += 1
            hour = int(rng.integers(0, 24))
            utc = d + pd.DateOffset(days=1) if hour >= 20 else d
            rev = round(float(np.exp(np.log(AOV) - 0.35**2 / 2
                                     + 0.35 * rng.normal())), 2)
            order_rows.append((
                f"O-{oid:06d}", ds, utc.strftime("%Y-%m-%d"), hour,
                f"C-{int(rng.integers(1, 9001)):05d}", "Organic", "Organic",
                rev, round(rev * 0.40, 2), "USD"))
    orders = pd.DataFrame(order_rows, columns=[
        "order_id", "order_date", "date_utc", "order_hour", "customer_id",
        "channel", "campaign", "revenue", "margin", "currency"])

    # touch paths: last touch is the order's credited channel; earlier touches
    # may be paid or organic, which is what makes attribution models disagree.
    for row in order_rows:
        oid_s, last = row[0], row[5]
        if last == "Organic":
            touches = ["Organic"] * int(rng.choice([1, 2], p=[0.7, 0.3]))
        else:
            k = int(rng.choice([1, 2, 3], p=[0.45, 0.35, 0.20]))
            earlier = [str(x) for x in rng.choice(
                CHANNELS + ["Organic"], size=k - 1,
                p=[0.30, 0.25, 0.18, 0.27])]
            touches = earlier + [last]
        for seq, ch in enumerate(touches, start=1):
            camp = ("Organic" if ch == "Organic"
                    else str(rng.choice(CAMPAIGNS[ch])))
            touch_rows.append((oid_s, seq, ch, camp))
    touches = pd.DataFrame(touch_rows, columns=[
        "order_id", "sequence", "channel", "campaign"])

    # -------------------------------------------------- platform conversions
    # Counted on the platform's UTC date, with channel-specific over-crediting.
    utc_counts = orders[orders["channel"].isin(CHANNELS)].groupby(
        ["date_utc", "channel"]).size()
    clicks_by_cc = ads.groupby(["date", "channel"])["clicks"].sum()
    conv_rows = []
    for d in DATES:
        ds = d.strftime("%Y-%m-%d")
        for ch in CHANNELS:
            base_n = int(utc_counts.get((ds, ch), 0))
            n = base_n + int(round(OVERCOUNT[ch] * base_n))
            if ch == "Social" and ds == "2026-02-14":
                n += 38  # duplicate pixel events documented in the incident log
            if ch == "Search" and ds in {"2026-03-03", "2026-03-04",
                                         "2026-03-05"}:
                n = -1  # tracking outage -> NULL on load
            click_share = ads[(ads["date"] == ds) & (ads["channel"] == ch)][
                "clicks"].to_numpy(dtype=float)
            share = click_share / click_share.sum()
            parts = np.floor(n * share).astype(int)
            parts[-1] += n - int(parts.sum())
            for k, camp in enumerate(CAMPAIGNS[ch]):
                a = ads[(ads["date"] == ds) & (ads["channel"] == ch)
                        & (ads["campaign"] == camp)].iloc[0]
                conv_rows.append((
                    ds, ch, camp, int(a["impressions"]), int(a["clicks"]),
                    float(a["spend"]), a["currency"],
                    int(parts[k]) if n >= 0 else -1))
    plat = pd.DataFrame(conv_rows, columns=[
        "date", "channel", "campaign", "impressions", "clicks",
        "spend", "currency", "platform_conversions"])
    plat.loc[plat["platform_conversions"] < 0, "platform_conversions"] = np.nan

    # ----------------------------------------------------------- campaign list
    camp_rows = []
    for ch in CHANNELS:
        for camp in CAMPAIGNS[ch]:
            camp_rows.append((camp, ch, Q4_START.strftime("%Y-%m-%d"),
                              END.strftime("%Y-%m-%d"),
                              "conversion" if ch != "Display" else "reach",
                              round(float(BASE_SPEND[ch] * 90 * 1.09
                                          * SHARES[ch][
                                              CAMPAIGNS[ch].index(camp)]), 2)))
    campaigns = pd.DataFrame(camp_rows, columns=[
        "campaign_id", "channel", "start_date", "end_date",
        "objective", "planned_budget"])

    # -------------------------------------------------------------- experiment
    n_user = 12000
    uid = np.arange(1, n_user + 1)
    variant = np.array(["control"] * (n_user // 2)
                       + ["treatment"] * (n_user // 2))
    rng.shuffle(variant)
    device = rng.choice(["mobile", "desktop"], n_user, p=[0.55, 0.45])
    user_type = rng.choice(["new", "returning"], n_user, p=[0.50, 0.50])
    day = rng.integers(1, 15, n_user)
    exposed = (rng.random(n_user) < 0.96).astype(int)
    base_rate = np.where(
        (device == "mobile") & (user_type == "new"), 0.118,
        np.where((device == "mobile") & (user_type == "returning"), 0.128,
                 np.where((device == "desktop") & (user_type == "new"),
                          0.130, 0.122)))
    lift = np.where((variant == "treatment")
                    & (device == "mobile") & (user_type == "new"), 0.026, 0.0)
    conv = (rng.random(n_user) < base_rate + lift).astype(int)
    revenue = np.where(
        conv == 1,
        np.round(np.exp(np.log(AOV) - 0.35**2 / 2
                        + 0.35 * rng.normal(size=n_user))), 0.0)
    refund_p = np.where(variant == "treatment", 0.037, 0.026)
    refund = (rng.random(n_user) < refund_p).astype(int)
    experiment = pd.DataFrame({
        "user_id": [f"U-{i:06d}" for i in uid],
        "assignment_date": [str((Q1_START + pd.DateOffset(days=int(d - 1)))
                                .date()) for d in day],
        "day": day,
        "variant": variant,
        "exposed": exposed,
        "converted": conv,
        "revenue": revenue,
        "refund": refund,
        "device": device,
        "user_type": user_type,
    })

    # ------------------------------------------------------------ write files
    with _conn() as con:
        ads.drop(columns=["spend_usd"]).to_sql(
            "ad_daily", con, index=False, if_exists="replace")
        plat.to_sql("platform_daily", con, index=False, if_exists="replace")
        orders.to_sql("orders", con, index=False, if_exists="replace")
        touches.to_sql("touches", con, index=False, if_exists="replace")
        campaigns.to_sql("campaigns", con, index=False, if_exists="replace")
        experiment.to_sql("experiment", con, index=False, if_exists="replace")

    fin = pd.DataFrame([
        ("aov", AOV), ("margin_rate", 0.35), ("refund_cost_share", 1.00),
        ("annual_exposed_users", 1000000), ("fx_eur_usd", FX_EUR_USD),
        ("timezone_hour_cutoff", 20), ("mde_pp", 2.0), ("alpha", 0.05),
        ("target_power", 0.80),
    ], columns=["key", "value"])
    fin.to_csv(PROJ / "finance_assumptions.csv", index=False)

    inc = pd.DataFrame([
        ("2026-02-14", "Social", "duplicate_events",
         "retarget pixel fired twice", 38),
        ("2026-03-03", "Search", "tracking_outage", "conversions missing", 0),
        ("2026-03-04", "Search", "tracking_outage", "conversions missing", 0),
        ("2026-03-05", "Search", "tracking_outage", "conversions missing", 0),
        ("2025-10-01", "Display", "currency",
         "Display_Holiday billed in EUR", FX_EUR_USD),
        ("2025-10-01", "All", "timezone",
         "platform UTC dates vs warehouse local dates", 20),
    ], columns=["date", "channel", "issue", "detail", "value"])
    inc.to_csv(PROJ / "tracking_incidents.csv", index=False)

    print(f"  ad_daily rows: {len(ads)}, orders: {len(orders)}, "
          f"touches: {len(touches)}, experiment: {len(experiment)}")
    print(f"  spend USD: {ads['spend_usd'].sum():,.2f}, "
          f"revenue: {orders['revenue'].sum():,.2f}")
    print(f"  paid orders: {orders['channel'].isin(CHANNELS).sum():,}, "
          f"organic: {orders['channel'].eq('Organic').sum():,}")
    figures(ads, orders, plat, experiment, touches)
    for fp in sorted(IMAGES.glob("ch21_*.png")):
        print(f"  {fp.name}: {fp.stat().st_size:,} bytes")
    print("done.")


def figures(ads, orders, plat, experiment, touches) -> None:
    colors = {"Search": "#2980b9", "Social": "#f39c12",
              "Display": "#c0392b"}

    # 1. daily spend and revenue
    s = ads.groupby("date")["spend_usd"].sum()
    r = orders.groupby("order_date")["revenue"].sum().reindex(s.index).fillna(0)
    f, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(s.index, s.values, lw=1.2, alpha=0.8, label="spend (USD)")
    ax.plot(r.index, r.values, lw=1.2, alpha=0.8, label="revenue (USD)")
    ax.axvline("2026-01-01", color="k", ls="--", lw=1)
    ax.set(title="Daily spend and revenue", ylabel="USD")
    ax.legend(fontsize=8)
    f.autofmt_xdate(); f.tight_layout()
    f.savefig(IMAGES / "ch21_spend_trend.png", dpi=130); plt.close(f)

    # 2. platform vs warehouse ROAS
    spend = ads.groupby("channel")["spend_usd"].sum()
    ware = orders[orders["channel"].isin(CHANNELS)].groupby(
        "channel")["revenue"].sum()
    platrev = plat.groupby("channel")["platform_conversions"].sum() * 185.0
    f, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(CHANNELS)); w = 0.35
    ax.bar(x - w / 2, [ware[c] / spend[c] for c in CHANNELS], w,
           label="warehouse", color="#2c3e50")
    ax.bar(x + w / 2, [platrev[c] / spend[c] for c in CHANNELS], w,
           label="platform", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels(CHANNELS)
    ax.set(title="ROAS: platform vs warehouse", ylabel="ROAS")
    ax.legend(fontsize=8)
    f.tight_layout(); f.savefig(IMAGES / "ch21_channel_roas.png", dpi=130)
    plt.close(f)

    # 3. reconciliation gap by day
    pc = plat.groupby(["date", "channel"])[
        "platform_conversions"].sum().unstack()
    oc = orders[orders["channel"].isin(CHANNELS)].groupby(
        ["date_utc", "channel"]).size().unstack().reindex(pc.index).fillna(0)
    gap = (pc - oc).rolling(7).mean()
    f, ax = plt.subplots(figsize=(9, 4.2))
    for c in CHANNELS:
        ax.plot(gap.index, gap[c], lw=1.2, label=c, color=colors[c])
    ax.axhline(0, color="k", lw=1)
    ax.set(title="7-day reconciliation gap (platform minus warehouse, UTC)",
           ylabel="conversions / day")
    ax.legend(fontsize=8)
    f.autofmt_xdate(); f.tight_layout()
    f.savefig(IMAGES / "ch21_reconciliation.png", dpi=130); plt.close(f)

    # 4. saturation curves
    f, ax = plt.subplots(figsize=(7, 4.2))
    grid = np.linspace(0.4, 1.6, 100)
    for ch in CHANNELS:
        base = BASE_SPEND[ch]
        y = EFFICIENCY[ch] * grid ** ELASTICITY[ch]
        ax.plot(base * grid, y * base / 1000, lw=2, label=ch, color=colors[ch])
    ax.set(title="Response curves (revenue per $1k spend)",
           xlabel="daily spend (USD)", ylabel="revenue ($000)")
    ax.legend(fontsize=8)
    f.tight_layout(); f.savefig(IMAGES / "ch21_saturation.png", dpi=130)
    plt.close(f)

    # 5. attribution comparison
    t = touches.merge(orders[["order_id", "revenue"]], on="order_id")
    mx = t.groupby("order_id")["sequence"].transform("max")
    paid = t["channel"].isin(CHANNELS)
    rows = {}
    for m, w in [("last_click", (t["sequence"] == mx).astype(float)),
                 ("first_click", (t["sequence"] == 1).astype(float)),
                 ("linear", 1.0 / mx)]:
        rows[m] = (t.assign(amt=w * t["revenue"] * paid)
                   .groupby("channel")["amt"].sum())
    comp = pd.DataFrame(rows).reindex(CHANNELS).fillna(0)
    f, ax = plt.subplots(figsize=(7.5, 4.2))
    comp.plot.bar(ax=ax, color={"last_click": "#2c3e50",
                                "first_click": "#2980b9",
                                "linear": "#f39c12"})
    ax.set(title="Attributed revenue by model", ylabel="USD")
    ax.tick_params(axis="x", rotation=0)
    f.tight_layout(); f.savefig(IMAGES / "ch21_attribution.png", dpi=130)
    plt.close(f)

    # 6. experiment conversion by arm
    g = experiment.groupby("variant")["converted"].agg(["sum", "count"])
    rates = (g["sum"] / g["count"]).to_numpy()
    se = np.sqrt(rates * (1 - rates) / g["count"].to_numpy())
    f, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(["control", "treatment"], rates * 100,
           yerr=1.96 * se * 100, capsize=6, color=["#7f8c8d", "#27ae60"])
    ax.set(title="Checkout conversion by arm", ylabel="conversion (%)")
    f.tight_layout(); f.savefig(IMAGES / "ch21_experiment.png", dpi=130)
    plt.close(f)

    # 7. guardrail movement
    g2 = experiment.groupby("variant")["refund"].mean()
    f, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(["control", "treatment"], g2.values * 100,
           color=["#7f8c8d", "#c0392b"])
    ax.set(title="Refund guardrail by arm", ylabel="refund rate (%)")
    f.tight_layout(); f.savefig(IMAGES / "ch21_guardrail.png", dpi=130)
    plt.close(f)

    # 8. four-panel dashboard
    f, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    axes[0, 0].plot(s.index, s.values, lw=1, color="#2980b9")
    axes[0, 0].set_title("Spend by day")
    for ch in CHANNELS:
        axes[0, 1].plot(gap.index, gap[ch], lw=1, label=ch, color=colors[ch])
    axes[0, 1].axhline(0, color="k", lw=1)
    axes[0, 1].set_title("Reconciliation gap")
    axes[0, 1].legend(fontsize=7)
    comp.plot.bar(ax=axes[1, 0], fontsize=7)
    axes[1, 0].set_title("Attribution models")
    axes[1, 1].bar(["control", "treatment"], rates * 100,
                   color=["#7f8c8d", "#27ae60"])
    axes[1, 1].set_title("Experiment conversion (%)")
    for a in axes.ravel():
        a.tick_params(labelsize=7)
    f.tight_layout(); f.savefig(IMAGES / "ch21_dashboard.png", dpi=130)
    plt.close(f)


if __name__ == "__main__":
    main()
