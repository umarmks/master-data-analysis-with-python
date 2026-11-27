#!/usr/bin/env python3
"""Generate the Chapter 20 churn project data and figures.

Run from the project root:

    python scripts/generate_chapter20_sources.py

Outputs (fixtures under data/raw/project20/, figures under images/):

    kestrel_membership.db   members, orders, tickets, sessions, email
    cohort_activity.csv    cohort x months-since-signup retention triangle
    fixed_horizon.csv      two snapshots with 90-day churn labels and leaks
    exit_survey.csv        themed exit responses from a sample of leavers
    finance_margin.csv     margin per member per month by segment

Everything is deterministic.
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
PROJ = ROOT / "data" / "raw" / "project20"
IMAGES = ROOT / "images"
SEED = 2007
CUTOFF = pd.Timestamp("2026-01-01")

PLANS = ["Basic", "Plus", "Premium"]
CHANNELS = ["Organic", "Paid", "Referral"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    PROJ.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("generating Chapter 20 churn project data ...")

    n = 12000
    signup = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 1095, n), unit="D")
    plan = rng.choice(PLANS, n, p=[0.50, 0.32, 0.18])
    channel = rng.choice(CHANNELS, n, p=[0.50, 0.30, 0.20])
    payment = rng.choice(["Card", "Bank", "Invoice"], n, p=[0.65, 0.25, 0.10])
    engagement = np.round(rng.beta(2.2, 3.0, n), 3)
    ticket_rate = rng.gamma(2.0, 0.35, n)
    order_rate = rng.gamma(3.0, 0.7, n) * np.where(plan == "Premium", 1.4, 1.0)

    hazard = (
        0.011
        * np.where(plan == "Basic", 1.45, np.where(plan == "Plus", 1.0, 0.65))
        * np.where(channel == "Paid", 1.25, np.where(channel == "Referral", 0.80, 1.0))
        * np.where(payment == "Invoice", 1.30, 1.0)
        * np.exp(-1.6 * engagement + 0.22 * ticket_rate)
    )
    duration = np.minimum(rng.exponential(1 / hazard), 60.0)
    end = signup + pd.to_timedelta(np.round(duration * 30.44).astype(int), unit="D")
    churned = end <= CUTOFF
    end_clean = end.where(churned, pd.NaT)
    involuntary = churned & (rng.random(n) < 0.14)
    reason = np.where(involuntary, "payment_failed",
                      np.where(churned,
                               rng.choice(["price", "service", "moved", "competitor"],
                                          n, p=[0.30, 0.30, 0.20, 0.20]), "none"))

    members = pd.DataFrame({
        "member_id": [f"M-{i + 1:05d}" for i in range(n)],
        "signup_date": signup.strftime("%Y-%m-%d"),
        "end_date": end_clean.strftime("%Y-%m-%d"),
        "plan": plan, "acquisition_channel": channel,
        "payment_method": payment,
        "churned": churned.astype(int),
        "churn_type": np.where(involuntary, "involuntary",
                               np.where(churned, "voluntary", "active")),
        "cancellation_reason": reason,
        "engagement_index": engagement,
    })
    con = _conn()
    members.to_sql("members", con, index=False, if_exists="replace")
    con.close()

    # event tables, limited to the observation window
    order_rows, ticket_rows, session_rows, email_rows = [], [], [], []
    for i in range(n):
        sid = signup[i]
        eid = end[i] if churned[i] else CUTOFF
        months = pd.date_range(sid, min(eid, CUTOFF), freq="MS")
        for m in months:
            k = rng.poisson(order_rate[i] / 4)
            for _ in range(k):
                amt = round(float(rng.lognormal(4.4, 0.35)), 2)
                order_rows.append((f"O-{len(order_rows)+1:07d}",
                                   members["member_id"][i],
                                   m.strftime("%Y-%m-%d"), amt,
                                   round(amt * float(rng.uniform(0.25, 0.40)), 2)))
            if rng.random() < ticket_rate[i] / 12:
                ticket_rows.append((f"T-{len(ticket_rows)+1:07d}",
                                    members["member_id"][i],
                                    m.strftime("%Y-%m-%d"),
                                    str(rng.choice(["billing", "product", "delivery"]))))
            s = rng.poisson(4 * engagement[i])
            for _ in range(s):
                session_rows.append((members["member_id"][i],
                                     m.strftime("%Y-%m-%d"),
                                     str(rng.choice(["web", "app"]))))
            opened = rng.binomial(4, 0.25 + 0.5 * engagement[i])
            email_rows.append((members["member_id"][i],
                               m.strftime("%Y-%m-%d"), opened))

    con = _conn()
    pd.DataFrame(order_rows, columns=[
        "order_id", "member_id", "month", "amount", "margin"]).to_sql(
        "orders", con, index=False, if_exists="replace")
    pd.DataFrame(ticket_rows, columns=[
        "ticket_id", "member_id", "month", "topic"]).to_sql(
        "tickets", con, index=False, if_exists="replace")
    pd.DataFrame(session_rows, columns=[
        "member_id", "month", "platform"]).to_sql(
        "sessions", con, index=False, if_exists="replace")
    pd.DataFrame(email_rows, columns=[
        "member_id", "month", "opens"]).to_sql(
        "email", con, index=False, if_exists="replace")
    con.close()

    # cohort retention triangle aligned by tenure
    coh = members.assign(cohort=members["signup_date"].str[:7]).copy()
    coh["tenure"] = ((pd.to_datetime(coh["end_date"]).fillna(CUTOFF)
                      - pd.to_datetime(coh["signup_date"])).dt.days / 30.44).astype(int)
    rows = []
    for c, g in coh.groupby("cohort"):
        size = len(g)
        for k in range(13):
            active = int((g["tenure"] >= k).sum())
            rows.append((c, size, k, active))
    cad = pd.DataFrame(rows, columns=[
        "cohort", "cohort_size", "months_since_signup", "active"])
    cad.to_csv(PROJ / "cohort_activity.csv", index=False)

    # per-member lookups make the feature build linear rather than quadratic
    om = {}
    for row in order_rows:
        om.setdefault(row[1], []).append((row[2], row[3], row[4]))
    tm = {}
    for row in ticket_rows:
        tm.setdefault(row[1], []).append(row[2])
    sm = {}
    for row in session_rows:
        sm.setdefault(row[1], []).append(row[2])
    em = {}
    for row in email_rows:
        em.setdefault(row[0], []).append((row[1], row[2]))

    # fixed-horizon tables: features as of snapshot, label = churn within 90 days
    frames = []
    for snap in [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-01")]:
        snap_s = snap.strftime("%Y-%m-%d")
        horizon = snap + pd.Timedelta(days=90)
        eligible = signup <= snap
        rows = []
        idx = np.where(eligible)[0]
        for i in idx:
            active_at_snap = (end[i] > snap) or (pd.isna(end[i]))
            if not active_at_snap:
                continue
            months = pd.date_range(snap - pd.DateOffset(months=6),
                                   snap - pd.DateOffset(days=1), freq="MS")
            mset = {m.strftime("%Y-%m-%d") for m in months}
            mid = members["member_id"][i]
            sub_o = [o for o in om.get(mid, []) if o[0] in mset]
            sub_t = [m_ for m_ in tm.get(mid, []) if m_ in mset]
            sub_s = [m_ for m_ in sm.get(mid, []) if m_ in mset]
            sub_e = [e for e in em.get(mid, []) if e[0] in mset]
            label = int((not pd.isna(end[i])) and end[i] <= horizon)
            rows.append((
                members["member_id"][i], snap_s,
                str(plan[i]), str(channel[i]),
                int((snap - signup[i]).days / 30.44),
                round(float(plan[i] == "Premium"), 2),
                round(float(plan[i] == "Plus"), 2),
                round(float(channel[i] == "Paid"), 2),
                round(float(payment[i] == "Invoice"), 2),
                len(sub_o), round(float(sum(x[1] for x in sub_o)), 2),
                round(float(sum(x[2] for x in sub_o)), 2),
                len(sub_t), len(sub_s),
                round(float(np.mean([x[1] for x in sub_e]))
                      if sub_e else 0.0, 3),
                label,
                str(reason[i]),
                int((CUTOFF - end[i]).days) if label else -1,
                round(float(sum(x[2] for x in sub_o)) if label else 0.0, 2),
            ))
        frames.append(pd.DataFrame(rows, columns=[
            "member_id", "snapshot", "plan", "acquisition_channel",
            "tenure_months", "is_premium", "is_plus",
            "is_paid_channel", "is_invoice", "orders_6m", "revenue_6m",
            "margin_6m", "tickets_6m", "sessions_6m", "email_open_rate",
            "churn_90d", "cancellation_reason",
            "days_after_end", "final_refund"]))
    fh = pd.concat(frames, ignore_index=True)
    fh.to_csv(PROJ / "fixed_horizon.csv", index=False)

    # exit survey themes from a sample of leavers
    leavers = members[members["churned"] == 1].sample(
        min(400, int(members["churned"].sum())), random_state=SEED)
    themes = ["price", "service", "missing_feature", "competitor", "moved"]
    surv = pd.DataFrame({
        "member_id": leavers["member_id"],
        "theme": rng.choice(themes, len(leavers),
                            p=[0.28, 0.30, 0.17, 0.15, 0.10]),
        "score": rng.integers(1, 6, len(leavers)),
        "date": leavers["end_date"],
    })
    surv.to_csv(PROJ / "exit_survey.csv", index=False)

    # margin per member-month by plan for CLV
    fin = members.groupby("plan").agg(
        members=("member_id", "size"),
        margin=("engagement_index", "mean")).reset_index()
    fin["margin_per_month"] = np.round(24 + 18 * fin["margin"], 2)
    fin["observed_churn"] = members.groupby("plan")["churned"].mean().values
    fin.to_csv(PROJ / "finance_margin.csv", index=False)

    print(f"  members: {len(members)}, churned {int(members['churned'].sum())} "
          f"({members['churned'].mean():.4f})")
    print(f"  fixed_horizon rows: {len(fh)}")
    figures(cad, fh, surv, fin)
    for fp in sorted(IMAGES.glob("ch20_*.png")):
        print(f"  {fp.name}: {fp.stat().st_size:,} bytes")
    print("done.")


def _conn():
    return sqlite3.connect(PROJ / "kestrel_membership.db")


def figures(cad, fh, surv, fin) -> None:
    # 1. retention triangle
    tri = cad.pivot(index="cohort", columns="months_since_signup",
                    values="active").div(cad.groupby("cohort")[
                        "cohort_size"].first(), axis=0)
    f, ax = plt.subplots(figsize=(8.5, 5.0))
    for _, row in tri.head(8).iterrows():
        ax.plot(row.dropna().index, row.dropna().values, marker="o", lw=1)
    ax.set(title="Cohort retention by tenure", xlabel="Months since signup",
           ylabel="Retained share")
    ax.set_ylim(0, 1.05)
    f.tight_layout(); f.savefig(IMAGES / "ch20_triangle.png", dpi=130); plt.close(f)

    # 2. survival-style curve
    mean_ret = tri.mean()
    f, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(mean_ret.index, mean_ret.values, lw=2, color="#c0392b")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set(title="Average retention curve", ylabel="Retained share")
    f.tight_layout(); f.savefig(IMAGES / "ch20_survival.png", dpi=130); plt.close(f)

    # 3. churn by tenure bucket
    d = fh[fh["snapshot"] == "2025-01-01"]
    tb = pd.cut(d["tenure_months"], [-1, 6, 12, 24, 60],
                labels=["0-6", "7-12", "13-24", "25+"])
    rate = d.groupby(tb, observed=True)["churn_90d"].mean()
    f, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar([str(x) for x in rate.index], rate.values, color="#8e44ad")
    ax.set(title="90-day churn by tenure", ylabel="churn rate")
    f.tight_layout(); f.savefig(IMAGES / "ch20_tenure.png", dpi=130); plt.close(f)

    # 4. churn by plan and channel
    f, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    d.groupby("plan")["churn_90d"].mean().plot.bar(ax=axes[0], color="#2980b9")
    axes[0].set(title="Churn by plan")
    d.groupby("is_paid_channel")["churn_90d"].mean().plot.bar(
        ax=axes[1], color="#f39c12")
    axes[1].set(title="Churn by paid channel")
    f.tight_layout(); f.savefig(IMAGES / "ch20_segments.png", dpi=130); plt.close(f)

    # 5. engagement distribution
    f, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(d["email_open_rate"], bins=30, color="#27ae60", alpha=0.8)
    ax.set(title="Email engagement", xlabel="open rate")
    f.tight_layout(); f.savefig(IMAGES / "ch20_engagement.png", dpi=130); plt.close(f)

    # 6. exit themes
    f, ax = plt.subplots(figsize=(7.0, 4.0))
    surv["theme"].value_counts().plot.bar(ax=ax, color="#c0392b")
    ax.set(title="Exit survey themes")
    f.tight_layout(); f.savefig(IMAGES / "ch20_themes.png", dpi=130); plt.close(f)

    # 7. calendar vs cohort effects
    cal = cad.groupby("cohort")["active"].sum()
    f, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.plot(cal.index, cal.values, marker="o", lw=2)
    ax.set(title="Active members by cohort", ylabel="active")
    ax.tick_params(axis="x", rotation=45)
    f.tight_layout(); f.savefig(IMAGES / "ch20_calendar.png", dpi=130); plt.close(f)

    # 8. CLV by plan
    f, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(fin["plan"], fin["margin_per_month"], color="#2c3e50")
    ax.set(title="Margin per month by plan", ylabel="$")
    f.tight_layout(); f.savefig(IMAGES / "ch20_clv.png", dpi=130); plt.close(f)


if __name__ == "__main__":
    main()
