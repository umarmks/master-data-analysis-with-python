#!/usr/bin/env python3
"""Print every output block used in Chapter 21, from the current project data.

Run from the project root:

    python scripts/chapter21_outputs.py

The strings in PREAMBLE and BLOCKS are also used verbatim as the [source,python]
blocks in chapters/chapter21-Marketing.adoc, so the printed values are exactly
the values embedded in the chapter.
"""

from __future__ import annotations

PREAMBLE = """\
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJ = Path("data/raw/project21")
DB = PROJ / "kestrel_marketing.db"
FX = 1.08
CHANNELS = ["Search", "Social", "Display"]

def wilson(k, n):
    p = k / n
    z = stats.norm.ppf(0.975)
    den = 1 + z**2 / n
    ctr = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return ctr - half, ctr + half

def two_prop(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    return p1, p2, z, 2 * (1 - stats.norm.cdf(abs(z)))

def diff_ci(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = stats.norm.ppf(0.975)
    return p2 - p1 - z * se, p2 - p1 + z * se

def required_n(p1, lift, alpha=0.05, power=0.80):
    p2 = p1 + lift
    pbar = (p1 + p2) / 2
    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    num = (za * np.sqrt(2 * pbar * (1 - pbar))
           + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
    return int(np.ceil(num / lift**2))

def power_at(p1, lift, n, alpha=0.05):
    p2 = p1 + lift
    se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    zc = stats.norm.ppf(1 - alpha / 2)
    return stats.norm.cdf(lift / se - zc) + stats.norm.cdf(-lift / se - zc)

def bh(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out

con = sqlite3.connect(DB)
ads = pd.read_sql("SELECT * FROM ad_daily", con)
platdb = pd.read_sql("SELECT * FROM platform_daily", con)
orders = pd.read_sql("SELECT * FROM orders", con)
touches = pd.read_sql("SELECT * FROM touches", con)
campaigns = pd.read_sql("SELECT * FROM campaigns", con)
exp = pd.read_sql("SELECT * FROM experiment", con)
con.close()
fin = pd.read_csv(PROJ / "finance_assumptions.csv").set_index("key")["value"]
inc = pd.read_csv(PROJ / "tracking_incidents.csv")
ads["spend_usd"] = np.round(
    ads["spend"] * np.where(ads["currency"] == "EUR", FX, 1.0), 2)
ads["quarter"] = np.where(ads["date"] < "2026-01-01", "Q4 2025", "Q1 2026")
orders["quarter"] = np.where(
    orders["order_date"] < "2026-01-01", "Q4 2025", "Q1 2026")
paid = orders[orders["channel"].isin(CHANNELS)]
print({"ad_daily": len(ads), "orders": len(orders),
       "touches": len(touches), "experiment": len(exp)})
"""

BLOCKS = [
    ("Frame: two decisions", """\
print({
    "decision_1": "budget allocation across Search, Social, Display",
    "decision_2": "ship or hold the checkout redesign",
    "reason": "one is correlational, the other is randomised",
})
"""),
    ("Frame: metrics", """\
print({
    "primary_budget": "warehouse-attributed revenue per dollar of spend",
    "primary_test": "user-level checkout conversion",
    "guardrail_test": "refund rate",
    "secondary": ["CAC", "marginal CAC", "ROAS by attribution model"],
})
"""),
    ("Frame: comparison windows", """\
q = ads.groupby("quarter")["spend_usd"].sum()
r = orders.groupby("quarter")["revenue"].sum()
print({k: {"spend": round(float(q[k]), 2),
           "revenue": round(float(r[k]), 2)} for k in q.index})
print({"spend_growth": round(float(q["Q1 2026"] / q["Q4 2025"] - 1), 4),
       "revenue_growth": round(float(r["Q1 2026"] / r["Q4 2025"] - 1), 4),
       "days": {"Q4 2025": 92, "Q1 2026": 90}})
"""),
    ("Frame: attribution stance", """\
print({
    "reconciliation": "platform is a claim, the warehouse is the ledger",
    "primary_model": "last click",
    "sensitivity": ["first click", "linear"],
    "incrementality": "attribution is not causation without a holdout",
})
"""),
    ("Frame: pre-registered test plan", """\
print({
    "metric": "conversion", "mde_pp": float(fin["mde_pp"]),
    "alpha": float(fin["alpha"]), "power": float(fin["target_power"]),
    "planned_n_per_arm": int(required_n(0.124, float(fin["mde_pp"]) / 100)),
    "segments": ["mobile new", "mobile returning",
                 "desktop new", "desktop returning"],
    "guardrail": "refund rate", "looks": 1, "horizon_days": 14,
})
"""),
    ("Assess: table inventory", """\
print({"ad_daily": len(ads), "orders": len(orders), "touches": len(touches),
       "campaigns": len(campaigns), "experiment": len(exp),
       "date_min": ads["date"].min(), "date_max": ads["date"].max()})
print({"paid_orders": int(orders["channel"].isin(CHANNELS).sum()),
       "organic_orders": int(orders["channel"].eq("Organic").sum()),
       "duplicate_order_ids": int(orders["order_id"].duplicated().sum())})
"""),
    ("Assess: naive reconciliation", """\
pl = platdb.groupby(["date", "channel"])["platform_conversions"].sum().rename("platform")
wl = paid.groupby(["order_date", "channel"]).size().rename("warehouse")
wl.index = wl.index.set_names(["date", "channel"])
naive = pd.concat([pl, wl], axis=1).fillna(0)
g = naive.groupby(level="channel").sum()
g["gap"] = g["platform"] - g["warehouse"]
print(g[["platform", "warehouse", "gap"]].round(0).to_string())
print({k: round(float(v), 4) for k, v in
       (g["gap"] / g["warehouse"]).items()})
"""),
    ("Assess: UTC-aligned reconciliation", """\
wu = paid.groupby(["date_utc", "channel"]).size().rename("warehouse")
wu.index = wu.index.set_names(["date", "channel"])
aligned = pd.concat([pl, wu], axis=1).fillna(0)
g2 = aligned.groupby(level="channel").sum()
g2["gap"] = g2["platform"] - g2["warehouse"]
print(g2.round(0).to_string())
print({"naive_abs_gap": round(float(
           (naive["platform"] - naive["warehouse"]).abs().sum()), 0),
       "aligned_abs_gap": round(float(
           (aligned["platform"] - aligned["warehouse"]).abs().sum()), 0)})
"""),
    ("Assess: largest daily gaps", """\
gd = (aligned.assign(gap=aligned["platform"] - aligned["warehouse"])
      .groupby(level="date")["gap"].sum().abs().nlargest(5))
print(gd.round(0).to_string())
print({"note": "2026-02-14 is the duplicate day; 2026-03-03..05 is the Search outage"})
"""),
    ("Assess: timezone alignment", """\
late = orders["date_utc"].ne(orders["order_date"])
print({"late_orders": int(late.sum()), "share": round(float(late.mean()), 4),
       "cutoff_hour": int(fin["timezone_hour_cutoff"])})
print({"naive_local_gap": round(float(
           (naive["platform"] - naive["warehouse"]).abs().sum()), 0),
       "utc_aligned_gap": round(float(
           (aligned["platform"] - aligned["warehouse"]).abs().sum()), 0)})
"""),
    ("Assess: duplicate events", """\
ds = "2026-02-14"
row = aligned.loc[(ds, "Social")]
dup = int(inc.loc[inc["issue"] == "duplicate_events", "value"].iloc[0])
print({"date": ds, "platform": int(row["platform"]),
       "warehouse": int(row["warehouse"]),
       "gap": int(row["platform"] - row["warehouse"]),
       "documented_duplicate": dup})
print({"after_dedupe": int(row["platform"] - dup - row["warehouse"])})
"""),
    ("Assess: tracking outage", """\
out = platdb[platdb["platform_conversions"].isna()]
print({"outage_rows": len(out), "dates": out["date"].unique().tolist(),
       "channels": out["channel"].unique().tolist()})
print({"rule": "exclude NULL days from the conversion gap, keep their spend"})
"""),
    ("Assess: currency mismatch", """\
eur = ads[ads["currency"] == "EUR"]
print({"eur_rows": len(eur), "campaigns": eur["campaign"].unique().tolist(),
       "reported_sum": round(float(eur["spend"].sum()), 2),
       "usd_sum": round(float(eur["spend_usd"].sum()), 2),
       "fx": float(fin["fx_eur_usd"])})
print({"display_spend_reported": round(float(
           ads.loc[ads["channel"] == "Display", "spend"].sum()), 2),
       "display_spend_usd": round(float(
           ads.loc[ads["channel"] == "Display", "spend_usd"].sum()), 2)})
"""),
    ("Assess: discrepancy log", """\
print(pd.DataFrame([
    ("over-credit", "all", "platform exceeds warehouse on UTC dates", "use warehouse"),
    ("duplicate events", "Social", "38 on 2026-02-14", "subtract 38"),
    ("tracking outage", "Search", "3 NULL days in March", "exclude the days"),
    ("currency", "Display", "Holiday campaign billed in EUR", "multiply by 1.08"),
    ("timezone", "all", "17% of orders cross UTC midnight", "compare on UTC"),
], columns=["issue", "channel", "finding", "treatment"]).to_string(index=False))
"""),
    ("Prepare: daily spend table", """\
daily = ads.groupby(["date", "channel"], as_index=False).agg(
    spend=("spend_usd", "sum"), clicks=("clicks", "sum")).merge(
    platdb.groupby(["date", "channel"], as_index=False)[
        "platform_conversions"].sum(), on=["date", "channel"])
print({"rows": len(daily), "columns": daily.columns.tolist()})
print(daily.head(6).round(2).to_string(index=False))
"""),
    ("Prepare: attributed order table", """\
t = touches.merge(
    orders[["order_id", "revenue", "channel"]].rename(
        columns={"channel": "order_channel"}), on="order_id")
mx = t.groupby("order_id")["sequence"].transform("max")
ispaid = t["channel"].isin(CHANNELS)
t["w_last"] = (t["sequence"] == mx).astype(float) * ispaid
t["w_first"] = (t["sequence"] == 1).astype(float) * ispaid
t["w_linear"] = (1.0 / mx) * ispaid
for m in ["last", "first", "linear"]:
    t["rev_" + m] = t["w_" + m] * t["revenue"]
print({"touch_rows": len(t), "orders": int(t["order_id"].nunique()),
       "organic_touch_rows": int(t["channel"].eq("Organic").sum())})
print(t[["order_id", "sequence", "channel", "w_last",
         "w_linear"]].head(6).to_string(index=False))
"""),
    ("Prepare: spend and revenue by quarter", """\
sq = ads.groupby(["quarter", "channel"])["spend_usd"].sum().unstack()
oq = paid.groupby(["quarter", "channel"])["revenue"].sum().unstack()
print(sq.round(0).to_string())
print(oq.round(0).to_string())
print({"spend_growth": round(float(
           sq.loc["Q1 2026"].sum() / sq.loc["Q4 2025"].sum() - 1), 4),
       "revenue_growth": round(float(
           oq.loc["Q1 2026"].sum() / oq.loc["Q4 2025"].sum() - 1), 4)})
"""),
    ("Prepare: CAC and ROAS", """\
sc = ads.groupby("channel")["spend_usd"].sum()
cnt = t.groupby("channel")[["w_last", "w_first", "w_linear"]].sum().reindex(CHANNELS)
rev = t.groupby("channel")[["rev_last", "rev_first", "rev_linear"]].sum().reindex(CHANNELS)
econ = pd.DataFrame({"spend": sc, "orders": cnt["w_last"],
                     "revenue": rev["rev_last"]})
econ["CAC"] = econ["spend"] / econ["orders"]
econ["ROAS"] = econ["revenue"] / econ["spend"]
print(econ.round(2).to_string())
"""),
    ("Prepare: marginal CAC", """\
dsp = ads.groupby(["date", "channel"])["spend_usd"].sum()
drv = paid.groupby(["order_date", "channel"])["revenue"].sum()
dsp.index = dsp.index.set_names(["date", "channel"])
drv.index = drv.index.set_names(["date", "channel"])
fitdf = pd.concat([dsp.rename("spend"), drv.rename("rev")], axis=1).dropna()
beta = {}
for ch in CHANNELS:
    gg = fitdf.xs(ch, level="channel")
    beta[ch] = float(np.polyfit(np.log(gg["spend"]), np.log(gg["rev"]), 1)[0])
mc = pd.DataFrame({"elasticity": pd.Series(beta), "CAC": econ["CAC"]})
mc["marginal_CAC"] = mc["CAC"] / mc["elasticity"]
print(mc.round(3).to_string())
"""),
    ("Prepare: experiment dataset", """\
print({"users": len(exp), "control": int(exp["variant"].eq("control").sum()),
       "treatment": int(exp["variant"].eq("treatment").sum()),
       "days": int(exp["day"].nunique()),
       "exposed": round(float(exp["exposed"].mean()), 4),
       "conversion": round(float(exp["converted"].mean()), 4)})
"""),
    ("Prepare: assignment balance", """\
bal = exp.groupby("variant").agg(
    n=("user_id", "size"), exposed=("exposed", "mean"),
    mobile=("device", lambda s: s.eq("mobile").mean()),
    new=("user_type", lambda s: s.eq("new").mean()),
    day=("day", "mean"))
print(bal.round(4).to_string())
"""),
    ("Prepare: sample ratio mismatch", """\
nn = len(exp)
nt = int(exp["variant"].eq("treatment").sum())
zsr = (nt - nn * 0.5) / np.sqrt(nn * 0.25)
print({"treatment_share": round(nt / nn, 4), "expected": 0.5,
       "z": round(float(zsr), 3),
       "p": round(float(2 * (1 - stats.norm.cdf(abs(zsr)))), 4)})
"""),
    ("Explore: monthly spend and revenue", """\
ms = ads.assign(month=ads["date"].str[:7]).groupby("month")["spend_usd"].sum()
mr = (orders.assign(month=orders["order_date"].str[:7])
      .groupby("month")["revenue"].sum())
print(pd.DataFrame({"spend": ms.round(0), "revenue": mr.round(0),
                    "roas": (mr / ms).round(3)}).to_string())
"""),
    ("Explore: channel efficiency over time", """\
msc = (ads.assign(month=ads["date"].str[:7])
       .groupby(["month", "channel"])["spend_usd"].sum())
mrc = (paid.assign(month=paid["order_date"].str[:7])
       .groupby(["month", "channel"])["revenue"].sum())
print((mrc / msc).unstack().round(3).to_string())
"""),
    ("Explore: saturation curves", """\
alpha_beta = {}
for ch in CHANNELS:
    gg = fitdf.xs(ch, level="channel")
    bb, aa = np.polyfit(np.log(gg["spend"]), np.log(gg["rev"]), 1)
    alpha_beta[ch] = (float(np.exp(aa)), float(bb))
sat = pd.DataFrame({
    ch: {"alpha": v[0], "beta": v[1],
         "mROAS": v[1] * float(econ.loc[ch, "ROAS"])}
    for ch, v in alpha_beta.items()}).T
print(sat.round(3).to_string())
"""),
    ("Explore: campaign performance", """\
cs = ads.groupby("campaign")["spend_usd"].sum()
cr = paid.groupby("campaign")["revenue"].sum()
cp = pd.DataFrame({"spend": cs, "revenue": cr}).dropna()
cp["ROAS"] = cp["revenue"] / cp["spend"]
print(cp.round(2).to_string())
"""),
    ("Explore: experiment overall", """\
ge = exp.groupby("variant")["converted"].agg(["sum", "count"])
p1, p2, zc, pval = two_prop(
    ge.loc["control", "sum"], ge.loc["control", "count"],
    ge.loc["treatment", "sum"], ge.loc["treatment", "count"])
lo, hi = diff_ci(ge.loc["control", "sum"], ge.loc["control", "count"],
                 ge.loc["treatment", "sum"], ge.loc["treatment", "count"])
print({"control": round(float(p1), 4), "treatment": round(float(p2), 4),
       "lift_pp": round(float((p2 - p1) * 100), 3),
       "ci_pp": [round(float(lo * 100), 3), round(float(hi * 100), 3)],
       "z": round(float(zc), 3), "p": round(float(pval), 4)})
"""),
    ("Explore: experiment segments", """\
rows = []
for (dv, ut), gg in exp.groupby(["device", "user_type"]):
    gs = gg.groupby("variant")["converted"].agg(["sum", "count"])
    a, b, zz, pp = two_prop(
        gs.loc["control", "sum"], gs.loc["control", "count"],
        gs.loc["treatment", "sum"], gs.loc["treatment", "count"])
    rows.append((f"{dv} {ut}", a, b, (b - a) * 100, pp))
seg = pd.DataFrame(rows, columns=["segment", "control", "treatment",
                                  "lift_pp", "p"])
print(seg.round(4).to_string(index=False))
"""),
    ("Explore: guardrail movement", """\
gr = exp.groupby("variant")["refund"].agg(["sum", "count"])
r1, r2, zg, pg = two_prop(
    gr.loc["control", "sum"], gr.loc["control", "count"],
    gr.loc["treatment", "sum"], gr.loc["treatment", "count"])
print({"control": round(float(r1), 4), "treatment": round(float(r2), 4),
       "diff_pp": round(float((r2 - r1) * 100), 3),
       "z": round(float(zg), 3), "p": round(float(pg), 4)})
"""),
    ("Analyze: attribution comparison", """\
R = rev.divide(sc, axis=0)
print(R.round(3).to_string())
print({"model_totals": {m: round(float(rev["rev_" + m].sum()), 2)
                        for m in ["last", "first", "linear"]}})
"""),
    ("Analyze: double-count check", """\
print({"single_model_total": round(float(rev["rev_last"].sum()), 2),
       "sum_of_three_models": round(float(rev.sum().sum()), 2),
       "paid_order_revenue": round(float(paid["revenue"].sum()), 2)})
"""),
    ("Analyze: incrementality bounds", """\
org = orders[orders["channel"] == "Organic"]
print({"organic_share": round(float(len(org) / len(orders)), 4),
       "organic_revenue_share": round(float(
           org["revenue"].sum() / orders["revenue"].sum()), 4)})
oc = ((g2["platform"] - g2["warehouse"]) / g2["warehouse"]).dropna()
print({"platform_overcount_share": {k: round(float(v), 4)
                                   for k, v in oc.items()}})
print({"note": "platform ROAS is an upper bound; no holdout, no causal lift"})
"""),
    ("Analyze: CAC and payback", """\
profit = float(fin["aov"]) * float(fin["margin_rate"])
pb = pd.DataFrame({"CAC": econ["CAC"], "ROAS": econ["ROAS"]})
pb["profit_per_order"] = round(profit, 2)
pb["payback_orders"] = pb["CAC"] / profit
pb["payback_days"] = pb["payback_orders"] / (cnt["w_last"] / 90)
print(pb.round(2).to_string())
"""),
    ("Analyze: marginal return", """\
sq1 = ads[ads["quarter"] == "Q1 2026"].groupby("channel")["spend_usd"].sum() / 90
delta_daily = 48000.0 / 90
gain_daily = 0.0
for ch in ["Search", "Social"]:
    aa, bb = alpha_beta[ch]
    ss = float(sq1[ch])
    gain_daily += aa * ((ss + delta_daily / 2)**bb - ss**bb)
aa, bb = alpha_beta["Display"]
ss = float(sq1["Display"])
gain_daily -= aa * ((ss - delta_daily)**bb - ss**bb)
print({"quarterly_shift": 48000.0,
       "new_daily_spend": {k: round(float(v + delta_daily / (2 if k != "Display" else -1)), 1)
                           for k, v in sq1.items()},
       "predicted_quarterly_gain": round(float(gain_daily * 90), 2)})
"""),
    ("Analyze: reallocation plan", """\
new_daily = {ch: float(sq1[ch]) for ch in CHANNELS}
new_daily["Search"] += delta_daily / 2
new_daily["Social"] += delta_daily / 2
new_daily["Display"] -= delta_daily
plan = pd.DataFrame([
    (ch, round(float(sq1[ch]), 0), round(new_daily[ch], 0),
     "increase" if new_daily[ch] > float(sq1[ch])
     else ("decrease" if new_daily[ch] < float(sq1[ch]) else "hold"))
    for ch in ["Search", "Social", "Display"]
], columns=["channel", "current_daily", "new_daily", "action"])
print(plan.to_string(index=False))
print({"budget_neutral": True,
       "from": "Display", "to": ["Search", "Social"],
       "expected_quarterly_gain": round(float(gain_daily * 90), 2)})
"""),
    ("Analyze: experiment effect and interval", """\
print({"absolute_lift_pp": round(float((p2 - p1) * 100), 3),
       "relative_lift_pct": round(float((p2 / p1 - 1) * 100), 2),
       "ci_pp": [round(float(lo * 100), 3), round(float(hi * 100), 3)],
       "control_wilson": tuple(round(float(x), 4)
                              for x in wilson(ge.loc["control", "sum"],
                                              ge.loc["control", "count"])),
       "treatment_wilson": tuple(round(float(x), 4)
                                 for x in wilson(ge.loc["treatment", "sum"],
                                                 ge.loc["treatment", "count"]))})
"""),
    ("Analyze: power", """\
mde = float(fin["mde_pp"]) / 100
print({"planned_n_per_arm": int(required_n(p1, mde)),
       "observed_n_per_arm": 6000,
       "power_at_mde": round(float(power_at(p1, mde, 6000)), 4),
       "power_at_observed": round(float(power_at(p1, p2 - p1, 6000)), 4)})
"""),
    ("Analyze: segment multiplicity", """\
seg["q"] = bh(seg["p"].to_numpy())
print(seg.round(4).to_string(index=False))
print({"raw_hits": int((seg["p"] < 0.05).sum()),
       "bh_hits": int((seg["q"] < 0.05).sum())})
"""),
    ("Analyze: ship/no-ship expected value", """\
aov = float(fin["aov"])
extra_conv = (p2 - p1) * 6000
extra_ref = (r2 - r1) * 6000
gain = extra_conv * aov * float(fin["margin_rate"])
loss = extra_ref * aov * float(fin["refund_cost_share"])
print({"extra_conversions_per_arm": round(float(extra_conv), 1),
       "extra_refunds_per_arm": round(float(extra_ref), 1),
       "gain_per_arm": round(float(gain), 2),
       "guardrail_cost_per_arm": round(float(loss), 2),
       "net_per_arm": round(float(gain - loss), 2),
       "annual_net_at_1m_users": round(float(
           (gain - loss) * (float(fin["annual_exposed_users"]) / 6000)), 2)})
print({"decision": "hold",
       "reason": "guardrail cost exceeds the conversion gain"})
"""),
    ("Validate: attribution sensitivity", """\
print(R.round(3).to_string())
print({"best_by_model": {m: str(R["rev_" + m].idxmax())
                         for m in ["last", "first", "linear"]},
       "display_lowest_in_all": bool(all(
           R["rev_" + m].idxmin() == "Display"
           for m in ["last", "first", "linear"]))})
"""),
    ("Validate: reconciliation-rule sensitivity", """\
plat_raw = platdb.groupby("channel")["platform_conversions"].sum()
dup = float(inc.loc[inc["issue"] == "duplicate_events", "value"].iloc[0])
plat_clean = plat_raw.copy()
plat_clean["Social"] = plat_clean["Social"] - dup
sp_tot = sc.sum()
print({"raw_platform_roas": round(float(plat_raw.sum() * aov / sp_tot), 3),
       "deduped_platform_roas": round(float(plat_clean.sum() * aov / sp_tot), 3),
       "warehouse_roas": round(float(paid["revenue"].sum() / sp_tot), 3)})
disp_rev = float(paid.loc[paid["channel"] == "Display", "revenue"].sum())
print({"display_roas_reported_spend": round(float(
           disp_rev / ads.loc[ads["channel"] == "Display", "spend"].sum()), 3),
       "display_roas_usd_spend": round(float(
           disp_rev / sc["Display"]), 3)})
"""),
    ("Validate: conservative and optimistic curves", """\
def realloc_gain(shift):
    total = 0.0
    for ch in ["Search", "Social"]:
        aa, bb = alpha_beta[ch]
        ss = float(sq1[ch])
        b2 = max(0.05, bb + shift)
        a2 = aa * ss**(bb - b2)
        total += a2 * ((ss + delta_daily / 2)**b2 - ss**b2)
    aa, bb = alpha_beta["Display"]
    ss = float(sq1["Display"])
    b2 = max(0.05, bb - shift)
    a2 = aa * ss**(bb - b2)
    total -= a2 * ((ss - delta_daily)**b2 - ss**b2)
    return total * 90

print({name: round(float(realloc_gain(shift)), 2)
       for name, shift in [("conservative", -0.10),
                           ("base", 0.0), ("optimistic", 0.10)]})
print({"decision_holds": bool(realloc_gain(-0.10) > 0)})
"""),
    ("Validate: experiment stability", """\
lifts, pvals = [], []
for dd in sorted(exp["day"].unique()):
    sub = exp[exp["day"] != dd]
    gs = sub.groupby("variant")["converted"].agg(["sum", "count"])
    a, b, zz, pp = two_prop(
        gs.loc["control", "sum"], gs.loc["control", "count"],
        gs.loc["treatment", "sum"], gs.loc["treatment", "count"])
    lifts.append((b - a) * 100)
    pvals.append(pp)
print({"leave_one_day_out": len(lifts),
       "lift_min_pp": round(float(min(lifts)), 3),
       "lift_max_pp": round(float(max(lifts)), 3),
       "p_min": round(float(min(pvals)), 4),
       "p_max": round(float(max(pvals)), 4)})
print({"segment_hits_after_bh": int((seg["q"] < 0.05).sum()),
       "largest_raw_segment": str(seg.loc[seg["p"].idxmin(), "segment"])})
"""),
    ("Validate: assumptions and falsifiers", """\
print([
    "attribution is not incrementality without a holdout",
    "response curves are correlational, not a geo experiment",
    "the test is ITT; 4% of assigned users were not exposed",
    "a single interim look was pre-registered and not taken early",
])
print({"falsifier": "a guardrail that is flat or improved",
       "result": "not triggered: refunds rose 0.70 pp"})
"""),
    ("Deliver: reconciled dataset", """\
rec = aligned.copy()
rec["gap"] = rec["platform"] - rec["warehouse"]
print({"rows": len(rec), "columns": rec.columns.tolist()})
print(rec.head(6).round(1).to_string())
"""),
    ("Deliver: test decision memo", """\
print([
    "DECISION: hold the checkout redesign.",
    f"Conversion rose {(p2 - p1) * 100:.2f} pp (p={pval:.3f}), below the 2 pp MDE.",
    f"Refunds rose {(r2 - r1) * 100:.2f} pp (p={pg:.3f}).",
    f"No segment survives BH across {len(seg)} tests (min q={seg['q'].min():.3f}).",
])
print({"owner": "checkout lead", "next_step": "fix refund flow, re-run with 9k per arm"})
"""),
    ("Deliver: budget recommendation", """\
print({"from": "Display", "quarterly_shift": 48000.0,
       "to": {"Search": 24000.0, "Social": 24000.0},
       "expected_quarterly_gain": round(float(gain_daily * 90), 2),
       "owner": "growth lead",
       "guardrail": "judge on warehouse ROAS, not platform ROAS"})
print(pd.DataFrame([
    ("conservative", round(float(realloc_gain(-0.10)), 2)),
    ("base", round(float(realloc_gain(0.0)), 2)),
    ("optimistic", round(float(realloc_gain(0.10)), 2)),
], columns=["scenario", "quarterly_gain_usd"]).to_string(index=False))
"""),
    ("Deliver: dashboard views", """\
print({"channel_view": ["daily spend", "reconciliation gap",
                        "ROAS by attribution model", "response curves"],
       "experiment_view": ["conversion by arm", "refund guardrail",
                           "segment q-values", "14-day stability"]})
print({"figure": "images/ch21_dashboard.png",
       "note": "one page, two decisions, one metric each"})
"""),
    ("Deliver: handover", """\
print(["reconciled daily table", "attributed order table",
       "test decision memo", "budget scenario table", "dashboard"])
print(["python scripts/generate_chapter21_sources.py",
       "python scripts/chapter21_outputs.py"])
"""),
    ("Rubric and common failures", """\
print({"reconciliation": 20, "experimental": 25, "business": 25,
       "validation": 15, "communication": 15})
print([
    "summing attributed conversions across channels",
    "trusting platform ROAS",
    "peeking at the experiment daily",
    "ignoring the guardrail metric",
    "a budget shift from a spend-revenue correlation",
])
print(["marketing-mix regression", "geo holdout",
       "next-quarter forecast under two budgets"])
"""),
    ("Control totals and final decisions", """\
print({"ad_daily": len(ads), "orders": len(orders),
       "touches": len(touches), "experiment": len(exp)})
print({"budget_decision": "shift $48k from Display to Search and Social",
       "test_decision": "hold"})
print(["platform numbers are claims, not ledger entries",
       "a significant primary does not override a harmed guardrail",
       "attribution is not incrementality"])
"""),
]


def main() -> None:
    import io
    import contextlib

    ns = dict(globals())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(PREAMBLE, "<preamble>", "exec"), ns)
    print("### 00 Preamble")
    print(buf.getvalue(), end="")
    print()
    for i, (label, code) in enumerate(BLOCKS, start=1):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(code, f"<block {i:02d}>", "exec"), ns)
        print(f"### {i:02d} {label}")
        print(buf.getvalue(), end="")
        print()


if __name__ == "__main__":
    main()
