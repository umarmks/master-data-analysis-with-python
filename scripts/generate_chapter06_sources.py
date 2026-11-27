#!/usr/bin/env python3
"""Generate the cleaning fixtures that Chapter 6 repairs.

Run from the project root:

    python scripts/generate_chapter06_sources.py

Outputs (all under data/raw/audit/):

    orders_messy_40k.csv     40,000 order lines with 14 planted defect types
    customers_messy.csv      ~2,300 messy customer records for household merge
    households_truth.csv     record_id -> true household (for scoring the merge)
    income_churn.csv         3,000 customers; income ~12% missing by segment
    fx_rates.csv             weekly FX rates to USD (as-of join)
    eu_orders_fx.csv         600 multi-currency orders (as-of join)

Everything is deterministic. The orders fixture is derived from a clean sample
of kestrel_orders.csv, so every defect in it is planted and countable.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
AUDIT = RAW / "audit"

SEED = 606

COLUMNS = [
    "order_id", "order_date", "customer_id", "channel", "region", "category",
    "product_id", "quantity", "unit_price", "discount_pct", "shipping_cost",
    "returned", "revenue", "store_id",
]

REGIONS = ["Midwest", "Northeast", "South", "West"]
CATEGORIES = ["Camping", "Footwear", "Apparel", "Accessories"]


# --------------------------------------------------------------- orders fixture
def _clean_base(rng: np.random.Generator) -> pd.DataFrame:
    """A sample of rows that carry no defects, so anything wrong is planted."""
    src = pd.read_csv(RAW / "kestrel_orders.csv", dtype=str, keep_default_na=False)
    src["discount_pct"] = src["discount_pct"].replace("", "0.0")
    ok = (
        src["order_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}")
        & pd.to_datetime(src["order_date"], errors="coerce", format="%Y-%m-%d").notna()
        & src["customer_id"].str.fullmatch(r"C-\d{6}")
        & src["region"].isin(REGIONS)
        & src["category"].isin(CATEGORIES)
        & src["product_id"].str.fullmatch(r"P-\d{4}")
        & src["store_id"].str.fullmatch(r"S-\d{2}")
        & src["quantity"].str.fullmatch(r"\d+")
        & src["returned"].isin(["0", "1"])
    )
    clean = src[ok].drop_duplicates().reset_index(drop=True)
    picks = np.sort(rng.choice(len(clean), size=39630, replace=False))
    return clean.iloc[picks].reset_index(drop=True)


def messy_orders() -> None:
    """40,000 rows: 39,630 base rows plus duplicates, with 14 defect types."""
    rng = np.random.default_rng(SEED)
    df = _clean_base(rng)

    # 1. missing values (400): blank region / discount / customer_id
    df.loc[0:149, "region"] = ""
    df.loc[150:299, "discount_pct"] = ""
    df.loc[300:399, "customer_id"] = ""

    # 2. sentinel values (300): tokens that mean missing but read as data
    df.loc[400:459, "customer_id"] = "-1"
    df.loc[460:519, "quantity"] = "9999"
    df.loc[520:559, "region"] = "unknown"
    df.loc[560:599, "region"] = "N/A"
    df.loc[600:659, "discount_pct"] = "9999"
    df.loc[640:699, "order_date"] = "1900-01-01"

    # 3. case variants (200): category drift
    df.loc[700:799, "category"] = df.loc[700:799, "category"].str.lower()
    df.loc[800:899, "category"] = df.loc[800:899, "category"].str.upper()

    # 4. whitespace padding (200, incl. non-breaking space)
    df.loc[900:1049, "category"] = " " + df.loc[900:1049, "category"] + " "
    df.loc[1050:1099, "region"] = df.loc[1050:1099, "region"] + "\u00a0"

    # 5. currency-formatted numbers (150)
    df.loc[1100:1249, "unit_price"] = "$" + df.loc[1100:1249, "unit_price"]

    # 6. negative quantities (80)
    df.loc[1250:1329, "quantity"] = "-" + df.loc[1250:1329, "quantity"]

    # 7. zero prices (60)
    df.loc[1330:1389, "unit_price"] = "0.00"
    df.loc[1330:1389, "revenue"] = "0.00"

    # 8. bad dates (140): impossible, unparseable, future
    df.loc[1390:1449, "order_date"] = "2026-02-30"
    df.loc[1450:1489, "order_date"] = "unknown"
    df.loc[1490:1529, "order_date"] = "2030-05-01"

    # 9. mixed date formats (120): US style, same instant, different text
    us = pd.to_datetime(df.loc[1530:1649, "order_date"], format="%Y-%m-%d")
    df.loc[1530:1649, "order_date"] = us.dt.strftime("%m/%d/%Y")

    # 10. out-of-range discounts (70)
    df.loc[1650:1689, "discount_pct"] = "150.0"
    df.loc[1690:1719, "discount_pct"] = "-10.0"

    # 11. broken foreign keys (90)
    df.loc[1720:1769, "product_id"] = "P-9999"
    df.loc[1770:1809, "store_id"] = "S-99"

    # 12. revenue mismatch (60): does not recompute from the formula
    vals = df.loc[1810:1869, "revenue"].astype(float)
    df.loc[1810:1869, "revenue"] = (vals * 1.5 + 10).round(2).map(lambda v: f"{v:.2f}")

    # a block where missingness co-occurs (100 rows): region, discount and
    # customer_id are all blank, so the co-occurrence heatmap has an off-diagonal
    df.loc[1870:1969, "region"] = ""
    df.loc[1870:1969, "discount_pct"] = ""
    df.loc[1870:1969, "customer_id"] = ""

    # 13. exact duplicate rows (250) and 14. conflicting business keys (120)
    exact = df.iloc[20000:20250].copy()
    conflict = df.iloc[21000:21120].copy()
    conflict["quantity"] = (
        pd.to_numeric(conflict["quantity"], errors="coerce").astype("Int64") + 1
    ).astype(str)
    conflict["revenue"] = (
        (pd.to_numeric(conflict["revenue"], errors="coerce") * 1.10)
        .round(2)
        .map(lambda v: f"{v:.2f}")
    )
    out = pd.concat([df, exact, conflict], ignore_index=True)
    assert len(out) == 40000, len(out)

    AUDIT.mkdir(parents=True, exist_ok=True)
    out[COLUMNS].to_csv(AUDIT / "orders_messy_40k.csv", index=False)
    print(f"  orders_messy_40k.csv: {len(out)} rows")


# ------------------------------------------------------------- customer fixture
GIVEN = {
    "m": ["James", "Robert", "William", "Michael", "David", "Richard", "Joseph",
          "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark", "Steven", "Paul"],
    "f": ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan",
          "Jessica", "Sarah", "Karen", "Nancy", "Lisa", "Margaret", "Betty", "Sandra"],
}
NICKNAMES = {
    "Robert": ["Bob", "Rob", "Bobby"],
    "William": ["Bill", "Will", "Billy"],
    "Elizabeth": ["Liz", "Beth", "Betty"],
    "Margaret": ["Maggie", "Peggy"],
    "James": ["Jim", "Jimmy"],
    "Michael": ["Mike"],
    "Jennifer": ["Jen", "Jenny"],
    "Patricia": ["Pat", "Patty"],
    "Richard": ["Rick", "Rich"],
    "David": ["Dave"],
    "Joseph": ["Joe"],
    "Charles": ["Charlie", "Chuck"],
    "Susan": ["Sue"],
}
SURNAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Wilson", "Anderson", "Taylor", "Thomas", "Moore",
    "Jackson", "Martin", "Lee", "Walker", "Hall", "Young", "King", "Wright",
    "Nguyen", "Patel", "Rossi", "Kowalski", "Nowak", "Muller", "O'Brien",
    "Garcia", "Fernandez", "Schmidt", "Novak", "Hansen",
]
STREETS = [
    ("Elm", "Street"), ("Oak", "Avenue"), ("Maple", "Drive"), ("Cedar", "Lane"),
    ("Pine", "Road"), ("Birch", "Court"), ("Walnut", "Street"),
    ("Sunset", "Boulevard"), ("Lakeview", "Drive"), ("Hillcrest", "Road"),
    ("Aspen", "Court"), ("Willow", "Lane"),
]
STREET_ABBR = {"Street": "St", "Avenue": "Ave", "Drive": "Dr", "Lane": "Ln",
               "Road": "Rd", "Court": "Ct", "Boulevard": "Blvd"}
CITIES = [("Springfield", "IL", "62704"), ("Riverton", "OH", "43065"),
          ("Fairview", "TX", "75069"), ("Clayton", "MO", "63105"),
          ("Georgetown", "KY", "40324"), ("Salem", "OR", "97301")]
STATE_FULL = {"IL": "Illinois", "OH": "Ohio", "TX": "Texas", "MO": "Missouri",
              "KY": "Kentucky", "OR": "Oregon"}
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "example.com"]


def _typo(word: str, rnd: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rnd.randrange(1, len(word) - 1)
    kind = rnd.random()
    if kind < 0.34:
        return word[:i] + word[i + 1:]                 # drop a character
    if kind < 0.67:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]  # swap
    return word[:i] + word[i + 1] + word[i + 1:]       # double a character


def _noisy_name(given: str, surname: str, rnd: random.Random) -> str:
    r = rnd.random()
    if r < 0.45:
        return f"{given} {surname}"
    if r < 0.60:
        return f"{surname}, {given}"
    if r < 0.72:
        return f"{given.upper()} {surname.upper()}"
    if r < 0.84 and given in NICKNAMES:
        return f"{rnd.choice(NICKNAMES[given])} {surname}"
    if r < 0.92:
        return f"{given}  {surname}"
    return f"{given} {_typo(surname, rnd)}"


def _noisy_address(number: int, base: str, stype: str, rnd: random.Random) -> str:
    r = rnd.random()
    if r < 0.35:
        return f"{number} {base} {stype}"
    if r < 0.55:
        return f"{number} {base} {STREET_ABBR[stype]}"
    if r < 0.75:
        return f"{number} {base} {stype}, Apt {rnd.randint(1, 24)}"
    if r < 0.88:
        return f"{number} {base} {stype}".lower()
    return f"{number} {base} {_typo(stype, rnd)}"


def _noisy_phone(rnd: random.Random) -> str:
    area, mid, last = rnd.randint(200, 989), rnd.randint(200, 999), rnd.randint(0, 9999)
    style = rnd.random()
    if style < 0.3:
        return f"({area}) {mid}-{last:04d}"
    if style < 0.55:
        return f"{area}-{mid}-{last:04d}"
    if style < 0.8:
        return f"{area}{mid}{last:04d}"
    return f"+1 {area} {mid} {last:04d}"


def messy_customers() -> None:
    """~2,300 messy records across ~950 households, plus a truth mapping."""
    rnd = random.Random(SEED)
    n_h = 950
    sizes = [1] * n_h
    while sum(sizes) < 2300:
        i = rnd.randrange(n_h)
        if sizes[i] < 5:
            sizes[i] += 1

    households = []
    for h in range(n_h):
        base, stype = rnd.choice(STREETS)
        city, state, zipc = rnd.choice(CITIES)
        households.append(
            {
                "household_id": f"H-{h + 1:04d}",
                "surname": rnd.choice(SURNAMES),
                "number": rnd.randint(1, 9900),
                "base": base,
                "stype": stype,
                "city": city,
                "state": state,
                "zip": zipc,
            }
        )

    # Fifteen pairs of roommates: same address, different surname, different home.
    pairs = rnd.sample(range(n_h), 30)
    for k in range(0, 30, 2):
        a, b = pairs[k], pairs[k + 1]
        for field in ("number", "base", "stype", "city", "state", "zip"):
            households[b][field] = households[a][field]
        if households[b]["surname"] == households[a]["surname"]:
            households[b]["surname"] = "Bennett" if households[a]["surname"] != "Bennett" else "Carter"

    records, truth = [], []
    rid = 0
    for h, size in enumerate(sizes):
        hh = households[h]
        for _ in range(size):
            rid += 1
            given = rnd.choice(GIVEN["m"] if rnd.random() < 0.5 else GIVEN["f"])
            name = _noisy_name(given, hh["surname"], rnd)
            address = _noisy_address(hh["number"], hh["base"], hh["stype"], rnd)
            city = hh["city"].upper() if rnd.random() < 0.2 else hh["city"]
            state = STATE_FULL[hh["state"]] if rnd.random() < 0.15 else hh["state"]
            zipc = hh["zip"] + "-" + f"{rnd.randint(1, 9999):04d}" if rnd.random() < 0.1 else hh["zip"]
            if rnd.random() < 0.3:
                email = f"{hh['surname'].lower().replace(chr(39), '')}family{rnd.randint(1, 99)}@{rnd.choice(EMAIL_DOMAINS)}"
            else:
                email = f"{given.lower()}.{hh['surname'].lower().replace(chr(39), '')}@{rnd.choice(EMAIL_DOMAINS)}"
            signup = f"20{rnd.randint(22, 25)}-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}"
            records.append(
                [f"R-{rid:06d}", name, address, city, state, zipc, email,
                 _noisy_phone(rnd), signup]
            )
            truth.append([f"R-{rid:06d}", hh["household_id"]])

    frame = pd.DataFrame(
        records,
        columns=["record_id", "full_name", "address_line", "city", "state",
                 "postal_code", "email", "phone", "signup_date"],
    )
    frame.to_csv(AUDIT / "customers_messy.csv", index=False)
    pd.DataFrame(truth, columns=["record_id", "household_id"]).to_csv(
        AUDIT / "households_truth.csv", index=False
    )
    print(f"  customers_messy.csv: {len(frame)} records, {n_h} households")


# --------------------------------------------------------------- income fixture
def income_churn() -> None:
    """Income missing at ~12%, with the mechanism tied to segment (MAR)."""
    rng = np.random.default_rng(SEED)
    n = 3000
    segments = rng.choice(["Basic", "Plus", "Premium"], size=n, p=[0.45, 0.35, 0.20])
    center = {"Basic": 45000.0, "Plus": 68000.0, "Premium": 105000.0}
    mu = np.log(np.array([center[s] for s in segments]))
    income = np.exp(rng.normal(mu, 0.35))
    tenure = rng.integers(1, 61, size=n)

    z = (np.log(income) - mu) / 0.35
    lin = 0.9 - 0.7 * z - 0.9 * ((tenure - 30) / 30.0)
    p_churn = 1.0 / (1.0 + np.exp(-lin))
    churn = rng.binomial(1, p_churn)

    p_miss = {"Basic": 0.16, "Plus": 0.10, "Premium": 0.05}
    miss = rng.random(n) < np.array([p_miss[s] for s in segments])
    income_out = np.round(income / 500.0) * 500.0
    income_out = income_out.astype(object)
    income_out[miss] = np.nan

    out = pd.DataFrame(
        {
            "customer_id": [f"C-{100000 + i}" for i in range(n)],
            "segment": segments,
            "tenure_months": tenure,
            "income": income_out,
            "churn": churn,
        }
    )
    out.to_csv(AUDIT / "income_churn.csv", index=False)
    print(f"  income_churn.csv: {len(out)} rows, "
          f"{out['income'].isna().mean():.2%} missing income")


# ------------------------------------------------------------------- fx fixture
def fx_fixtures() -> None:
    """Weekly rates to USD, and multi-currency orders to convert as-of join."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2025-01-06", "2026-03-30", freq="W-MON")
    base = {"EUR": 1.08, "GBP": 1.27, "CAD": 0.74}
    rows = []
    for cur, start in base.items():
        walk = start + np.cumsum(rng.normal(0, 0.004, size=len(dates)))
        for d, rate in zip(dates, walk):
            rows.append([d.strftime("%Y-%m-%d"), cur, round(float(rate), 4)])
    rates = pd.DataFrame(rows, columns=["rate_date", "currency", "usd_per_unit"])
    rates.to_csv(AUDIT / "fx_rates.csv", index=False)

    n = 600
    order_dates = pd.to_datetime("2024-12-15") + pd.to_timedelta(
        rng.integers(0, 470, size=n), unit="D"
    )
    currencies = rng.choice(["EUR", "GBP", "CAD"], size=n, p=[0.6, 0.25, 0.15])
    amounts = np.round(rng.uniform(25, 2400, size=n), 2)
    orders = pd.DataFrame(
        {
            "order_id": [f"FX-{i + 1:04d}" for i in range(n)],
            "order_date": order_dates.strftime("%Y-%m-%d"),
            "currency": currencies,
            "amount": amounts,
        }
    )
    orders.to_csv(AUDIT / "eu_orders_fx.csv", index=False)
    print(f"  fx_rates.csv: {len(rates)} rates; eu_orders_fx.csv: {len(orders)} orders")


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    print("generating Chapter 6 cleaning fixtures ...")
    messy_orders()
    messy_customers()
    income_churn()
    fx_fixtures()
    print("done.")


if __name__ == "__main__":
    main()
