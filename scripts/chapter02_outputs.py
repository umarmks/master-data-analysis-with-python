#!/usr/bin/env python3
"""Print every output block used in Chapter 2, from the current files."""

from __future__ import annotations

import csv
import glob
import json
import re
import statistics
import timeit
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from operator import itemgetter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


# 2.3 core types
show("0.1+0.2", 0.1 + 0.2)
show("19.99*3", 19.99 * 3)
show("1/3", 1 / 3)
show("Decimal", Decimal("19.99") * 3)

sku = "P-1009"
print("### sku slicing")
print(sku[:2], sku[-4:], sku.lower())
print()

revenue = 4392118.03
print("### f-string money")
print(f"Net revenue: ${revenue:,.2f}")
print()

raw_amount = "  $4,250.75 "
print("### parse amount")
print(float(raw_amount.strip().replace("$", "").replace(",", "")))
print()

print("### truthiness")
for value in [None, "", 0, "0", False, []]:
    print(repr(value), bool(value))
print()

a = [1, 2, 3]
b = a
c = list(a)
print("### identity")
print(a is b, a is c, a == c)
print()

print("### conversions")
print(int("42"), float("3.5"), bool("False"), str(2026))
print()

# 2.4 collections
products = [("P-1009", 1299.00), ("P-1003", 45.00), ("P-1008", 349.00)]
print("### sorted products")
print(sorted(products, key=itemgetter(1), reverse=True))
print()

counts = Counter(["Web", "App", "Web", "Phone", "Web", "App"])
print("### Counter")
print(counts)
print(counts.most_common(2))
print()

big_list = list(range(1_000_000))
big_set = set(big_list)
list_time = timeit.timeit(lambda: 999_999 in big_list, number=200)
set_time = timeit.timeit(lambda: 999_999 in big_set, number=200)
print("### membership timing (varies by machine)")
print(f"list: {list_time:.4f}s  set: {set_time:.4f}s")
print()

# 2.5 control flow
net_values = [120.50, 89.00, 245.75, 0.00]
total = 0
for value in net_values:
    total += value
print("### accumulator")
print(f"Total: ${total:,.2f}")
print()

prices = [59.0, 89.0, 1299.0, 45.0]
print("### comprehension")
print([p for p in prices if p > 100])
print({name: len(name) for name in ["Web", "App", "Phone"]})
print()

channels = ["Web", "App", "Phone"]
revenues = [2702000, 1318000, 372118]
print("### enumerate zip")
for rank, (channel, revenue_value) in enumerate(zip(channels, revenues), start=1):
    print(f"{rank}. {channel}: ${revenue_value:,}")
print()

rows = [("Web", 120.50), ("App", 89.00), ("Web", 245.75), ("Phone", 45.00)]
totals = defaultdict(float)
for channel, net in rows:
    totals[channel] += net
print("### defaultdict grouping")
print(dict(totals))
print()


# 2.6 functions
def net_revenue(revenue_value, returned=0, shipping_cost=0.0):
    """Return revenue after returns and shipping for one line."""
    return round(revenue_value * (1 - returned) - shipping_cost, 2)


print("### net_revenue")
print(net_revenue(173.57, 0, 7.95))
print(net_revenue(120.00, 1, 3.95))
print()


def total(*values):
    return sum(values)


print("### *args")
print(total(1.5, 2.5, 3.0))
print()


def describe(**fields):
    return ", ".join(f"{k}={v}" for k, v in sorted(fields.items()))


print("### **kwargs")
print(describe(channel="Web", region="South", category="Camping"))
print()

# 2.7 standard library
print("### pathlib")
print((Path("data") / "raw" / "kestrel_orders.csv").as_posix())
print(Path("data/raw/kestrel_orders.csv").suffix)
print()

print("### glob")
print(sorted(p.name for p in Path(RAW).glob("*.csv")))
print()

with open(RAW / "expenses.csv", newline="", encoding="utf-8") as handle:
    expense_rows = list(csv.DictReader(handle))
print("### csv DictReader")
print(len(expense_rows), expense_rows[0])
print()

config = {"input": "data/raw/expenses.csv", "report": "reports/expenses.txt"}
print("### json")
print(json.dumps(config, indent=2))
print()

match = re.search(r"\d[\d,]*\.?\d*", raw_amount)
print("### regex")
print(match.group())
print()

samples = [185.78, 179.99, 190.93, 184.31, 182.85, 180.16]
print("### statistics")
print(round(statistics.mean(samples), 2), round(statistics.median(samples), 2))
print()

print("### datetime")
start = date(2026, 1, 1)
end = date(2026, 3, 31)
print((end - start).days, "days")
print()

# 2.8 files and exceptions
def parse_amount(raw):
    try:
        return float(raw.strip().replace("$", "").replace(",", ""))
    except (AttributeError, ValueError):
        return None


print("### parse_amount")
print(parse_amount("  $4,250.75 "))
print(parse_amount("n/a"))
print()


def normalize_category(name):
    return name.strip().title()


good, rejected = [], []
with open(RAW / "expenses.csv", newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        amount = parse_amount(row["amount"])
        if amount is None:
            rejected.append(row)
        else:
            row["amount"] = amount
            row["category"] = normalize_category(row["category"])
            good.append(row)
print("### clean counts")
print(f"kept={len(good)} rejected={len(rejected)}")
print("rejected:", rejected)
print()


def dedupe(records):
    seen = set()
    unique = []
    for record in records:
        key = (record["date"], record["category"], record["description"], record["amount"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


unique_expenses = dedupe(good)
print("### dedupe")
print(f"before={len(good)} after={len(unique_expenses)}")
print()

try:
    with open(RAW / "expenses.csv", encoding="ascii") as handle:
        handle.read()
except UnicodeDecodeError as error:
    print("### unicode error")
    print(type(error).__name__, "-", error.reason)
    print()

# 2.9 debugging and testing
def validate(records):
    assert records, "no rows to summarize"
    for record in records:
        assert {"channel", "net"} <= record.keys(), record
    return True


print("### validate")
print(validate([{"channel": "Web", "net": 120.5}]))
print()

# 2.10 projects
monthly = defaultdict(float)
monthly_orders = Counter()
for record in unique_expenses:
    month = record["date"][:7]
    monthly[month] += record["amount"]
    monthly_orders[month] += 1
print("### expense tracker monthly")
for month in sorted(monthly):
    print(f"{month}: ${monthly[month]:,.2f} ({monthly_orders[month]} items)")
print()

with open(RAW / "access.log", encoding="utf-8") as handle:
    log_lines = [line.strip() for line in handle if line.strip()]
pattern = re.compile(
    r'(?P<ip>\S+) .* "GET (?P<url>\S+) HTTP/1.1" (?P<status>\d{3}) \S+ (?P<time>[0-9.]+)$'
)
parsed = [pattern.search(line) for line in log_lines]
records = [m.groupdict() for m in parsed if m]
status_counts = Counter(record["status"] for record in records)
url_counts = Counter(record["url"] for record in records)
slowest = sorted(records, key=lambda r: float(r["time"]), reverse=True)[:5]
print("### log parser")
print("lines:", len(log_lines), "parsed:", len(records))
print("statuses:", dict(sorted(status_counts.items())))
print("top urls:", url_counts.most_common(3))
print("slowest:")
for record in slowest:
    print(f"  {record['time']}s {record['status']} {record['url']}")
print()

print("### report generator")
unique_desc = len({record["description"] for record in unique_expenses})
print(f"months={len(monthly)} items={len(unique_expenses)} categories="
      f"{len({record['category'] for record in unique_expenses})} descriptions={unique_desc}")
print(f"largest month: {max(monthly, key=monthly.get)} ${max(monthly.values()):,.2f}")
