#!/usr/bin/env python3
"""Generate the Excel fixtures that Chapter 14 uses.

Run from the project root:

    python scripts/generate_chapter14_sources.py

Outputs (under data/raw/excel/):

    sales_workbook.xlsx    a realistic messy workbook with formulas and cached values
    monthly_sales.xlsx     12 monthly worksheets
    header_probe.xlsx      sheets whose real header is on different rows
    budget_input.csv       12 months of budget and actual for what-if work

Everything is deterministic. Workbooks are written with xlsxwriter so formula
cells carry cached values that openpyxl can read with data_only=True.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
XL = RAW / "excel"
SEED = 1407


def sales_workbook() -> None:
    rng = np.random.default_rng(SEED)
    path = XL / "sales_workbook.xlsx"
    wb = xlsxwriter.Workbook(path)

    title = wb.add_format({"bold": True, "font_size": 14})
    meta = wb.add_format({"italic": True, "font_color": "#666666"})
    head = wb.add_format({"bold": True, "bg_color": "#DDEBF7", "border": 1})
    money_fmt = wb.add_format({"num_format": "#,##0.00"})
    date_fmt = wb.add_format({"num_format": "yyyy-mm-dd"})

    ws = wb.add_worksheet("Raw Data")
    ws.merge_range("A1:H1", "Kestrel Monthly Sales Export", title)
    ws.write("A2", "Generated 2026-04-02; do not edit by hand", meta)
    headers = ["order_id", "order_date", "region", "category",
               "product", "qty", "unit_price", "revenue"]
    for c, h in enumerate(headers):
        ws.write(2, c, h, head)

    products = [("Camping Tent", 58.53), ("Trail Shoe", 88.29),
                ("Rain Shell", 102.10), ("Day Pack", 44.63)]
    regions = ["Midwest", "Northeast", "South", "West", "West "]
    n = 200
    for i in range(n):
        r = i + 3
        prod, price = products[i % len(products)]
        qty = int(rng.integers(1, 5))
        region = regions[i % len(regions)]
        rev = round(qty * price * (1 - (i % 4) * 0.05), 2)
        ws.write(r, 0, f"O-{100000 + i}")
        d = pd.Timestamp("2025-01-01") + pd.DateOffset(days=int(rng.integers(0, 365)))
        if i % 17 == 0:
            ws.write(r, 1, d.strftime("%m/%d/%Y"))  # text date
        else:
            ws.write_datetime(r, 1, d.to_pydatetime(), date_fmt)
        ws.write(r, 2, region)
        ws.write(r, 3, prod.split()[0].upper() if i % 23 == 0 else prod.split()[0])
        ws.write(r, 4, prod)
        ws.write(r, 5, qty)
        ws.write(r, 6, price, money_fmt)
        ws.write_formula(r, 7, f"=F{r+1}*G{r+1}", money_fmt, rev)

    total_row = n + 3
    ws.write(total_row, 0, "TOTAL", head)
    ws.write_formula(total_row, 5, f"=SUM(F4:F{n+3})", None, 0)
    ws.write_formula(total_row, 7, f"=SUM(H4:H{n+3})", money_fmt, 0.0)
    ws.set_column("A:A", 14)
    ws.set_column("B:B", 14)
    ws.set_column("C:E", 18)
    ws.set_column("F:H", 13)

    look = wb.add_worksheet("Lookups")
    for c, h in enumerate(["product", "price", "category"]):
        look.write(0, c, h, head)
    for i, (prod, price) in enumerate(products):
        look.write(i + 1, 0, prod)
        look.write(i + 1, 1, price, money_fmt)
        look.write(i + 1, 2, prod.split()[0])
    look.set_column("A:C", 18)

    tgt = wb.add_worksheet("Targets")
    for c, h in enumerate(["region", "target"]):
        tgt.write(0, c, h, head)
    for i, region in enumerate(["Midwest", "Northeast", "South", "West"]):
        tgt.write(i + 1, 0, region)
        tgt.write_formula(i + 1, 1, f"={120000 + i * 5000}", money_fmt,
                          120000 + i * 5000)
    tgt.set_column("A:B", 16)

    summary = wb.add_worksheet("Summary")
    summary.write(0, 0, "Kestrel Q1 Summary", title)
    labels = ["Total revenue", "Total units", "Average price", "Target", "Notes"]
    cached = [182346.72, 452, 58.53, 135000.0, "hard-coded override"]
    for i, (label, val) in enumerate(zip(labels, cached)):
        summary.write(i + 2, 0, label, head)
        if i == 3:
            summary.write(i + 2, 1, val, money_fmt)  # override, not a formula
        elif i == 4:
            summary.write(i + 2, 1, val)
        else:
            summary.write_formula(i + 2, 1, f"=ROUND({val},2)", money_fmt, val)
    summary.set_column("A:A", 20)
    summary.set_column("B:B", 18)

    notes = wb.add_worksheet("Notes")
    notes.write(0, 0, "Assumptions and caveats", title)
    for i, line in enumerate([
        "Revenue is gross, before returns and shipping.",
        "Regional targets are placeholders pending finance sign-off.",
        "The West region has a trailing space in the source feed.",
        "The Hidden sheet is a stale copy; do not use it.",
    ]):
        notes.write(i + 2, 0, line)

    hidden = wb.add_worksheet("Hidden")
    hidden.write(0, 0, "stale copy")
    hidden.write(1, 0, 12345)
    hidden.hide()

    wb.close()
    print(f"  sales_workbook.xlsx: 7 sheets, {n} data rows")


def monthly_sheets() -> None:
    rng = np.random.default_rng(SEED)
    path = XL / "monthly_sales.xlsx"
    wb = xlsxwriter.Workbook(path)
    head = wb.add_format({"bold": True, "bg_color": "#DDEBF7"})
    for m in range(1, 13):
        ws = wb.add_worksheet(f"2025-{m:02d}")
        for c, h in enumerate(["order_id", "region", "revenue"]):
            ws.write(0, c, h, head)
        for i in range(24):
            ws.write(i + 1, 0, f"O-{m:02d}{i:03d}")
            ws.write(i + 1, 1, ["West", "South", "East"][i % 3])
            ws.write(i + 1, 2, round(float(rng.uniform(100, 900)), 2))
        ws.set_column("A:C", 14)
    wb.close()
    print("  monthly_sales.xlsx: 12 sheets")


def header_probe() -> None:
    path = XL / "header_probe.xlsx"
    wb = xlsxwriter.Workbook(path)
    head = wb.add_format({"bold": True})
    for name, skip in [("header_row_1", 0), ("header_row_2", 1),
                       ("header_row_3", 2), ("header_row_5", 4)]:
        ws = wb.add_worksheet(name)
        for r in range(skip):
            ws.write(r, 0, "title or note")
        for c, h in enumerate(["id", "value"]):
            ws.write(skip, c, h, head)
        for i in range(5):
            ws.write(skip + i + 1, 0, f"R-{i}")
            ws.write(skip + i + 1, 1, i * 10)
    wb.close()
    print("  header_probe.xlsx: 4 sheets with different header rows")


def budget_input() -> None:
    rng = np.random.default_rng(SEED)
    months = pd.date_range("2025-01-01", periods=12, freq="MS").strftime("%Y-%m")
    df = pd.DataFrame({
        "month": months,
        "budget": np.round(rng.uniform(80000, 120000, 12), 2),
        "actual": np.round(rng.uniform(70000, 130000, 12), 2),
        "orders": rng.integers(400, 700, 12),
    })
    df.to_csv(XL / "budget_input.csv", index=False)
    print(f"  budget_input.csv: {len(df)} rows")


def main() -> None:
    XL.mkdir(parents=True, exist_ok=True)
    print("generating Chapter 14 Excel fixtures ...")
    sales_workbook()
    monthly_sheets()
    header_probe()
    budget_input()
    print("done.")


if __name__ == "__main__":
    main()
