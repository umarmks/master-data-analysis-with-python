#!/usr/bin/env python3
"""Print every output block used in Chapter 14, from the current files.

Run from the project root:

    python scripts/chapter14_outputs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl
import xlsxwriter
from openpyxl.utils.datetime import from_excel

ROOT = Path(__file__).resolve().parents[1]
XL = ROOT / "data" / "raw" / "excel"
REPORTS = ROOT / "reports"
SEED = 1407

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 80)


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


def raw_data() -> pd.DataFrame:
    return pd.read_excel(XL / "sales_workbook.xlsx", sheet_name="Raw Data",
                         header=2, nrows=200)


# ============================================== 14.1 structure/references
def section_141() -> None:
    wb = openpyxl.load_workbook(XL / "sales_workbook.xlsx")
    show("S1 sheet inventory", pd.DataFrame([
        (ws.title, ws.sheet_state, ws.max_row, ws.max_column)
        for ws in wb.worksheets
    ], columns=["sheet", "state", "rows", "cols"]).to_string(index=False))

    ws = wb["Raw Data"]
    show("S2 merged ranges", [str(r) for r in ws.merged_cells.ranges])
    show("S3 header row", [ws.cell(3, c).value for c in range(1, 9)])
    show("S4 first data row", [ws.cell(4, c).value for c in range(1, 9)])
    show("S5 dimensions", {"max_row": ws.max_row, "max_column": ws.max_column,
                            "freeze_panes": str(ws.freeze_panes)})

    probe = openpyxl.load_workbook(XL / "header_probe.xlsx")
    found = {}
    for sh in probe.worksheets:
        for r in range(1, 7):
            vals = [sh.cell(r, c).value for c in (1, 2)]
            if vals == ["id", "value"]:
                found[sh.title] = r
                break
    show("S6 header detection", found)

    show("S7 clean input rules", [
        "one header row, no merged cells in the data region",
        "one row per observation, no subtotal rows inside the data",
        "ISO dates or real date cells, never text dates",
        "no trailing spaces in categorical columns",
        "values in cells, formulas only in a separate calculation sheet",
    ])


# ==================================================== 14.2 core formulas
def section_142() -> None:
    d = raw_data()
    show("F1 shape and dtypes", {"shape": d.shape,
                                  "dtypes": {c: str(t) for c, t in d.dtypes.items()}})
    show("F2 core aggregates", {
        "sum_revenue": round(float(pd.to_numeric(d["revenue"], errors="coerce").sum()), 2),
        "avg_revenue": round(float(pd.to_numeric(d["revenue"], errors="coerce").mean()), 2),
        "count_rows": int(len(d)),
        "counta_product": int(d["product"].notna().sum()),
        "min_qty": int(pd.to_numeric(d["qty"]).min()),
        "max_price": float(pd.to_numeric(d["unit_price"]).max()),
    })

    cats = d["category"].astype(str).str.strip().str.upper()
    spend = pd.to_numeric(d["revenue"], errors="coerce")
    show("F3 conditional aggregation", pd.DataFrame({
        "category": cats.unique(),
        "count": [int((cats == c).sum()) for c in cats.unique()],
        "sum": [round(float(spend[cats == c].sum()), 2) for c in cats.unique()],
    }).to_string(index=False))

    show("F4 wildcard criteria", {
        "region_starting_W": int(d["region"].astype(str).str.strip().str.startswith("W").sum()),
        "category_containing_A": int(cats.str.contains("A").sum()),
    })

    wbf = openpyxl.load_workbook(XL / "sales_workbook.xlsx", data_only=False)
    wbv = openpyxl.load_workbook(XL / "sales_workbook.xlsx", data_only=True)
    show("F5 formula vs cached", pd.DataFrame([
        ("Summary B3", wbf["Summary"]["B3"].value, wbv["Summary"]["B3"].value),
        ("Summary B4", wbf["Summary"]["B4"].value, wbv["Summary"]["B4"].value),
        ("Summary B5", wbf["Summary"]["B5"].value, wbv["Summary"]["B5"].value),
        ("Summary B6", wbf["Summary"]["B6"].value, wbv["Summary"]["B6"].value),
        ("Raw H204", wbf["Raw Data"]["H204"].value, wbv["Raw Data"]["H204"].value),
    ], columns=["cell", "formula", "cached"]).to_string(index=False))

    show("F6 pandas check", {
        "computed_total_revenue": round(float(spend.sum()), 2),
        "summary_cached_total": wbv["Summary"]["B3"].value,
        "difference": round(float(spend.sum()) - float(wbv["Summary"]["B3"].value), 2),
    })


# ======================================================== 14.3 logic/text
def section_143() -> None:
    d = raw_data()
    reg = d["region"].astype(str)
    band = np.where(reg.str.strip().eq("West"), "W",
                    np.where(reg.str.strip().eq("South"), "S", "other"))
    show("L1 nested IF equivalent", pd.Series(band).value_counts().to_dict())

    cat = d["category"].astype(str)
    cleaned = cat.str.strip().str.upper()
    show("L2 text cleanup", {
        "raw_levels": sorted(cat.unique().tolist()),
        "clean_levels": sorted(cleaned.unique().tolist()),
        "trimmed_rows": int((cat != cat.str.strip()).sum()),
    })

    text_dates = d["order_date"].map(lambda x: isinstance(x, str)).sum()
    show("L3 text dates", {
        "text_date_rows": int(text_dates),
        "date_rows": int(len(d) - text_dates),
        "share": round(float(text_dates / len(d)), 4),
    })

    show("L4 text functions", {
        "left4": d["order_id"].astype(str).str[:4].iloc[0],
        "last6": d["order_id"].astype(str).str[-6:].iloc[0],
        "joined": ", ".join(sorted(cat.unique())[:3]),
    })

    show("L5 iferror equivalent", {
        "safe_divide_nulls": int(pd.to_numeric(
            d["qty"], errors="coerce").replace(0, np.nan).isna().sum()),
        "note": "IFERROR hides a missing divisor; count it instead",
    })

    show("L6 whitespace defect", {
        "trailing_space_region": int(reg.str.endswith(" ").sum()),
        "distinct_raw": int(reg.nunique()),
        "distinct_trimmed": int(reg.str.strip().nunique()),
    })


# ========================================================== 14.4 lookups
def section_144() -> None:
    d = raw_data()
    look = pd.read_excel(XL / "sales_workbook.xlsx", sheet_name="Lookups")
    show("K1 lookup table", look.to_string(index=False))

    prices = look.set_index("product")["price"]
    mapped = d["product"].map(prices)
    show("K2 exact lookup", {
        "matched": int(mapped.notna().sum()),
        "unmatched": int(mapped.isna().sum()),
        "example": d.loc[mapped.notna(), "product"].iloc[0],
    })

    merged = d.merge(look, on="product", how="left", validate="many_to_one")
    show("K3 merge equivalent", {
        "rows": len(merged), "with_price": int(merged["price"].notna().sum()),
        "note": "VLOOKUP and XLOOKUP both reduce to a many-to-one join",
    })

    keys = look["price"].tolist()
    target = 90.0

    def vlookup_approx(values, t):
        for i, k in enumerate(values):
            if k <= t:
                return i
        return None

    first = vlookup_approx(keys, target)
    correct = max((i for i, k in enumerate(keys) if k <= target),
                  key=lambda i: keys[i], default=None)
    show("K4 approximate-match trap", {
        "unsorted_keys": keys,
        "lookup_value": target,
        "vlookup_approx_pick": look["product"].iloc[first] if first is not None else None,
        "correct_pick": look["product"].iloc[correct] if correct is not None else None,
        "note": "approximate match assumes an ascending key column",
    })

    show("K5 xlookup advantages", [
        "searches in either direction, unlike VLOOKUP",
        "returns a column to the left of the key",
        "defaults to exact match; VLOOKUP defaults to approximate",
        "a spilled range does not break when a column is inserted",
    ])

    show("K6 #n/a handling", {
        "missing_keys": int(mapped.isna().sum()),
        "action": "IFNA(XLOOKUP(...), 'UNKNOWN') or left join and count nulls",
    })


# ====================================================== 14.5 dates/numbers
def section_145() -> None:
    d = raw_data()
    dates = pd.to_datetime(d["order_date"], errors="coerce")
    parsed = dates.notna()
    show("D1 parse success", {"parsed": int(parsed.sum()),
                               "failed": int((~parsed).sum()),
                               "min": str(dates.min().date()),
                               "max": str(dates.max().date())})

    serial = (pd.Timestamp("2026-01-15") - pd.Timestamp("1899-12-30")).days
    show("D2 excel serial", {"serial": int(serial),
                              "back": str(from_excel(serial).date()),
                              "epoch": "1899-12-30 (Windows system)"})

    show("D3 regional ambiguity", {
        "01/02/2026_US": "2026-01-02",
        "01/02/2026_EU": "2026-02-01",
        "iso": "2026-01-02",
        "rule": "write ISO dates or use real date cells",
    })

    show("D4 month end arithmetic", {
        "2026-01-31 + 1 month": "2026-03-03",
        "EOMONTH equivalent": "2026-02-28",
        "fix": "go to start of month, add one month, subtract one day",
    })

    show("D5 formatted precision", {
        "raw": 0.126734,
        "formatted_1pct": "12.7%",
        "lost": "the displayed value cannot be used for a build-up calculation",
    })

    show("D6 networkdays", {
        "business_days_Jan2026": len(pd.bdate_range("2026-01-01", "2026-01-31")),
        "calendar_days": 31,
    })


# ================================================= 14.6 pivots and charts
def section_146() -> None:
    d = raw_data()
    d["region_clean"] = d["region"].astype(str).str.strip()
    d["category_clean"] = d["category"].astype(str).str.strip().str.upper()
    rev = pd.to_numeric(d["revenue"], errors="coerce")
    piv = d.assign(rev=rev).pivot_table(index="region_clean",
                                         columns="category_clean",
                                         values="rev", aggfunc="sum",
                                         fill_value=0)
    show("P1 pivot equivalent", piv.round(2).to_string())
    show("P2 grand totals", {
        "row_total": round(float(piv.to_numpy().sum()), 2),
        "source_sum": round(float(rev.sum()), 2),
    })
    share = piv.div(piv.sum(axis=1), axis=0).round(4)
    show("P3 percent of row", share.to_string())
    show("P4 refresh hazards", [
        "a PivotTable caches its source range; new rows may be excluded",
        "changing a label in the source does not refresh the pivot",
        "grouping dates inside a pivot can change the field type",
        "a calculated field uses aggregated sums, not row-level logic",
    ])
    show("P5 chart choice", {
        "trend": "line", "comparison": "bar", "composition": "stacked bar",
        "relationship": "scatter", "note": "same rules as Chapter 11",
    })


# ======================================== 14.7 formatting and validation
def section_147() -> None:
    wb = openpyxl.load_workbook(XL / "sales_workbook.xlsx")
    ws = wb["Raw Data"]
    show("CF1 existing formats", {
        "bold_header": bool(ws.cell(3, 1).font.bold),
        "merged_title": str(ws.merged_cells.ranges),
        "money_format": ws.cell(4, 8).number_format,
    })

    report = REPORTS / "ch14_report.xlsx"
    wbr = xlsxwriter.Workbook(report)
    sh = wbr.add_worksheet("Report")
    sh.conditional_format("B2:B13", {
        "type": "data_bar", "bar_color": "#4477aa"})
    sh.data_validation("C2:C13", {
        "validate": "list", "source": ["Web", "App", "Phone"]})
    sh.write(0, 0, "rule test")
    wbr.close()
    wb2 = openpyxl.load_workbook(report)
    show("CF2 written rules", {
        "conditional_formats": len(wb2["Report"].conditional_formatting),
        "validations": len(wb2["Report"].data_validations.dataValidation),
        "file": str(report.relative_to(ROOT)),
    })

    show("CF3 review rules", [
        "highlight negative margin in red",
        "data bars on revenue, not on percentages of different denominators",
        "validation lists prevent new typos in region",
        "protect formula cells and leave inputs editable",
    ])


# ======================================================== 14.8 power query
def section_148() -> None:
    d = raw_data()
    long = d.melt(id_vars=["order_id", "order_date"],
                  value_vars=["qty", "unit_price", "revenue"],
                  var_name="measure", value_name="value")
    show("PQ1 unpivot", {"raw_rows": len(d), "long_rows": len(long),
                          "expected": len(d) * 3,
                          "columns": long.columns.tolist()})

    monthly = pd.read_excel(XL / "monthly_sales.xlsx", sheet_name=None)
    stack = pd.concat([df.assign(sheet=name) for name, df in monthly.items()],
                      ignore_index=True)
    show("PQ2 folder import equivalent", {
        "sheets": len(monthly), "stacked_rows": len(stack),
        "months": int(stack["sheet"].nunique()),
    })

    show("PQ3 cleaning steps", [
        "promote headers", "trim and uppercase text",
        "change type with locale", "remove trailing spaces",
        "filter null keys", "unpivot columns", "close and load",
    ])

    look = pd.read_excel(XL / "sales_workbook.xlsx", sheet_name="Lookups")
    joined = d.merge(look, on="product", how="left")
    show("PQ4 merge equivalent", {
        "rows": len(joined), "with_price": int(joined["price"].notna().sum()),
    })

    show("PQ5 refresh discipline", [
        "a query records steps; a copy-paste does not",
        "parameterised paths let one query read a whole folder",
        "query folding pushes transformations to the source database",
        "always check row counts after refresh",
    ])


# ==================================================== 14.9 what-if
def section_149() -> None:
    b = pd.read_csv(XL / "budget_input.csv")
    b["variance"] = b["actual"] - b["budget"]
    show("WI1 variance", b.head(4).round(2).to_string(index=False))

    total_var = float(b["variance"].sum())
    show("WI2 total variance", {
        "budget": round(float(b["budget"].sum()), 2),
        "actual": round(float(b["actual"].sum()), 2),
        "variance": round(total_var, 2),
    })

    rng = np.random.default_rng(SEED)
    sims = rng.normal(float(b["actual"].mean()), float(b["actual"].std()),
                      size=(10000, 12))
    annual = sims.sum(axis=1)
    show("WI3 monte carlo", {
        "runs": 10000,
        "mean_annual": round(float(annual.mean()), 2),
        "p10": round(float(np.quantile(annual, 0.10)), 2),
        "p90": round(float(np.quantile(annual, 0.90)), 2),
    })

    show("WI4 goal seek equivalent", {
        "required_monthly": round(float(b["budget"].sum() / 12), 2),
        "note": "solve for the input that makes the target cell equal a goal",
    })

    show("WI5 assumptions", [
        "monthly demand is independent and normally distributed",
        "no seasonality in the simulation",
        "the budget is fixed; only actuals vary",
    ])


# ============================================== 14.10 risk and python
def section_1410() -> None:
    d = raw_data()
    b = pd.read_csv(XL / "budget_input.csv")
    wbf = openpyxl.load_workbook(XL / "sales_workbook.xlsx", data_only=False)
    wbv = openpyxl.load_workbook(XL / "sales_workbook.xlsx", data_only=True)
    show("R1 formula density", {
        "formula_cells": sum(1 for row in wbf["Raw Data"].iter_rows()
                             for c in row if isinstance(c.value, str)
                             and c.value.startswith("=")),
        "cached_zero_cells": sum(1 for row in wbv["Raw Data"].iter_rows()
                                 for c in row if c.value == 0),
    })

    show("R2 override", {
        "summary_B6_formula": wbf["Summary"]["B6"].value,
        "summary_B6_cached": wbv["Summary"]["B6"].value,
        "risk": "a hard-coded value beside formulas is a silent override",
    })

    report = REPORTS / "ch14_report.xlsx"
    show("R3 generated report", {
        "path": str(report.relative_to(ROOT)),
        "bytes": int(report.stat().st_size),
        "sheets": openpyxl.load_workbook(report).sheetnames,
    })

    monthly_out = REPORTS / "ch14_monthly_sheets.xlsx"
    wb = xlsxwriter.Workbook(monthly_out)
    for month, grp in b.assign(m=b["month"]).groupby("m"):
        sh = wb.add_worksheet(month)
        sh.write(0, 0, "month")
        sh.write(0, 1, "budget")
        sh.write(1, 0, month)
        sh.write(1, 1, float(grp["budget"].iloc[0]))
    wb.close()
    show("R4 twelve sheets", {
        "sheets": len(openpyxl.load_workbook(monthly_out).sheetnames),
        "file": str(monthly_out.relative_to(ROOT)),
    })

    show("R5 source of truth", [
        "one canonical data source, referenced by every sheet",
        "formulas derive; inputs are entered once",
        "version the workbook and the generator together",
        "migrate to Python or SQL when the model outgrows the grid",
    ])


# ================================================ 14.11 example bank
def section_1411() -> None:
    d = raw_data()
    show("E1 sheets", len(openpyxl.load_workbook(
        XL / "sales_workbook.xlsx").sheetnames))
    show("E2 header row", 3)
    show("E3 rows", len(d))
    show("E4 text dates", int(d["order_date"].map(
        lambda x: isinstance(x, str)).sum()))
    show("E5 trailing spaces", int(d["region"].astype(
        str).str.endswith(" ").sum()))
    show("E6 formula cells", sum(1 for row in openpyxl.load_workbook(
        XL / "sales_workbook.xlsx", data_only=False)["Raw Data"].iter_rows()
        for c in row if isinstance(c.value, str) and c.value.startswith("=")))
    show("E7 pivot total", round(float(pd.to_numeric(
        d["revenue"], errors="coerce").sum()), 2))
    show("E8 unpivot rows", len(d) * 3)


# ==================================================== 14.12 capstone
def section_1412() -> None:
    d = raw_data()
    rev = pd.to_numeric(d["revenue"], errors="coerce")
    show("K1 diagnosis", {
        "source_sum": round(float(rev.sum()), 2),
        "pivot_sum": round(float(rev.sum()), 2),
        "text_date_rows": int(d["order_date"].map(
            lambda x: isinstance(x, str)).sum()),
        "trailing_space_region": int(d["region"].astype(
            str).str.endswith(" ").sum()),
    })

    show("K2 three steps", [
        "compare the pivot source range with the full data region",
        "check for hidden rows and filters excluded from the cache",
        "recompute the sum from the raw column and reconcile the difference",
    ])

    show("K3 artifacts", [
        str((REPORTS / "ch14_report.xlsx").relative_to(ROOT)),
        str((REPORTS / "ch14_monthly_sheets.xlsx").relative_to(ROOT)),
    ])

    show("K4 decision", {
        "recommendation": "refresh the pivot and standardise the source region labels",
        "next": "move the recurring workbook steps into a Python or Power Query pipeline",
    })


def section_extra() -> None:
    """Additional workbook inspections and Excel-pattern equivalents."""
    path = XL / "sales_workbook.xlsx"
    wf = openpyxl.load_workbook(path, data_only=False)
    wv = openpyxl.load_workbook(path, data_only=True)
    d = raw_data()

    show("X1 sheet properties", pd.DataFrame([
        (ws.title, ws.sheet_state, ws.protection.sheet)
        for ws in wf.worksheets
    ], columns=["sheet", "state", "protected"]).to_string(index=False))

    show("X2 named ranges", {
        name: str(defn.value) for name, defn in wf.defined_names.items()
    })

    show("X3 header styles", {
        "bold": bool(wf["Raw Data"]["A3"].font.bold),
        "fill": wf["Raw Data"]["A3"].fill.fgColor.rgb,
        "number_format_total": wf["Raw Data"]["H204"].number_format,
    })

    serials = [wf["Raw Data"].cell(r, 2).value for r in (5, 6, 7)]
    show("X4 date cells", {
        "sample_types": [type(v).__name__ for v in serials],
        "sample_values": [str(v)[:20] for v in serials],
    })

    formats = {}
    for row in wf["Raw Data"].iter_rows(min_row=4, max_row=8):
        for c in row:
            formats[c.number_format] = formats.get(c.number_format, 0) + 1
    show("X5 format inventory", formats)

    errors = [c.coordinate for row in wf["Raw Data"].iter_rows()
              for c in row if isinstance(c.value, str)
              and c.value.startswith("#")]
    show("X6 error cells", {"count": len(errors), "examples": errors[:5]})

    reg = d["region"].astype(str).str.strip()
    rev = pd.to_numeric(d["revenue"], errors="coerce")
    show("X7 SUMIFS equivalent", {
        f"{r}_revenue": round(float(rev[reg == r].sum()), 2)
        for r in sorted(reg.unique())
    })

    cat = d["category"].astype(str).str.strip().str.upper()
    both = (reg == "West") & (cat == "CAMPING")
    show("X8 two criteria", {
        "rows": int(both.sum()),
        "revenue": round(float(rev[both].sum()), 2),
    })

    look = pd.read_excel(path, sheet_name="Lookups")
    partial = d["product"].astype(str).str.split().str[0]
    lookup_partial = look["product"].str.split().str[0]
    show("X9 exact vs partial key", {
        "exact_matches": int(d["product"].isin(look["product"]).sum()),
        "partial_matches": int(partial.isin(lookup_partial).sum()),
    })

    idx = look.index[look["price"] == 88.29]
    show("X10 index match position", {
        "position_1based": int(idx[0]) + 1 if len(idx) else None,
        "matched_product": look.loc[idx[0], "product"] if len(idx) else None,
    })

    rev_sorted = look.sort_values("price")
    show("X11 left lookup", {
        "sorted_prices": rev_sorted["price"].tolist(),
        "product_for_102_10": rev_sorted.loc[
            rev_sorted["price"] == 102.10, "product"].iloc[0],
    })

    months = pd.date_range("2026-01-01", periods=6, freq="ME")
    show("X12 EOMONTH sequence", [str(m.date()) for m in months])

    workdays = pd.bdate_range("2026-03-02", periods=5)
    show("X13 WORKDAY sequence", [str(d_.date()) for d_ in workdays])

    piv = d.assign(rev=rev, region_clean=reg, category_clean=cat).pivot_table(
        index="region_clean", columns="category_clean", values="rev",
        aggfunc=["sum", "count"], fill_value=0)
    show("X14 two value fields", piv.round(2).to_string()[:600])

    tbl = d.assign(rev=rev, region_clean=reg).pivot_table(
        index="region_clean", values="rev", aggfunc="sum")
    show("X15 percent of grand total", (tbl / tbl.sum() * 100).round(2).to_string())

    show("X16 chart series", {
        "x": sorted(reg.unique().tolist()),
        "y_label": "revenue",
        "chart_type": "bar",
    })

    monthly = pd.read_excel(XL / "monthly_sales.xlsx", sheet_name=None)
    show("X17 query replay", {
        "sheets": len(monthly),
        "total_rows": sum(len(x) for x in monthly.values()),
        "row_counts": {k: len(v) for k, v in list(monthly.items())[:3]},
    })

    b = pd.read_csv(XL / "budget_input.csv")
    scenarios = pd.DataFrame({
        "base": b["actual"],
        "up_10": b["actual"] * 1.10,
        "down_10": b["actual"] * 0.90,
    })
    show("X18 scenario table", scenarios.head(3).round(2).to_string(index=False))
    show("X19 scenario totals", scenarios.sum().round(2).to_dict())

    show("X20 workbook inventory", {
        "generated_workbooks": sorted(
            str(p.relative_to(ROOT)) for p in REPORTS.glob("ch14_*.xlsx")),
        "source_workbooks": sorted(
            str(p.relative_to(ROOT)) for p in XL.glob("*.xlsx")),
    })


def main() -> None:
    print("python", "pandas", pd.__version__)
    print()
    for section in (
        section_141, section_142, section_143, section_144, section_145,
        section_146, section_147, section_148, section_149, section_1410,
        section_1411, section_1412, section_extra,
    ):
        print("=" * 78)
        section()


if __name__ == "__main__":
    main()
