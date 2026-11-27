#!/usr/bin/env python3
"""Clean a CSV of expenses and write a rejected-rows log.

Example:

    python scripts/csv_clean.py \
        --input data/raw/expenses.csv \
        --output data/processed/expenses_clean.csv \
        --log reports/rejected_expenses.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kestrel.metrics import parse_amount  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", default="reports/rejected_rows.csv")
    args = parser.parse_args()

    good, rejected = [], []
    with open(args.input, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            amount = parse_amount(row["amount"])
            if amount is None:
                rejected.append(row)
                continue
            row["amount"] = amount
            row["category"] = row["category"].strip().title()
            good.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(good[0].keys()))
        writer.writeheader()
        writer.writerows(good)

    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["date", "category", "description", "amount"]
        )
        writer.writeheader()
        writer.writerows(rejected)

    print(f"clean rows:    {len(good)}")
    print(f"rejected rows: {len(rejected)}")
    print(f"wrote {output.as_posix()}")
    print(f"wrote {log.as_posix()}")


if __name__ == "__main__":
    main()
