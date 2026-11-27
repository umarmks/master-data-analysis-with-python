#!/usr/bin/env python3
"""Render a plain-text expense summary from a JSON configuration file.

Example:

    python scripts/weekly_report.py --config config/report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kestrel.metrics import parse_amount  # noqa: E402


def load_rows(path: Path) -> tuple[list[dict], int]:
    """Return (clean, rejected_count) for an expenses file."""
    clean, rejected = [], 0
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            amount = parse_amount(row["amount"])
            if amount is None:
                rejected += 1
                continue
            row["amount"] = amount
            row["category"] = row["category"].strip().title()
            clean.append(row)
    return clean, rejected


def dedupe(rows: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    unique: list[dict] = []
    for row in rows:
        key = (row["date"], row["category"], row["description"], row["amount"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def render(config: dict, rows: list[dict], rejected: int) -> str:
    monthly: dict[str, float] = defaultdict(float)
    by_category: dict[str, float] = defaultdict(float)
    for row in rows:
        monthly[row["date"][:7]] += row["amount"]
        by_category[row["category"]] += row["amount"]

    lines = [
        config["title"],
        "=" * len(config["title"]),
        f"Rows read:    {len(rows) + rejected}",
        f"Rows kept:    {len(rows)}",
        f"Rejected:     {rejected}",
        "",
        "Monthly totals",
        "-" * 14,
    ]
    for month in sorted(monthly):
        lines.append(f"{month}  ${monthly[month]:,.2f}")
    lines += ["", "Category totals", "-" * 15]
    for category, total in sorted(by_category.items(), key=lambda kv: kv[1]):
        lines.append(f"{category:<14} ${total:,.2f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/report.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows, rejected = load_rows(ROOT / config["input"])
    rows = dedupe(rows)
    report = render(config, rows, rejected)

    output = ROOT / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
