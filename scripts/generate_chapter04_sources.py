#!/usr/bin/env python3
"""Generate the source files that Chapter 4 reads.

Everything here is deterministic: the same run always produces the same bytes
(except the timestamps inside the manifest examples, which the chapter reads
from the files themselves). Run from the project root:

    python scripts/generate_chapter04_sources.py

Outputs (all under data/raw/):

    sales_exports/sales_2024_01.csv ... sales_2024_12.csv   12 monthly, drifting schemas
    eu_orders.csv                     semicolon + decimal comma, cp1252 accents
    contacts_utf8_bom.csv             UTF-8 with a byte-order mark
    kestrel_q1_2026.xlsx              title rows, merged cells, a formula
    campaigns.jsonl                   JSON Lines
    store_payload.json                nested JSON
    kestrel.db                        SQLite: orders, customers, products, stores
    legacy_products.txt               fixed-width text
    store_orders.xml                  small XML document
    exports_archive.zip               three CSVs inside one archive
    catalogue/page_1.html             scrapable list pages
    catalogue/page_2.html
    catalogue/product_<id>.html       detail pages
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

CATEGORY_NAMES = {
    "Camping": ["Ridge Tent", "Trail Lantern", "Sleeping Bag", "Camp Stove", "Hiking Pole"],
    "Footwear": ["Trail Runner", "Summit Boot", "River Sandal", "Trail Shoe", "Alpine Boot"],
    "Apparel": ["Alpine Jacket", "Storm Shell", "Base Layer", "Fleece Vest", "Rain Pant"],
    "Accessories": ["Daypack", "Water Bottle", "Headlamp", "Trek Pole", "Dry Bag"],
}
REGIONS = ["Midwest", "Northeast", "South", "West"]


# ---------------------------------------------------------------- monthly CSVs
def monthly_exports(products: pd.DataFrame, rng: np.random.Generator) -> None:
    """Write 12 monthly CSVs whose columns drift over the year."""
    out = RAW / "sales_exports"
    out.mkdir(parents=True, exist_ok=True)
    skus = products["product_id"].tolist()
    price_lookup = dict(zip(products["product_id"], products["list_price"]))
    months = pd.period_range("2024-01", "2024-12", freq="M")
    top = 50000

    for period in months:
        n = int(rng.integers(180, 240))
        rows = []
        for i in range(n):
            sku = skus[int(rng.integers(0, len(skus)))]
            qty = int(rng.integers(1, 5))
            price = round(float(price_lookup[sku]) * float(rng.uniform(0.9, 1.1)), 2)
            day = int(rng.integers(1, 28))
            rows.append(
                {
                    "order_id": f"O-{period.year}{period.month:02d}{top + i:06d}",
                    "order_date": f"{period.year}-{period.month:02d}-{day:02d}",
                    "sku": sku,
                    "quantity": qty,
                    "price": price,
                    "region": REGIONS[int(rng.integers(0, len(REGIONS)))],
                    "customer_id": f"C-{int(rng.integers(1, 42869)):06d}",
                    "discount_pct": float(rng.choice([0.0, 5.0, 10.0, 15.0, 20.0])),
                }
            )
        frame = pd.DataFrame(rows)
        m = period.month
        if m <= 3:  # earliest layout
            frame = frame.rename(columns={"quantity": "qty"})
            frame = frame[["order_id", "order_date", "sku", "qty", "price", "region", "customer_id"]]
            frame = frame.rename(columns={"customer_id": "customer"})
            frame = frame[["order_id", "order_date", "sku", "qty", "price", "customer"]]
        elif m <= 6:  # renamed qty -> quantity, customer -> customer_id
            frame = frame[["order_id", "order_date", "sku", "quantity", "price", "customer_id"]]
        elif m <= 9:  # price -> unit_price, region restored
            frame = frame.rename(columns={"price": "unit_price"})
            frame = frame[["order_id", "order_date", "sku", "quantity", "unit_price", "customer_id", "region"]]
        else:  # reordered, adds discount_pct
            frame = frame.rename(columns={"price": "unit_price"})
            frame = frame[["order_date", "order_id", "sku", "quantity", "unit_price", "customer_id", "region", "discount_pct"]]
        frame.to_csv(out / f"sales_{period.year}_{period.month:02d}.csv", index=False)


# --------------------------------------------------------------- odd CSV files
def eu_orders() -> None:
    text = (
        "order_id;order_date;product;quantity;unit_price;total\n"
        "EU-1001;05.01.2026;Ridge Tent;1;249,00;249,00\n"
        "EU-1002;05.01.2026;Trail Lantern;2;39,50;79,00\n"
        "EU-1003;06.01.2026;Summit Boot;1;189,90;189,90\n"
        "EU-1004;07.01.2026;Base Layer;3;24,75;74,25\n"
        "EU-1005;08.01.2026;Headlamp;1;32,00;32,00\n"
    )
    # cp1252 so the accented warehouse names matter.
    (RAW / "eu_orders.csv").write_bytes(
        ("warehouse;order_id;order_date;product;quantity;unit_price;total;note\n"
         "München (HQ);EU-1001;05.01.2026;Ridge Tent;1;249,00;249,00;priority\n"
         "Zürich;EU-1002;05.01.2026;Trail Lantern;2;39,50;79,00;\n"
         "Wien;EU-1003;06.01.2026;Summit Boot;1;189,90;189,90;gift wrap\n"
         "Genève;EU-1004;07.01.2026;Base Layer;3;24,75;74,25;\n"
         "München (HQ);EU-1005;08.01.2026;Headlamp;1;32,00;32,00;backorder\n").encode("cp1252")
    )


def bom_contacts() -> None:
    frame = pd.DataFrame(
        {
            "name": ["Ada Lovelace", "Émile Zola", "Zoë Bell"],
            "email": ["ada@example.com", "emile@example.fr", "zoe@example.com"],
            "note": ["vip", "reseller", ""],
        }
    )
    frame.to_csv(RAW / "contacts_utf8_bom.csv", index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------ Excel file
def excel_report() -> None:
    import xlsxwriter

    path = RAW / "kestrel_q1_2026.xlsx"
    book = xlsxwriter.Workbook(str(path))
    title = book.add_format({"bold": True, "font_size": 14})
    header = book.add_format({"bold": True, "bg_color": "#DDEBF7", "border": 1})
    money = book.add_format({"num_format": "$#,##0.00"})

    notes = book.add_worksheet("Notes")
    notes.write("A1", "Export notes", title)
    notes.write("A3", "Region totals only; line detail lives in the orders table.")
    notes.write("A4", "Prepared by the finance team, 2026-04-02.")

    sheet = book.add_worksheet("Q1 Sales")
    sheet.merge_range("A1:F1", "Kestrel Outfitters - Q1 2026 Sales Extract", title)
    # row 2 is intentionally blank; the header is on row 3 (index 2).
    cols = ["region", "orders", "units", "net_revenue", "returns", "return_rate"]
    for c, name in enumerate(cols):
        sheet.write(2, c, name, header)
    data = [
        ("Midwest", 3205, 7344, 562785.63, 612, 0.0791),
        ("Northeast", 3010, 6901, 530972.14, 548, 0.0794),
        ("South", 4102, 9402, 701233.80, 741, 0.0788),
        ("West", 3458, 7901, 602410.55, 657, 0.0831),
    ]
    for r, row in enumerate(data, start=3):
        for c, value in enumerate(row):
            fmt = money if c == 3 else None
            sheet.write(r, c, value, fmt)
    last = 3 + len(data)  # zero-based row after the data
    sheet.write(last + 1, 0, "Total", header)
    sheet.write(last + 1, 2, None)
    sheet.write_formula(last + 1, 3, f"=SUM(D4:D{last})", money, 2397402.12)

    book.close()


# ------------------------------------------------------------------- JSON files
def json_files() -> None:
    campaigns = [
        {"campaign_id": "CMP-01", "name": "Spring Trailhead", "channel": "email", "start": "2026-03-01", "end": "2026-03-21", "budget": 12000.0, "clicks": 8420, "conversions": 311},
        {"campaign_id": "CMP-02", "name": "Summit Boot Launch", "channel": "social", "start": "2026-02-10", "end": "2026-03-02", "budget": 24000.0, "clicks": 15310, "conversions": 402},
        {"campaign_id": "CMP-03", "name": "Return to Camp", "channel": "search", "start": "2026-03-05", "end": "2026-04-01", "budget": 18000.0, "clicks": 11220, "conversions": 508},
        {"campaign_id": "CMP-04", "name": "Loyalty Double Points", "channel": "email", "start": "2026-01-15", "end": "2026-02-14", "budget": 4000.0, "clicks": 6100, "conversions": 260},
        {"campaign_id": "CMP-05", "name": "Weekend Flash", "channel": "social", "start": "2026-03-22", "end": "2026-03-24", "budget": 3500.0, "clicks": 9800, "conversions": 190},
    ]
    with (RAW / "campaigns.jsonl").open("w", encoding="utf-8") as handle:
        for row in campaigns:
            handle.write(json.dumps(row) + "\n")

    payload = {
        "store_id": "S-01",
        "name": "Kestrel Chicago Loop",
        "opened": "2019-04-12",
        "address": {
            "street": "145 W Madison St",
            "city": "Chicago",
            "region": "Midwest",
            "postal": "60602",
        },
        "staff": [
            {"role": "manager", "name": "R. Okafor", "since": "2019-04-12"},
            {"role": "associate", "name": "L. Nguyen", "since": "2021-06-01"},
            {"role": "associate", "name": "D. Petrov", "since": "2023-09-15"},
        ],
        "monthly_net": [
            {"month": "2026-01", "net": 118220.40},
            {"month": "2026-02", "net": 112005.10},
            {"month": "2026-03", "net": 124510.75},
        ],
    }
    (RAW / "store_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ------------------------------------------------------------ fixed-width + XML
def fixed_width_and_xml(products: pd.DataFrame) -> None:
    lines = []
    for _, row in products.iterrows():
        lines.append(
            f"{row['product_id']:<8}{row['product_name']:<24}{row['category']:<12}{row['list_price']:>10.2f}"
        )
    (RAW / "legacy_products.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    xml = ['<?xml version="1.0" encoding="UTF-8"?>', "<orders>"]
    sample = [
        ("O-300001", "2026-01-04", "P-1001", 2, 58.53, "West"),
        ("O-300002", "2026-01-05", "P-2003", 1, 129.00, "South"),
        ("O-300003", "2026-01-05", "P-3002", 3, 39.95, "Midwest"),
    ]
    for order_id, date, sku, qty, price, region in sample:
        xml.append(
            f'  <order id="{order_id}" date="{date}" region="{region}">\n'
            f'    <line sku="{sku}" quantity="{qty}" unit_price="{price}"/>\n'
            f"  </order>"
        )
    xml.append("</orders>")
    (RAW / "store_orders.xml").write_text("\n".join(xml) + "\n", encoding="utf-8")


def archive() -> None:
    path = RAW / "exports_archive.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for region in ["north", "south", "west"]:
            frame = pd.DataFrame(
                {
                    "order_id": [f"O-{region[:1].upper()}-{i:03d}" for i in range(1, 4)],
                    "sku": ["P-1001", "P-2003", "P-3002"],
                    "units": [1, 2, 4],
                    "net_revenue": [58.53, 258.0, 159.8],
                }
            )
            zf.writestr(f"region_{region}.csv", frame.to_csv(index=False))


# --------------------------------------------------------------------- SQLite
def sqlite_db(orders: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame) -> None:
    path = RAW / "kestrel.db"
    if path.exists():
        path.unlink()
    stores = pd.DataFrame(
        {
            "store_id": [f"S-{i:02d}" for i in range(1, 13)],
            "city": [
                "Chicago", "Denver", "Portland", "Austin", "Seattle", "Boston",
                "Atlanta", "Phoenix", "Minneapolis", "Philadelphia", "San Diego", "Nashville",
            ],
            "region": [
                "Midwest", "West", "West", "South", "West", "Northeast",
                "South", "West", "Midwest", "Northeast", "West", "South",
            ],
            "opened": pd.date_range("2016-03-01", periods=12, freq="YS").strftime("%Y-%m-%d"),
        }
    )
    with sqlite3.connect(path) as conn:
        orders.to_sql("orders", conn, index=False)
        customers.to_sql("customers", conn, index=False)
        products.to_sql("products", conn, index=False)
        stores.to_sql("stores", conn, index=False)
        conn.execute("CREATE INDEX idx_orders_channel ON orders(channel)")
        conn.execute("CREATE INDEX idx_orders_date ON orders(order_date)")
    print("  sqlite tables: orders, customers, products, stores")


# --------------------------------------------------------------- HTML catalogue
def catalogue(products: pd.DataFrame, rng: np.random.Generator) -> None:
    out = RAW / "catalogue"
    out.mkdir(parents=True, exist_ok=True)
    picks = products.head(20).reset_index(drop=True)
    ratings = ["4.1", "3.8", "4.6", "4.0", "3.5", "4.8", "2.9", "4.3", "3.9", "4.7",
               "4.2", "3.7", "4.5", "3.6", "4.9", "2.7", "4.4", "3.3", "4.0", "3.8"]
    avail = ["In stock", "In stock", "Low stock", "In stock", "Out of stock"]

    def page_html(number: int, chunk: pd.DataFrame, start: int) -> str:
        items = []
        for j, (_, row) in enumerate(chunk.iterrows()):
            idx = start + j
            items.append(
                f'      <li class="product">\n'
                f'        <a class="title" href="/product/{row["product_id"]}">{row["product_name"]}</a>\n'
                f'        <span class="price" data-currency="USD">${row["list_price"]:.2f}</span>\n'
                f'        <span class="rating">{ratings[idx]}</span>\n'
                f'        <span class="availability">{avail[idx % len(avail)]}</span>\n'
                f"      </li>"
            )
        links = []
        if number > 1:
            links.append(f'<a class="prev" href="/catalogue?page={number - 1}">Previous</a>')
        if number < 2:
            links.append(f'<a class="next" href="/catalogue?page={number + 1}">Next</a>')
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\">"
            f"<title>Catalogue - page {number}</title></head>\n<body>\n"
            f"  <h1>Trailhead Supply - Catalogue (page {number} of 2)</h1>\n"
            "  <ul class=\"catalogue\">\n" + "\n".join(items) + "\n  </ul>\n"
            "  <nav>" + " ".join(links) + "</nav>\n</body>\n</html>\n"
        )

    first, second = picks.iloc[:10], picks.iloc[10:20]
    (out / "page_1.html").write_text(page_html(1, first, 0), encoding="utf-8")
    (out / "page_2.html").write_text(page_html(2, second, 10), encoding="utf-8")

    for idx, row in picks.iterrows():
        detail = (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\">"
            f"<title>{row['product_name']}</title></head>\n<body>\n"
            f"  <h1 class=\"product-title\">{row['product_name']}</h1>\n"
            f"  <p class=\"sku\">{row['product_id']}</p>\n"
            f"  <p class=\"category\">{row['category']}</p>\n"
            f'  <span class="price">${row["list_price"]:.2f}</span>\n'
            f'  <span class="rating">{ratings[idx]}</span>\n'
            f'  <span class="availability">{avail[idx % len(avail)]}</span>\n'
            "  <div class=\"description\">Synthetic catalogue page for the book.</div>\n"
            "</body>\n</html>\n"
        )
        (out / f"product_{row['product_id']}.html").write_text(detail, encoding="utf-8")
    print(f"  catalogue pages: 2 list pages, {len(picks)} detail pages")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260402)

    products = pd.read_csv(RAW / "kestrel_products.csv")
    customers = pd.read_csv(RAW / "kestrel_customers.csv", dtype={"customer_id": "string"})

    orders = pd.read_csv(
        RAW / "kestrel_orders.csv", dtype={"customer_id": "string", "region": "category"}
    )
    orders = orders.drop_duplicates()
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    orders = orders.dropna(subset=["order_date"])
    orders["order_date"] = orders["order_date"].dt.strftime("%Y-%m-%d")
    orders["region"] = orders["region"].astype("object")

    print("generating monthly exports with drifting schemas ...")
    monthly_exports(products, rng)
    print("generating odd-format files ...")
    eu_orders()
    bom_contacts()
    excel_report()
    json_files()
    fixed_width_and_xml(products)
    archive()
    print("generating SQLite database (this loads ~488k rows) ...")
    sqlite_db(orders, customers, products)
    print("generating HTML catalogue ...")
    catalogue(products, rng)
    print("done.")


if __name__ == "__main__":
    main()
