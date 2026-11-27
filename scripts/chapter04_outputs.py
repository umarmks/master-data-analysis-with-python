#!/usr/bin/env python3
"""Print every output block used in Chapter 4, from the current files.

Run from the project root:

    python scripts/chapter04_outputs.py

The chapter's API and scraping examples talk to the local mock server in
scripts/mock_kestrel_api.py, so they are deterministic and offline. The one real
network call (Open-Meteo) is cached under data/raw/weather on first run.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import shutil
import sqlite3
import time
import zipfile
from pathlib import Path
from urllib.parse import urlencode, urljoin

import pandas as pd
import requests

from mock_kestrel_api import running_server

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 14)
pd.set_option("display.max_rows", 40)

EXPORTS = RAW / "sales_exports"


def show(label: str, value: object) -> None:
    print(f"### {label}")
    print(value)
    print()


# ----------------------------------------------------------- 4.1 strategy
def section_41() -> None:
    card = pd.DataFrame(
        [
            ("kestrel_orders", "orders API", "one row per order line", "2024-01-01", "daily 06:00", "external", "order_id+product_id"),
            ("kestrel_customers", "CRM export", "one row per customer", "all time", "weekly Mon", "external", "customer_id"),
            ("kestrel_products", "PIM export", "one row per SKU", "current", "on change", "internal", "product_id"),
            ("ad_spend", "ad platform API", "one row per channel-day", "2024-01-01", "daily 09:00", "external", "channel+date"),
            ("finance_total", "finance system", "one row per month", "2024-01-01", "monthly close", "internal", "month"),
        ],
        columns=["dataset", "source", "grain", "history", "freshness", "owner", "key"],
    )
    show("source inventory", card)
    orders = pd.read_csv(RAW / "kestrel_orders.csv", usecols=["order_id"])
    show(
        "orders vs lines",
        {"lines": len(orders), "distinct orders": int(orders["order_id"].nunique())},
    )


# ------------------------------------------------------- 4.2 files/formats
def section_42() -> None:
    # --- CSV dialects: semicolon + decimal comma + cp1252
    raw_bytes = (RAW / "eu_orders.csv").read_bytes()
    show("eu file: first bytes", raw_bytes[:44])
    eu = pd.read_csv(
        RAW / "eu_orders.csv", sep=";", decimal=",", encoding="cp1252"
    )
    show("eu_orders read", eu)
    show("eu dtypes", eu.dtypes.astype(str).to_dict())

    # --- the cost of guessing the encoding
    try:
        pd.read_csv(RAW / "eu_orders.csv", sep=";", decimal=",", encoding="utf-8")
    except UnicodeDecodeError as exc:
        show("utf-8 fails", f"{type(exc).__name__}: {exc}")

    # --- quoting: embedded commas and quotes
    tricky = 'id,note,amount\n1,"Trail, waterproof",59.00\n2,"He said ""hi""",12.50\n'
    parsed = list(csv.DictReader(io.StringIO(tricky)))
    show("quoted csv parsed", parsed)
    show("pandas handles quoting", pd.read_csv(io.StringIO(tricky)))

    # --- BOM: pandas strips it, a hand-rolled reader does not
    bom = (RAW / "contacts_utf8_bom.csv").read_bytes()
    show("bom first bytes", bom[:3])
    with (RAW / "contacts_utf8_bom.csv").open(encoding="utf-8") as handle:
        header_utf8 = next(csv.reader(handle))
    with (RAW / "contacts_utf8_bom.csv").open(encoding="utf-8-sig") as handle:
        header_sig = next(csv.reader(handle))
    show("csv.reader utf-8 header", header_utf8)
    show("csv.reader utf-8-sig header", header_sig)
    show("pandas handles both", list(pd.read_csv(RAW / "contacts_utf8_bom.csv", encoding="utf-8").columns))

    # --- Excel: sheets, header offset, formula cached value
    xlsx = pd.ExcelFile(RAW / "kestrel_q1_2026.xlsx")
    show("sheet names", xlsx.sheet_names)
    titles = pd.read_excel(xlsx, sheet_name="Q1 Sales", header=None, nrows=4)
    show("raw first rows", titles)
    region = pd.read_excel(xlsx, sheet_name="Q1 Sales", skiprows=2)
    show("after skiprows=2", region)
    region = region.dropna(subset=["region"])
    region = region[region["region"] != "Total"]
    show("after tidying", region)

    import openpyxl

    wb = openpyxl.load_workbook(RAW / "kestrel_q1_2026.xlsx", data_only=False)
    with_formula = wb["Q1 Sales"]["D9"].value
    wb2 = openpyxl.load_workbook(RAW / "kestrel_q1_2026.xlsx", data_only=True)
    with_value = wb2["Q1 Sales"]["D9"].value
    show("formula vs cached value", {"data_only=False": with_formula, "data_only=True": with_value})
    show("excel serial to date", pd.to_datetime(45292, unit="D", origin="1899-12-30").date())

    # --- JSON Lines and nested JSON
    campaigns = pd.read_json(RAW / "campaigns.jsonl", lines=True)
    show("jsonl rows", len(campaigns))
    show("jsonl head", campaigns[["campaign_id", "channel", "budget", "conversions"]].head(3))

    payload = json.loads((RAW / "store_payload.json").read_text(encoding="utf-8"))
    show("nested keys", list(payload.keys()))
    flat = pd.json_normalize(
        payload, record_path="staff", meta=["store_id", ["address", "city"]]
    )
    show("json_normalize staff", flat)
    months = pd.json_normalize(payload["monthly_net"])
    show("json_normalize months", months.T.to_dict())

    # --- Parquet: schema, column selection, size
    clean = pd.read_csv(
        RAW / "kestrel_orders.csv", nrows=200000, dtype={"customer_id": "string"}
    )
    clean["order_date"] = pd.to_datetime(clean["order_date"], errors="coerce")
    clean = clean.dropna(subset=["order_date"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    csv_path = PROCESSED / "_ch04_slice.csv"
    parquet_path = PROCESSED / "_ch04_slice.parquet"
    clean.to_csv(csv_path, index=False)
    clean.to_parquet(parquet_path, index=False)
    show(
        "slice sizes (bytes)",
        {"rows": len(clean), "csv": csv_path.stat().st_size, "parquet": parquet_path.stat().st_size},
    )
    schema = pd.read_parquet(parquet_path).dtypes.astype(str)
    show("parquet schema (tail)", schema.tail(4).to_dict())
    picked = pd.read_parquet(parquet_path, columns=["order_id", "net" if "net" in schema else "revenue"])
    show("parquet column read", list(picked.columns))
    csv_path.unlink()
    parquet_path.unlink()

    # --- fixed width
    widths = [8, 24, 12, 10]
    fixed = pd.read_fwf(RAW / "legacy_products.txt", widths=widths, header=None,
                        names=["product_id", "product_name", "category", "list_price"])
    show("fixed width head", fixed.head(3))

    # --- fixed width read flat, to see why widths matter
    flat = pd.read_csv(RAW / "legacy_products.txt", header=None)
    show("flat read of fixed-width", flat.head(2))

    # --- XML
    import xml.etree.ElementTree as ET

    tree = ET.parse(RAW / "store_orders.xml")
    root = tree.getroot()
    rows = []
    for order in root.findall("order"):
        for line in order.findall("line"):
            rows.append(
                {
                    "order_id": order.get("id"),
                    "date": order.get("date"),
                    "sku": line.get("sku"),
                    "quantity": int(line.get("quantity")),
                    "unit_price": float(line.get("unit_price")),
                }
            )
    show("xml rows", pd.DataFrame(rows))

    # --- archive
    with zipfile.ZipFile(RAW / "exports_archive.zip") as zf:
        names = zf.namelist()
        members = []
        for name in names:
            with zf.open(name) as handle:
                members.append(pd.read_csv(handle).assign(source=name))
    members = pd.concat(members, ignore_index=True)
    show("archive members", names)
    show("archive rows", len(members))
    show("archive totals", members.groupby("source")["net_revenue"].sum().round(2))


# -------------------------------------------------------- 4.3 databases
def section_43() -> None:
    con = sqlite3.connect(RAW / "kestrel.db")
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", con
    )
    show("tables", tables["name"].tolist())
    show(
        "orders schema",
        pd.read_sql("PRAGMA table_info(orders)", con)[["name", "type"]].head(6),
    )
    show(
        "row counts",
        pd.read_sql(
            "SELECT 'orders' AS t, COUNT(*) AS n FROM orders "
            "UNION ALL SELECT 'customers', COUNT(*) FROM customers "
            "UNION ALL SELECT 'products', COUNT(*) FROM products "
            "UNION ALL SELECT 'stores', COUNT(*) FROM stores",
            con,
        ),
    )
    q1 = pd.read_sql(
        "SELECT channel, COUNT(*) AS lines, ROUND(SUM(revenue), 2) AS gross "
        "FROM orders WHERE order_date >= '2026-01-01' AND order_date <= '2026-03-31' "
        "GROUP BY channel ORDER BY gross DESC",
        con,
    )
    show("filtered aggregate in SQL", q1)
    con.close()

    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{RAW / 'kestrel.db'}")
    top = pd.read_sql(
        "SELECT region, COUNT(DISTINCT order_id) AS orders "
        "FROM orders GROUP BY region ORDER BY orders DESC",
        engine,
    )
    show("sqlalchemy read", top)

    # write back a small table
    summary = q1.rename(columns={"lines": "line_count"})
    summary.to_sql("q1_channel_summary", engine, if_exists="replace", index=False)
    back = pd.read_sql("SELECT * FROM q1_channel_summary", engine)
    show("to_sql round trip", list(back["channel"]))
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE q1_channel_summary")

    # DuckDB over Parquet (generated on the fly)
    import duckdb

    slice_path = PROCESSED / "_ch04_duck.parquet"
    pd.read_csv(RAW / "kestrel_orders.csv", nrows=100000).to_parquet(slice_path, index=False)
    duck = duckdb.query(
        f"SELECT channel, ROUND(SUM(revenue),2) AS gross "
        f"FROM read_parquet('{slice_path.as_posix()}') GROUP BY channel ORDER BY gross DESC"
    ).to_df()
    show("duckdb over parquet", duck)
    slice_path.unlink()


# -------------------------------------------------------------- 4.4 APIs
def section_44() -> None:
    with running_server() as base:
        resp = requests.get(f"{base}/v1/orders", params={"page": 1, "per_page": 10}, timeout=5)
        show("status and headers", {"status": resp.status_code, "content-type": resp.headers["Content-Type"]})
        body = resp.json()
        show("page 1 envelope", {k: body[k] for k in ["page", "per_page", "total", "next"]})
        show("page 1 rows", pd.DataFrame(body["results"]).head(3))

        # error handling
        missing = requests.get(f"{base}/v1/nope", timeout=5)
        show("404 handling", {"status": missing.status_code, "body": missing.json()})

        # auth
        anon = requests.get(f"{base}/v1/secret", timeout=5)
        authed = requests.get(
            f"{base}/v1/secret", headers={"Authorization": "Bearer demo-token"}, timeout=5
        )
        show("auth", {"without": anon.status_code, "with": authed.status_code, "body": authed.json()})

        # offset pagination
        collected, page = [], 1
        while True:
            r = requests.get(f"{base}/v1/orders", params={"page": page, "per_page": 10}, timeout=5)
            r.raise_for_status()
            data = r.json()
            collected.extend(data["results"])
            if data["next"] is None:
                break
            page = data["next"]
        show("offset pagination", {"pages": page, "rows": len(collected)})

        # cursor pagination
        seen, cursor = [], None
        while True:
            params = {"per_page": 8}
            if cursor:
                params["cursor"] = cursor
            data = requests.get(f"{base}/v1/orders/cursor", params=params, timeout=5).json()
            seen.extend(data["results"])
            cursor = data["next_cursor"]
            if cursor is None:
                break
        show("cursor pagination", {"rows": len(seen), "distinct": len({r['order_id'] for r in seen})})

        # retry with backoff
        attempts = []
        for attempt in range(1, 5):
            r = requests.get(f"{base}/flaky", timeout=5)
            attempts.append(r.status_code)
            if r.status_code == 200:
                break
            time.sleep(0.05 * 2 ** (attempt - 1))
        show("retry sequence", attempts)

        # session reuse
        with requests.Session() as session:
            session.headers["User-Agent"] = "kestrel-book/1.0"
            codes = [
                session.get(f"{base}/v1/orders", params={"page": p}, timeout=5).status_code
                for p in (1, 2)
            ]
        show("session statuses", codes)

    # nested API response flattened
    snippet = {
        "data": [
            {"id": "S-01", "name": "Chicago Loop", "address": {"city": "Chicago", "region": "Midwest"}},
            {"id": "S-02", "name": "Denver Union", "address": {"city": "Denver", "region": "West"}},
        ]
    }
    show("flattened data[]", pd.json_normalize(snippet["data"]))

    # real API, cached
    weather = cached_weather()
    daily = pd.DataFrame(weather["daily"])
    show("open-meteo cached", daily.head(3))
    show("open-meteo dates", daily["time"].tolist())


def cached_weather() -> dict:
    cache = RAW / "weather" / "open_meteo_chicago.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": 41.88,
            "longitude": -87.63,
            "start_date": "2026-01-01",
            "end_date": "2026-01-05",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "America/Chicago",
        },
        timeout=30,
        headers={"User-Agent": "kestrel-book/1.0"},
    )
    resp.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text, encoding="utf-8")
    return resp.json()


# ---------------------------------------------------------- 4.5 scraping
def section_45() -> None:
    from bs4 import BeautifulSoup

    with running_server() as base:
        # robots.txt
        robots = requests.get(f"{base}/robots.txt", timeout=5).text
        show("robots.txt", robots.strip())

        # list page
        html = requests.get(f"{base}/catalogue", params={"page": 1}, timeout=5).text
        soup = BeautifulSoup(html, "lxml")
        first = soup.select_one("li.product")
        show("first card html", str(first).strip())

        items = []
        for card in soup.select("li.product"):
            items.append(
                {
                    "title": card.select_one("a.title").get_text(strip=True),
                    "href": card.select_one("a.title")["href"],
                    "price": float(card.select_one("span.price").get_text(strip=True).lstrip("$")),
                    "rating": float(card.select_one("span.rating").get_text(strip=True)),
                    "availability": card.select_one("span.availability").get_text(strip=True),
                }
            )
        show("page 1 items", pd.DataFrame(items)[["title", "price", "rating"]].head(3))
        show("page 1 count", len(items))

        # walk both pages
        all_items = []
        for number in (1, 2):
            page_html = requests.get(f"{base}/catalogue", params={"page": number}, timeout=5).text
            for card in BeautifulSoup(page_html, "lxml").select("li.product"):
                all_items.append(
                    {
                        "title": card.select_one("a.title").get_text(strip=True),
                        "href": card.select_one("a.title")["href"],
                        "price": float(card.select_one("span.price").get_text(strip=True).lstrip("$")),
                        "availability": card.select_one("span.availability").get_text(strip=True),
                    }
                )
            time.sleep(0.05)  # be polite between list pages
        frames = pd.DataFrame(all_items)
        show("catalogue total", len(frames))
        show("availability mix", frames["availability"].value_counts().to_dict())

        # follow three detail links politely
        details = []
        for href in frames["href"].head(3):
            detail = BeautifulSoup(
                requests.get(f"{base}{href}", timeout=5).text, "lxml"
            )
            details.append(
                {
                    "sku": detail.select_one("p.sku").get_text(strip=True),
                    "title": detail.select_one("h1.product-title").get_text(strip=True),
                    "category": detail.select_one("p.category").get_text(strip=True),
                    "price": detail.select_one("span.price").get_text(strip=True),
                }
            )
            time.sleep(0.05)
        show("detail pages", pd.DataFrame(details))


# -------------------------------------------------------- 4.6 cloud/SaaS
def section_46() -> None:
    options = pd.DataFrame(
        [
            ("S3 bucket (Parquet)", "large daily files", "high volume, typed", "needs credentials + storage cost"),
            ("Google Sheets export", "shared manual sheet", "zero setup", "breaks on layout change"),
            ("CRM nightly export", "stable schema, scheduled", "simple, auditable", "up to a day stale"),
            ("Warehouse table", "already modeled", "fast, governed", "needs warehouse access"),
        ],
        columns=["source", "when_to_use", "upside", "hidden_cost"],
    )
    show("source options", options)

    # The source of a CSV does not change how you read it: pandas accepts a
    # path, an open file, a buffer, or bytes fetched over HTTP.
    local = pd.read_csv(RAW / "sales_exports" / "sales_2024_12.csv", nrows=3)
    show("from a path", local[["order_id", "sku", "quantity"]].to_dict("records"))

    with running_server() as base:
        resp = requests.get(f"{base}/exports/sales_2024_12.csv", timeout=5)
        resp.raise_for_status()
        downloaded = pd.read_csv(io.BytesIO(resp.content), nrows=3)
    show("from fetched bytes", downloaded[["order_id", "sku", "quantity"]].to_dict("records"))
    show("same shape", (local.shape, downloaded.shape))


# -------------------------------------------------- 4.7 automation/provenance
def section_47() -> None:
    manifest = []
    for path in sorted(EXPORTS.glob("sales_*.csv")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        with path.open("r", encoding="utf-8") as handle:
            rows = sum(1 for _ in handle) - 1
        manifest.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "rows": rows,
                "sha256_12": digest,
                "modified": pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
            }
        )
    manifest_df = pd.DataFrame(manifest)
    show("manifest head", manifest_df.head(3))
    show("manifest totals", {"files": len(manifest_df), "rows": int(manifest_df["rows"].sum())})

    out = PROCESSED / "_ch04_manifest.csv"
    manifest_df.to_csv(out, index=False)
    show("manifest written", out.name)
    out.unlink()


# ------------------------------------------------------- 4.8 consolidation
def section_48() -> None:
    columns_by_file = {}
    for path in sorted(EXPORTS.glob("sales_*.csv")):
        columns_by_file[path.name] = list(pd.read_csv(path, nrows=0).columns)
    all_cols = sorted({c for cols in columns_by_file.values() for c in cols})
    drift = pd.DataFrame(
        {c: [int(c in cols) for cols in columns_by_file.values()] for c in all_cols},
        index=list(columns_by_file),
    )
    show("schema by file", drift)

    # reconcile before concatenating
    counts = {
        path.name: sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        for path in sorted(EXPORTS.glob("sales_*.csv"))
    }
    show("row counts", counts)
    show("total rows", sum(counts.values()))

    # harmonize and stack
    rename = {"qty": "quantity", "price": "unit_price", "customer": "customer_id"}
    frames = []
    for path in sorted(EXPORTS.glob("sales_*.csv")):
        frame = pd.read_csv(path)
        frame = frame.rename(columns=rename)
        for col in ["region", "discount_pct"]:
            if col not in frame:
                frame[col] = pd.NA
        frames.append(frame[["order_id", "order_date", "sku", "quantity", "unit_price",
                             "customer_id", "region", "discount_pct"]])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        combined = pd.concat(frames, ignore_index=True)
    combined["order_date"] = pd.to_datetime(combined["order_date"])
    show("combined shape", combined.shape)
    show("combined dtypes", combined.dtypes.astype(str).to_dict())
    show("null share after stack", combined.isna().mean().round(3).to_dict())
    show(
        "by month",
        combined.groupby(combined["order_date"].dt.to_period("M"))["quantity"].sum().head(4),
    )


# ------------------------------------------------------ 4.9 example bank
def section_49() -> None:
    # 1. probe a file before reading it
    raw = (RAW / "eu_orders.csv").read_bytes()
    header = raw.split(b"\n", 1)[0].decode("cp1252")
    counts = {d: header.count(d) for d in [";", ",", "\t", "|"]}
    show("probe: delimiter counts", counts)
    show("probe: physical lines", raw.count(b"\n"))
    show("probe: first 3 bytes", raw[:3])

    # 2. read a gzipped csv
    src = EXPORTS / "sales_2024_12.csv"
    gz = PROCESSED / "_ch04_sales12.csv.gz"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, gzip.open(gz, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    back = pd.read_csv(gz)
    show("gzip sizes", {"csv": src.stat().st_size, "gz": gz.stat().st_size})
    show("gzip read shape", back.shape)
    gz.unlink()

    # 3. read a window of rows
    window = pd.read_csv(EXPORTS / "sales_2024_12.csv", skiprows=range(1, 6), nrows=3)
    show("window rows", window[["order_id", "sku", "quantity"]].to_dict("records"))

    # 4. custom na_values and thousands separator
    txt = 'id,amount\n1,"1,234.56"\n2,N/A\n3,-\n'
    parsed = pd.read_csv(io.StringIO(txt), na_values=["N/A", "-"], thousands=",")
    show("custom na", parsed.to_dict("records"))

    # 5. build urls safely
    show("joined url", urljoin("https://api.example.com/v1/", "orders"))
    show("encoded query", "?" + urlencode({"page": 1, "per_page": 50, "region": "Mid west"}))

    # 6. stream a large download to disk
    with running_server() as base:
        with requests.get(f"{base}/exports/sales_2024_12.csv", stream=True, timeout=5) as resp:
            total = 0
            for chunk in resp.iter_content(chunk_size=4096):
                total += len(chunk)
    show("streamed bytes", total)

    # 7. verify a download with a checksum sidecar
    path = EXPORTS / "sales_2024_12.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = PROCESSED / "sales_2024_12.sha256"
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    line = sidecar.read_text(encoding="utf-8").strip()
    show("sidecar line", line)
    show("verify", line.split()[0] == hashlib.sha256(path.read_bytes()).hexdigest())
    sidecar.unlink()

    # 8. timezone-aware timestamps
    stamps = pd.Series(["2026-03-01T09:00:00-06:00", "2026-03-01T15:00:00Z"])
    ts = pd.to_datetime(stamps, utc=True)
    show("same instant in Chicago", ts.dt.tz_convert("America/Chicago").dt.strftime("%Y-%m-%d %H:%M %Z").tolist())

    # 9. ragged records align to the union of keys
    records = [{"id": "A", "x": 1}, {"id": "B", "y": 2}, {"id": "C", "x": 3, "y": 4}]
    ragged = pd.DataFrame(records)
    show("ragged records", ragged.to_dict("records"))
    show("ragged nulls", ragged.isna().sum().to_dict())

    # 10. a retry policy via HTTPAdapter
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=retry))
    with running_server() as base:
        r = session.get(f"{base}/flaky", timeout=5)
    show("adapter retried to", r.status_code)


# ---------------------------------------------------------- 4.10 capstone
def section_410() -> None:
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{RAW / 'kestrel.db'}")
    q1_region = pd.read_sql(
        "SELECT region, COUNT(DISTINCT order_id) AS orders, "
        "ROUND(SUM(revenue), 2) AS net "
        "FROM orders WHERE order_date BETWEEN '2026-01-01' AND '2026-03-31' "
        "GROUP BY region ORDER BY net DESC",
        engine,
    )
    show("capstone: Q1 net by region", q1_region)

    stores = pd.read_sql(
        "SELECT region, COUNT(*) AS stores FROM stores GROUP BY region", engine
    )
    plan = q1_region.merge(stores, on="region", how="left")
    plan["net_per_store"] = (plan["net"] / plan["stores"]).round(0)
    show("capstone: with store counts", plan)

    out = PROCESSED / "q1_2026_region.parquet"
    plan.to_parquet(out, index=False)
    back = pd.read_parquet(out)
    show("capstone: curated shape", back.shape)
    show("capstone: curated path", out.name)
    out.unlink()


def section_411() -> None:
    # read an HTML <table> with read_html
    table_html = """
    <table>
      <tr><th>region</th><th>orders</th><th>net</th></tr>
      <tr><td>South</td><td>7270</td><td>1482686.05</td></tr>
      <tr><td>Midwest</td><td>5284</td><td>1079606.86</td></tr>
    </table>
    """
    tables = pd.read_html(io.StringIO(table_html))
    show("read_html tables", len(tables))
    show("read_html first", tables[0].to_dict("records"))

    # parse an object-storage uri
    from urllib.parse import urlparse

    uri = "s3://kestrel-datalake/curated/orders/year=2026/month=01/part-0000.parquet"
    parsed = urlparse(uri)
    show("s3 bucket", parsed.netloc)
    show("s3 key", parsed.path.lstrip("/"))

    # an incremental watermark checkpoint
    checkpoint = PROCESSED / "_ch04_watermark.json"
    checkpoint.write_text(
        json.dumps({"updated_since": "2026-03-01T00:00:00Z", "last_run": "2026-03-02T06:00:00Z"}),
        encoding="utf-8",
    )
    show("checkpoint", json.loads(checkpoint.read_text(encoding="utf-8")))
    checkpoint.unlink()

    # read rate-limit headers from a real response
    with running_server() as base:
        resp = requests.get(f"{base}/v1/orders", params={"page": 1, "per_page": 10}, timeout=5)
        limits = {
            k: v for k, v in resp.headers.items() if k.lower().startswith("x-ratelimit")
        }
    show("rate limit headers", limits)


def section_412() -> None:
    # read every sheet of a workbook
    xlsx = pd.ExcelFile(RAW / "kestrel_q1_2026.xlsx")
    sheets = {
        name: pd.read_excel(xlsx, sheet_name=name, header=None)
        for name in xlsx.sheet_names
    }
    show("sheet shapes", {name: frame.shape for name, frame in sheets.items()})
    show("sheet heads", {name: list(frame.iloc[0].dropna().astype(str)) for name, frame in sheets.items()})

    # a POST with a JSON body
    with running_server() as base:
        payload = {"source": "erp", "orders": [{"id": "O-1"}, {"id": "O-2"}]}
        resp = requests.post(f"{base}/v1/echo", json=payload, timeout=5)
        resp.raise_for_status()
        show("post echo", resp.json())

    # read a query from a .sql file
    query = "SELECT channel, COUNT(*) AS n FROM orders GROUP BY channel ORDER BY n DESC"
    sql_file = PROCESSED / "_ch04_channel_counts.sql"
    sql_file.write_text(query, encoding="utf-8")
    con = sqlite3.connect(RAW / "kestrel.db")
    loaded = sql_file.read_text(encoding="utf-8")
    show("query from file", pd.read_sql(loaded, con).to_dict("records"))
    con.close()
    sql_file.unlink()


def section_413() -> None:
    # parse non-ISO dates with an explicit format
    eu = pd.read_csv(RAW / "eu_orders.csv", sep=";", decimal=",", encoding="cp1252")
    eu["order_date"] = pd.to_datetime(eu["order_date"], format="%d.%m.%Y")
    show("eu dates parsed", eu["order_date"].dt.strftime("%Y-%m-%d").tolist())

    # verify a consolidation with assertions
    rename = {"qty": "quantity", "price": "unit_price", "customer": "customer_id"}
    required = ["order_id", "order_date", "sku", "quantity", "unit_price", "customer_id"]
    frames = []
    for path in sorted(EXPORTS.glob("sales_*.csv")):
        frame = pd.read_csv(path).rename(columns=rename)
        for col in ["region", "discount_pct"]:
            if col not in frame:
                frame[col] = pd.NA
        frames.append(frame[required + ["region", "discount_pct"]])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        combined = pd.concat(frames, ignore_index=True)
    combined["order_date"] = pd.to_datetime(combined["order_date"])
    checks = {
        "row_count": len(combined) == 2573,
        "key_unique": bool(combined["order_id"].is_unique),
        "required_complete": bool(combined[required].notna().all().all()),
        "region_null_share": round(float(combined["region"].isna().mean()), 3),
    }
    show("consolidation assertions", checks)


def section_414() -> None:
    from bs4 import BeautifulSoup

    # collect and deduplicate detail links across pages
    with running_server() as base:
        hrefs = []
        for number in (1, 2):
            html = requests.get(f"{base}/catalogue", params={"page": number}, timeout=5).text
            hrefs += [
                a["href"] for a in BeautifulSoup(html, "lxml").select("li.product a.title")
            ]
    unique = sorted(set(hrefs))
    show("links", {"collected": len(hrefs), "unique": len(unique), "first": unique[0]})

    # validate a response against an expected schema
    with running_server() as base:
        body = requests.get(f"{base}/v1/orders", params={"page": 1}, timeout=5).json()
    expected = {"order_id", "order_date", "channel", "region", "units", "net_revenue"}
    got = set(body["results"][0])
    show(
        "schema check",
        {"missing": sorted(expected - got), "unexpected": sorted(got - expected)},
    )


def main() -> None:
    print("python", "pandas", pd.__version__)
    print()
    for section in (
        section_41,
        section_42,
        section_43,
        section_44,
        section_45,
        section_46,
        section_47,
        section_48,
        section_49,
        section_410,
        section_411,
        section_412,
        section_413,
        section_414,
    ):
        print("=" * 70)
        section()


if __name__ == "__main__":
    main()
