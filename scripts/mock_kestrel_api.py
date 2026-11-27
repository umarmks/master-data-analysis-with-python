#!/usr/bin/env python3
"""A tiny local HTTP server that Chapter 4 talks to.

Starting a real server keeps the API and scraping examples runnable and
deterministic: no external site, no key, no rate limit, and the same bytes every
run. Use it as a context manager:

    from mock_kestrel_api import running_server

    with running_server() as base_url:
        requests.get(f"{base_url}/v1/orders", params={"page": 1})

Routes
------
GET /robots.txt                     disallow rules for the scraper
GET /catalogue?page=1|2             list pages (served from data/raw/catalogue)
GET /product/<id>                   detail pages
GET /v1/orders?page=&per_page=      offset pagination
GET /v1/orders/cursor?cursor=       opaque cursor pagination
GET /v1/secret                      requires Authorization: Bearer demo-token
GET /flaky                          429 twice, then 200 (for retry/backoff)
"""

from __future__ import annotations

import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "data" / "raw" / "catalogue"

REGIONS = ["Midwest", "Northeast", "South", "West"]
CHANNELS = ["Web", "App", "Phone"]


def _orders() -> list[dict]:
    """Twenty-three deterministic orders for the pagination examples."""
    rows = []
    for i in range(1, 24):
        rows.append(
            {
                "order_id": f"O-9{i:05d}",
                "order_date": f"2026-02-{(i % 27) + 1:02d}",
                "channel": CHANNELS[i % 3],
                "region": REGIONS[i % 4],
                "units": (i % 4) + 1,
                "net_revenue": round(20.0 + i * 7.35, 2),
            }
        )
    return rows


_ORDERS = _orders()
_ROBOTS = "User-agent: *\nDisallow: /product/\nAllow: /catalogue\n"


class Handler(BaseHTTPRequestHandler):
    server_version = "KestrelMock/1.0"

    def log_message(self, *args) -> None:  # keep the console quiet
        pass

    def _send(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict, headers: dict | None = None) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(status, body, "application/json", headers)

    def _text(self, status: int, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
        self._send(status, text.encode("utf-8"), content_type)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/robots.txt":
            return self._text(200, _ROBOTS)

        if path == "/catalogue":
            page = int(query.get("page", ["1"])[0])
            file = CATALOGUE / f"page_{page}.html"
            if not file.exists():
                return self._text(404, "no such page")
            return self._text(200, file.read_text(encoding="utf-8"), "text/html; charset=utf-8")

        if path == "/product/":
            return self._json(404, {"error": "not found"})

        if path.startswith("/product/"):
            product_id = path.rsplit("/", 1)[-1]
            file = CATALOGUE / f"product_{product_id}.html"
            if not file.exists():
                return self._text(404, "no such product")
            return self._text(200, file.read_text(encoding="utf-8"), "text/html; charset=utf-8")

        if path.startswith("/exports/"):
            name = path.rsplit("/", 1)[-1]
            file = (ROOT / "data" / "raw" / "sales_exports" / name).resolve()
            if file.parent != (ROOT / "data" / "raw" / "sales_exports").resolve() or not file.exists():
                return self._json(404, {"error": "not found", "name": name})
            return self._send(
                200, file.read_bytes(), "text/csv; charset=utf-8"
            )

        if path == "/v1/orders":
            page = int(query.get("page", ["1"])[0])
            per_page = int(query.get("per_page", ["10"])[0])
            start = (page - 1) * per_page
            chunk = _ORDERS[start : start + per_page]
            total_pages = (len(_ORDERS) + per_page - 1) // per_page
            nxt = page + 1 if page < total_pages else None
            return self._json(
                200,
                {
                    "page": page,
                    "per_page": per_page,
                    "total": len(_ORDERS),
                    "next": nxt,
                    "results": chunk,
                },
                headers={
                    "X-RateLimit-Limit": "60",
                    "X-RateLimit-Remaining": str(60 - page),
                    "X-RateLimit-Reset": "60",
                },
            )

        if path == "/v1/orders/cursor":
            cursor = query.get("cursor", [None])[0]
            per_page = int(query.get("per_page", ["8"])[0])
            start = int(cursor) if cursor else 0
            chunk = _ORDERS[start : start + per_page]
            nxt = start + per_page
            return self._json(
                200,
                {
                    "results": chunk,
                    "next_cursor": str(nxt) if nxt < len(_ORDERS) else None,
                },
            )

        if path == "/v1/secret":
            if self.headers.get("Authorization") != "Bearer demo-token":
                return self._json(401, {"error": "unauthorized"})
            return self._json(200, {"ok": True, "scope": "orders:read"})

        if path == "/flaky":
            self.server.flaky_hits += 1
            if self.server.flaky_hits <= 2:
                return self._json(
                    429, {"error": "rate limited"}, headers={"Retry-After": "0"}
                )
            return self._json(200, {"ok": True, "attempt": self.server.flaky_hits})

        return self._json(404, {"error": "not found", "path": path})

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urlparse(self.path)
        if parsed.path == "/v1/echo":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body or b"{}")
            return self._json(
                200,
                {"received": payload, "count": len(payload.get("orders", []))},
            )
        return self._json(404, {"error": "not found", "path": parsed.path})


@contextlib.contextmanager
def running_server():
    """Yield the base URL of a local mock server, then shut it down."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.flaky_hits = 0  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


if __name__ == "__main__":
    with running_server() as base:
        print("mock server running at", base)
        input("press Enter to stop ...")
