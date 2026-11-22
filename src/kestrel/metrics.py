"""Reusable revenue helpers for the Kestrel case study."""

from __future__ import annotations


def parse_amount(raw: str) -> float | None:
    """Parse a currency string such as '  $4,250.75 ' into 4250.75.

    Returns ``None`` for values that cannot be parsed, so callers can decide
    whether to reject the row or record it for review.
    """
    if raw is None:
        return None
    try:
        return float(raw.strip().replace("$", "").replace(",", ""))
    except (AttributeError, ValueError):
        return None


def net_revenue(
    revenue: float, returned: int = 0, shipping_cost: float = 0.0
) -> float:
    """Return revenue after returns and shipping for one order line."""
    return round(revenue * (1 - returned) - shipping_cost, 2)


def average_order_value(net_total: float, orders: int) -> float | None:
    """Return net revenue per order, or ``None`` when there are no orders."""
    if orders == 0:
        return None
    return net_total / orders
