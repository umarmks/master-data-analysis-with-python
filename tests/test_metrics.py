from kestrel.metrics import average_order_value, net_revenue, parse_amount


def test_parse_amount_handles_formatting():
    assert parse_amount("  $4,250.75 ") == 4250.75


def test_parse_amount_returns_none_for_junk():
    assert parse_amount("n/a") is None


def test_net_revenue_subtracts_returns_and_shipping():
    assert net_revenue(173.57, 0, 7.95) == 165.62
    assert net_revenue(120.00, 1, 3.95) == -3.95


def test_average_order_value_guards_against_zero_orders():
    assert average_order_value(100.0, 0) is None
    assert average_order_value(370.0, 2) == 185.0


import pytest


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  $4,250.75 ", 4250.75),
        ("128.40", 128.40),
        ("n/a", None),
        ("", None),
    ],
)
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected

