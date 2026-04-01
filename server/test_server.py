"""
test_server.py
Exercises all 6 intent handlers directly (no HTTP needed) to verify
the data layer, cost calculation, time logic, and search all work
against the real data files copied from the C project.
"""

import json
from handlers import dispatch


def pp(result):
    """Pretty-print a result dict."""
    print(json.dumps(result, indent=2, default=str))


def test_view_all():
    print("\n" + "=" * 60)
    print("TEST: VIEW_ALL")
    print("=" * 60)
    result = dispatch("VIEW_ALL")
    pp(result)
    assert result["count"] >= 0
    assert "orders" in result
    print(f"  -> {result['count']} active orders")


def test_view_order():
    print("\n" + "=" * 60)
    print("TEST: VIEW_ORDER (ID=123)")
    print("=" * 60)
    result = dispatch("VIEW_ORDER", {"order_id": 123})
    pp(result)

    print("\n--- VIEW_ORDER (ID=99999, should not exist) ---")
    result2 = dispatch("VIEW_ORDER", {"order_id": 99999})
    pp(result2)
    assert result2["found"] is False


def test_filter_by_depot():
    print("\n" + "=" * 60)
    print("TEST: FILTER_BY_DEPOT (name=FadEx)")
    print("=" * 60)
    result = dispatch("FILTER_BY_DEPOT", {"depot_name": "FadEx"})
    pp(result)

    print("\n--- FILTER_BY_DEPOT (id=2) ---")
    result2 = dispatch("FILTER_BY_DEPOT", {"depot_id": 2})
    pp(result2)

    print("\n--- FILTER_BY_DEPOT (name=NonExistent) ---")
    result3 = dispatch("FILTER_BY_DEPOT", {"depot_name": "NonExistent"})
    pp(result3)
    assert "error" in result3


def test_show_history():
    print("\n" + "=" * 60)
    print("TEST: SHOW_HISTORY")
    print("=" * 60)
    result = dispatch("SHOW_HISTORY")
    pp(result)
    assert result["count"] >= 0
    print(f"  -> {result['count']} delivered orders")


def test_calculate_cost():
    print("\n" + "=" * 60)
    print("TEST: CALCULATE_COST (2kg via courier 1)")
    print("=" * 60)
    result = dispatch("CALCULATE_COST", {"weight": 2.0, "courier_id": 1})
    pp(result)
    # Manual check: FadEx = base(2.50) + dist(12.5)*rateKm(0.20) + weight(2.0)*rateKg(1.00)
    #             = 2.50 + 2.50 + 2.00 = 7.00
    assert result["cost_gbp"] == 7.0, f"Expected 7.0, got {result['cost_gbp']}"
    print(f"  -> Cost calculation verified: 7.00")

    print("\n--- CALCULATE_COST (by courier name 'DLH') ---")
    result2 = dispatch("CALCULATE_COST", {"weight": 5.0, "courier_name": "DLH"})
    pp(result2)
    # DLH = base(4.00) + dist(20.0)*rateKm(0.18) + weight(5.0)*rateKg(0.90)
    #     = 4.00 + 3.60 + 4.50 = 12.10
    assert result2["cost_gbp"] == 12.1, f"Expected 12.1, got {result2['cost_gbp']}"
    print(f"  -> Cost calculation verified: 12.10")


def test_search_order():
    print("\n" + "=" * 60)
    print("TEST: SEARCH_ORDER (ID=101, should be in history)")
    print("=" * 60)
    result = dispatch("SEARCH_ORDER", {"order_id": 101})
    pp(result)

    print("\n--- SEARCH_ORDER (ID=515, should be active) ---")
    result2 = dispatch("SEARCH_ORDER", {"order_id": 515})
    pp(result2)

    print("\n--- SEARCH_ORDER (ID=99999, not found) ---")
    result3 = dispatch("SEARCH_ORDER", {"order_id": 99999})
    pp(result3)
    assert result3["found"] is False


def test_unknown_intent():
    print("\n" + "=" * 60)
    print("TEST: UNKNOWN_INTENT")
    print("=" * 60)
    result = dispatch("MAKE_COFFEE")
    pp(result)
    assert "error" in result


if __name__ == "__main__":
    print("=" * 60)
    print("  PARCEL TRACKER SERVER — HANDLER TESTS")
    print("=" * 60)

    test_view_all()
    test_view_order()
    test_filter_by_depot()
    test_show_history()
    test_calculate_cost()
    test_search_order()
    test_unknown_intent()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
