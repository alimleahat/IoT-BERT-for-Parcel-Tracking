"""
handlers.py
One handler function per BERT intent.  Each receives params (dict) and
returns a JSON-serialisable response dict.

Intent → C function mapping:
  VIEW_ALL        → currentOrders()
  VIEW_ORDER      → searchOrder() [active only, by ID]
  FILTER_BY_DEPOT → *new* (not in C code)
  SHOW_HISTORY    → deliveredOrders() + syncDeliveredOrders()
  CALCULATE_COST  → calculateCost()
  SEARCH_ORDER    → searchOrder() [active + history]
"""

from data_layer import load_orders, load_history, load_depots, sync_delivered_orders
from time_utils import get_time_remaining, get_time_since_delivery, convert_to_timestamp
from cost import calculate_cost, get_courier_name, find_depot_by_name, find_depot_by_id


def _enrich_active_order(order, depots):
    """Add time_remaining and courier_name to an active order."""
    time_left, status = get_time_remaining(order["delivery_time"])
    return {
        **order,
        "time_remaining": time_left,
        "computed_status": "delivered" if status == 1 else "active",
        "courier_name": get_courier_name(depots, order["courier"]),
    }


def _enrich_history_order(order, depots):
    """Add time_since and courier_name to a delivered order."""
    return {
        **order,
        "time_since": get_time_since_delivery(order["delivery_time"]),
        "courier_name": get_courier_name(depots, order["courier"]),
    }


# ── VIEW_ALL ─────────────────────────────────────────────────────────────────

def handle_view_all(params):
    """Mirror of C currentOrders(): return all active orders sorted by
    delivery time (earliest first), with time-remaining computed.
    """
    depots = load_depots()
    orders = load_orders()

    # Sort by delivery time (earliest first) — mirrors sortCurrentOrdersByDeliveryTime()
    orders.sort(key=lambda o: convert_to_timestamp(o["delivery_time"]))

    enriched = [_enrich_active_order(o, depots) for o in orders]

    return {
        "intent": "VIEW_ALL",
        "count": len(enriched),
        "orders": enriched,
    }


# ── VIEW_ORDER ───────────────────────────────────────────────────────────────

def handle_view_order(params):
    """Look up an order across active orders AND history.

    Two ways to look up:
      - {"order_id": N}        → exact ID match (mirrors C searchOrder())
      - {"name":     "string"} → case-insensitive substring match on the
                                 product name (e.g. "airpods" → AirPodsPro)
    Name search returns a list response (it can return multiple matches
    or zero); ID search returns a single-order response.
    SEARCH_ORDER is aliased to this same handler.
    """
    name = params.get("name")
    if name:
        return _name_search(str(name))

    order_id = params.get("order_id")
    if order_id is None:
        return {"intent": "VIEW_ORDER", "error": "Missing order_id or name parameter"}

    order_id = int(order_id)
    depots = load_depots()

    # 1. Search active orders first (mirrors C logic order).
    for order in load_orders():
        if order["package_id"] == order_id:
            return {
                "intent": "VIEW_ORDER",
                "found": True,
                "source": "active",
                "order": _enrich_active_order(order, depots),
            }

    # 2. Fall through to history (delivered orders).
    for order in load_history():
        if order["package_id"] == order_id:
            return {
                "intent": "VIEW_ORDER",
                "found": True,
                "source": "history",
                "order": _enrich_history_order(order, depots),
            }

    return {
        "intent": "VIEW_ORDER",
        "found": False,
        "message": f"No order with ID {order_id} in active orders or history",
    }


def _name_search(name):
    """Case-insensitive substring search across active+history by product name.
    Returns a list-shaped response so the UI's existing order-list renderer
    can display multiple matches.
    """
    needle = name.strip().lower()
    if not needle:
        return {"intent": "VIEW_ORDER", "error": "Empty name"}

    depots = load_depots()
    matches = []

    for order in load_orders():
        if needle in order["name"].lower():
            matches.append({
                **_enrich_active_order(order, depots),
                "_source": "active",
            })
    for order in load_history():
        if needle in order["name"].lower():
            matches.append({
                **_enrich_history_order(order, depots),
                "_source": "history",
            })

    if not matches:
        return {
            "intent": "VIEW_ORDER",
            "query": name,
            "count": 0,
            "orders": [],
            "found": False,
            "message": f"No orders matching '{name}'",
        }

    return {
        "intent": "VIEW_ORDER",
        "query": name,
        "count": len(matches),
        "orders": matches,
        "found": True,
    }


# ── FILTER_BY_DEPOT ──────────────────────────────────────────────────────────

def handle_filter_by_depot(params):
    """New endpoint (not in C code): filter active orders by depot/courier.
    Accepts either depot_id (int) or depot_name (string).
    """
    depots = load_depots()
    orders = load_orders()

    depot = None
    depot_name = params.get("depot_name")
    depot_id = params.get("depot_id")

    if depot_name:
        depot = find_depot_by_name(depots, depot_name)
    elif depot_id is not None:
        depot = find_depot_by_id(depots, int(depot_id))

    if not depot:
        return {
            "intent": "FILTER_BY_DEPOT",
            "error": f"Depot not found: {depot_name or depot_id}",
            "available_depots": [{"id": d["depot_id"], "name": d["name"]} for d in depots],
        }

    filtered = [o for o in orders if o["courier"] == depot["depot_id"]]
    filtered.sort(key=lambda o: convert_to_timestamp(o["delivery_time"]))

    enriched = [_enrich_active_order(o, depots) for o in filtered]

    return {
        "intent": "FILTER_BY_DEPOT",
        "depot": depot["name"],
        "depot_id": depot["depot_id"],
        "count": len(enriched),
        "orders": enriched,
    }


# ── SHOW_HISTORY ─────────────────────────────────────────────────────────────

def handle_show_history(params):
    """Mirror of C deliveredOrders(): sync overdue orders to history,
    then return all history sorted latest-first.
    """
    sync_delivered_orders()

    depots = load_depots()
    history = load_history()

    # Sort latest first — mirrors sortDeliveredItems() (descending)
    history.sort(key=lambda o: convert_to_timestamp(o["delivery_time"]), reverse=True)

    enriched = [_enrich_history_order(o, depots) for o in history]

    return {
        "intent": "SHOW_HISTORY",
        "count": len(enriched),
        "orders": enriched,
    }


# ── CALCULATE_COST ───────────────────────────────────────────────────────────

def handle_calculate_cost(params):
    """Mirror of C calculateCost(): base + (distance * ratePerKm) + (weight * ratePerKg).

    Accepts either:
      - {"order_id": N}                 → look up the order (active+history),
                                           pull weight + courier from it
      - {"weight": W, "courier_id": I}  → compute directly
      - {"weight": W, "courier_name": S}→ resolve depot by name, compute
    """
    depots = load_depots()

    weight = params.get("weight")
    courier_id = params.get("courier_id")
    courier_name = params.get("courier_name")

    # Order-id lookup path: find the order, fill in weight + courier from it.
    order_id = params.get("order_id")
    if order_id is not None and weight is None:
        order_id = int(order_id)
        order = None
        for o in load_orders():
            if o["package_id"] == order_id:
                order = o
                break
        if order is None:
            for o in load_history():
                if o["package_id"] == order_id:
                    order = o
                    break
        if order is None:
            return {
                "intent": "CALCULATE_COST",
                "error": f"Order {order_id} not found",
            }
        weight = float(order["weight"])
        if courier_id is None and courier_name is None:
            courier_id = int(order["courier"])

    if weight is None:
        return {"intent": "CALCULATE_COST", "error": "Missing weight parameter"}
    weight = float(weight)

    if courier_name and not courier_id:
        depot = find_depot_by_name(depots, courier_name)
        if depot:
            courier_id = depot["depot_id"]
        else:
            return {
                "intent": "CALCULATE_COST",
                "error": f"Courier not found: {courier_name}",
                "available_couriers": [{"id": d["depot_id"], "name": d["name"]} for d in depots],
            }

    if courier_id is None:
        return {"intent": "CALCULATE_COST", "error": "Missing courier_id or courier_name"}

    courier_id = int(courier_id)
    cost = calculate_cost(depots, weight, courier_id)

    resp = {
        "intent": "CALCULATE_COST",
        "weight_kg": weight,
        "courier_id": courier_id,
        "courier_name": get_courier_name(depots, courier_id),
        "cost_gbp": cost,
    }
    if order_id is not None:
        resp["order_id"] = int(order_id)
    return resp


# ── SEARCH_ORDER ─────────────────────────────────────────────────────────────

def handle_search_order(params):
    """Alias for handle_view_order: same operation, same response.

    The model still has SEARCH_ORDER as a distinct class (we kept the
    original 6-way head from the report) but on the server it's just
    a forward to VIEW_ORDER so the user sees one consistent answer
    regardless of how the BERT classifier disambiguates the phrasing.
    """
    return handle_view_order(params)


# ── Dispatcher ───────────────────────────────────────────────────────────────

INTENT_HANDLERS = {
    "VIEW_ALL": handle_view_all,
    "VIEW_ORDER": handle_view_order,
    "FILTER_BY_DEPOT": handle_filter_by_depot,
    "SHOW_HISTORY": handle_show_history,
    "CALCULATE_COST": handle_calculate_cost,
    "SEARCH_ORDER": handle_search_order,
}


def dispatch(intent, params=None):
    """Route an intent string to its handler."""
    if params is None:
        params = {}

    handler = INTENT_HANDLERS.get(intent)
    if not handler:
        return {
            "error": f"Unknown intent: {intent}",
            "supported": list(INTENT_HANDLERS.keys()),
        }

    return handler(params)
