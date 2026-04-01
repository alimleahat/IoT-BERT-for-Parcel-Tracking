"""
data_layer.py
Reads/writes the same flat files as the C parcel tracker:
  - data/depots.txt   (DeliveryService records)
  - data/orders.txt   (active Order records)
  - data/history.txt  (delivered Order records)

File format is space-delimited, one record per line — identical to the C
fscanf/fprintf format so both codebases stay interchangeable.
"""

import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DEPOTS_FILE = os.path.join(DATA_DIR, "depots.txt")
ORDERS_FILE = os.path.join(DATA_DIR, "orders.txt")
HISTORY_FILE = os.path.join(DATA_DIR, "history.txt")

TIME_FMT = "%Y-%m-%d_%H:%M"


# ── Depot I/O ────────────────────────────────────────────────────────────────

def load_depots():
    """Mirror of C loadDepots(): parse depots.txt into list of dicts.

    Format per line:  depotID  name  distance  baseRate  ratePerKm  ratePerKg
    Example:          1 FadEx 12.5 2.50 0.20 1.00
    """
    depots = []
    if not os.path.exists(DEPOTS_FILE):
        return depots

    with open(DEPOTS_FILE, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            depots.append({
                "depot_id": int(parts[0]),
                "name": parts[1],
                "distance": float(parts[2]),
                "base_rate": float(parts[3]),
                "rate_per_km": float(parts[4]),
                "rate_per_kg": float(parts[5]),
            })
    return depots


# ── Order I/O ────────────────────────────────────────────────────────────────

def _parse_order_line(line):
    """Parse a single order line (same format for orders.txt and history.txt).

    Format: packageID  name  weight  deliverytime  status  cost  courier
    Example: 123 MacBook 4.00 2026-02-25_15:20 0 9.00 1
    """
    parts = line.strip().split()
    if len(parts) != 7:
        return None
    return {
        "package_id": int(parts[0]),
        "name": parts[1],
        "weight": float(parts[2]),
        "delivery_time": parts[3],
        "status": int(parts[4]),
        "cost": float(parts[5]),
        "courier": int(parts[6]),
    }


def _format_order_line(order):
    """Serialize an order dict back to the C-compatible line format."""
    return (
        f"{order['package_id']} {order['name']} {order['weight']:.2f} "
        f"{order['delivery_time']} {order['status']} {order['cost']:.2f} "
        f"{order['courier']}\n"
    )


def load_orders():
    """Mirror of C loadOrders(): read active orders from orders.txt."""
    orders = []
    if not os.path.exists(ORDERS_FILE):
        return orders

    with open(ORDERS_FILE, "r") as f:
        for line in f:
            order = _parse_order_line(line)
            if order:
                orders.append(order)
    return orders


def save_orders(orders):
    """Mirror of C saveOrders(): overwrite orders.txt with current list."""
    with open(ORDERS_FILE, "w") as f:
        for order in orders:
            f.write(_format_order_line(order))


def load_history():
    """Read delivered orders from history.txt."""
    history = []
    if not os.path.exists(HISTORY_FILE):
        return history

    with open(HISTORY_FILE, "r") as f:
        for line in f:
            order = _parse_order_line(line)
            if order:
                history.append(order)
    return history


def append_to_history(orders_to_archive):
    """Append delivered orders to history.txt (mirrors C fopen "a" mode)."""
    with open(HISTORY_FILE, "a") as f:
        for order in orders_to_archive:
            f.write(_format_order_line(order))


# ── Sync (mirrors C syncDeliveredOrders) ─────────────────────────────────────

def sync_delivered_orders():
    """Move orders whose delivery time has passed into history.txt,
    then compact orders.txt to contain only active orders.

    This is an exact mirror of orders.c:syncDeliveredOrders().
    """
    orders = load_orders()
    now = datetime.now()

    to_archive = []
    still_active = []

    for order in orders:
        try:
            delivery_dt = datetime.strptime(order["delivery_time"], TIME_FMT)
        except ValueError:
            still_active.append(order)
            continue

        if delivery_dt <= now and order["status"] == 0:
            order["status"] = 1
            to_archive.append(order)
        elif order["status"] == 1:
            to_archive.append(order)
        else:
            still_active.append(order)

    if to_archive:
        append_to_history(to_archive)
        save_orders(still_active)

    return still_active, to_archive
