"""
cost.py
Mirror of the C cost/depot helpers in processing.c:
  - calculateCost()
  - getCourierName()
"""

from data_layer import load_depots


def get_courier_name(depots, courier_id):
    """Mirror of C getCourierName(): look up depot name by ID."""
    for depot in depots:
        if depot["depot_id"] == courier_id:
            return depot["name"]
    return "Unknown"


def calculate_cost(depots, weight, courier_id):
    """Mirror of C calculateCost().

    Formula: base + (distance * ratePerKm) + (weight * ratePerKg)
    """
    for depot in depots:
        if depot["depot_id"] == courier_id:
            base = depot["base_rate"]
            distance_cost = depot["distance"] * depot["rate_per_km"]
            weight_cost = weight * depot["rate_per_kg"]
            return round(base + distance_cost + weight_cost, 2)

    return 0.0


def find_depot_by_name(depots, name):
    """Find depot by name (case-insensitive). Used by FILTER_BY_DEPOT."""
    name_lower = name.lower()
    for depot in depots:
        if depot["name"].lower() == name_lower:
            return depot
    return None


def find_depot_by_id(depots, depot_id):
    """Find depot by numeric ID."""
    for depot in depots:
        if depot["depot_id"] == depot_id:
            return depot
    return None
