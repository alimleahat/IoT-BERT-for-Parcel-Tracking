"""
time_utils.py
Python equivalents of the C time helpers in processing.c:
  - getTimeRemaining()
  - getTimeSinceDelivery()
  - convertToTimestamp()
"""

from datetime import datetime

TIME_FMT = "%Y-%m-%d_%H:%M"


def get_time_remaining(delivery_str):
    """Mirror of C getTimeRemaining().

    Returns (display_string, status) where status is 0=active, 1=delivered.
    """
    try:
        delivery_dt = datetime.strptime(delivery_str, TIME_FMT)
    except ValueError:
        return ("Invalid date", 0)

    now = datetime.now()
    diff = delivery_dt - now

    total_seconds = int(diff.total_seconds())

    if total_seconds <= 0:
        return ("Delivered", 1)

    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600

    if days == 0 and hours < 1:
        return ("Less than an hour", 0)

    return (f"{days} days {hours} hours", 0)


def get_time_since_delivery(delivery_str):
    """Mirror of C getTimeSinceDelivery().

    Returns a human-readable string like '5 days 3 hours ago'.
    """
    try:
        delivery_dt = datetime.strptime(delivery_str, TIME_FMT)
    except ValueError:
        return "Invalid date"

    now = datetime.now()
    diff = now - delivery_dt

    total_seconds = int(diff.total_seconds())

    if total_seconds < 0:
        return "Not delivered"

    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600

    if days == 0 and hours < 1:
        return "Less than an hour ago"

    return f"{days} days {hours} hours ago"


def convert_to_timestamp(delivery_str):
    """Mirror of C convertToTimestamp(). Returns datetime object."""
    return datetime.strptime(delivery_str, TIME_FMT)
