#!/usr/bin/env python3
"""
test_intent_e2e.py — one-shot end-to-end intent test.

Unlike serial_bridge.py, this doesn't read commands from stdin (which is
finicky when piped). It bundles the reset, READY wait, command injection,
and Flask POST in a single event loop, and exits when the scripted commands
are done.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests
import serial


COMMANDS = [
    # Each phrase is picked from the training distribution — strong margins in
    # the HF model, so the on-device INT8-weights hybrid kernel matches too.
    "show me all orders",                  # VIEW_ALL          → 8 active
    "view order 101",                      # VIEW_ORDER        → finds active 101
    "filter by depot FadEx",               # FILTER_BY_DEPOT   → active orders at FadEx
    "show history",                        # SHOW_HISTORY      → 16 delivered
    "calculate the cost of order 101",     # CALCULATE_COST    → £7.50 via FadEx
    "view order 123",                      # VIEW_ORDER        → falls through to history
]


def log(msg: str) -> None:
    print(f"[test] {msg}", flush=True)


def pulse_reset(ser: serial.Serial) -> None:
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(0.1)
    ser.setDTR(True)
    ser.setRTS(True)
    time.sleep(0.1)
    ser.reset_input_buffer()


def read_line(ser: serial.Serial, deadline: float) -> str | None:
    """Read one \\n- or \\r-terminated line (non-blocking up to deadline)."""
    buf = bytearray()
    while time.time() < deadline:
        chunk = ser.read(1)
        if not chunk:
            continue
        b = chunk[0]
        if b in (0x0A, 0x0D):
            if not buf:
                continue
            return buf.decode("utf-8", errors="replace").rstrip()
        buf.append(b)
    if buf:
        return buf.decode("utf-8", errors="replace").rstrip()
    return None


def wait_for_ready(ser: serial.Serial, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = read_line(ser, deadline)
        if line is None:
            continue
        print(line, flush=True)
        if line == "READY":
            return True
    return False


def post_flask(url: str, payload: dict) -> str:
    try:
        r = requests.post(url, json=payload, timeout=10.0)
        body = r.json()
    except Exception as e:
        body = {"error": f"bridge: {type(e).__name__}: {e}"}
    return json.dumps(body, separators=(",", ":"))


def run_one(ser: serial.Serial, flask_url: str, cmd: str, timeout: float = 30.0) -> None:
    log(f"→ sending: {cmd!r}")
    ser.write((cmd + "\n").encode("utf-8"))
    ser.flush()

    deadline = time.time() + timeout
    saw_rx_ack = False
    while time.time() < deadline:
        line = read_line(ser, deadline)
        if line is None:
            continue
        print(line, flush=True)
        if line.startswith("TX:"):
            body = line[3:].strip()
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as e:
                resp = json.dumps({"error": f"invalid JSON: {e}"}, separators=(",", ":"))
            else:
                resp = post_flask(flask_url, payload)
            log(f"← flask reply: {resp}")
            ser.write(f"RX:{resp}\n".encode("utf-8"))
            ser.flush()
            saw_rx_ack = True
        # End-of-round: after RX: ack, main.c prints the server response
        # framed by "─── SERVER RESPONSE ───" ... "──────────────────────".
        # The trailing row of box-drawing characters is our clean terminator.
        if saw_rx_ack and line.startswith("──────"):
            return
    log(f"[warn] command {cmd!r} hit timeout before ESP32 looped back")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbmodem1101")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--flask", default="http://127.0.0.1:5001/api/intent")
    ap.add_argument("--n", type=int, default=3, help="number of commands to run")
    args = ap.parse_args()

    log(f"opening {args.port} @ {args.baud}")
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    try:
        log("pulsing DTR/RTS to reset ESP32...")
        pulse_reset(ser)
        log("waiting for READY (up to 20s)...")
        if not wait_for_ready(ser, timeout=20.0):
            log("[FATAL] never saw READY")
            return 1
        log("ESP32 is READY — running commands")
        for cmd in COMMANDS[: args.n]:
            run_one(ser, args.flask, cmd, timeout=30.0)
            time.sleep(0.5)
        log("done")
        return 0
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
