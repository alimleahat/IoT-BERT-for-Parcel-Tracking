#!/usr/bin/env python3
"""
serial_bridge.py — USB serial bridge between the ESP32 and the Flask server.

Why this exists
---------------
The demo venue has no usable WiFi (Glide captive portal + no hotspot), so
rather than have the ESP32 talk HTTP directly, the ESP32 prints framed
request lines over its USB serial port and this script relays them to the
Flask server running on localhost.

Protocol (line-oriented, one frame per line)
--------------------------------------------
ESP32 -> host:
    TX:<json>        # request — host POSTs <json> to /api/intent
    READY            # sent once after model loads successfully
    <anything else>  # ESP_LOG output, intent/score lines, etc. — printed through

host  -> ESP32:
    RX:<json>        # response body from Flask (single line)
    <user command>   # forwarded verbatim from the operator's keyboard

Run it with:
    python serial_bridge.py                         # defaults to /dev/cu.usbmodem101
    python serial_bridge.py --port /dev/cu.usbmodem1101
    python serial_bridge.py --flask http://127.0.0.1:5001
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import requests
import serial  # pyserial


# ── ANSI colours for the operator terminal ──────────────────────────────────
C_DIM    = "\033[2m"
C_CYAN   = "\033[36m"
C_GREEN  = "\033[32m"
C_YELLOW = "\033[33m"
C_RED    = "\033[31m"
C_BOLD   = "\033[1m"
C_RESET  = "\033[0m"


def log_info(msg: str) -> None:
    print(f"{C_DIM}[bridge]{C_RESET} {msg}", flush=True)


def log_tx(msg: str) -> None:
    print(f"{C_CYAN}→ Flask:{C_RESET} {msg}", flush=True)


def log_rx(msg: str) -> None:
    print(f"{C_GREEN}← Flask:{C_RESET} {msg}", flush=True)


def log_err(msg: str) -> None:
    print(f"{C_RED}[bridge ERR]{C_RESET} {msg}", flush=True)


# ── Flask POST ──────────────────────────────────────────────────────────────

def post_to_flask(flask_url: str, payload: dict, timeout: float = 10.0) -> str:
    """POST to /api/intent, return a single-line JSON string."""
    try:
        r = requests.post(flask_url, json=payload, timeout=timeout)
        try:
            body = r.json()
        except Exception:
            body = {"error": "non-JSON response", "status": r.status_code,
                    "text": r.text[:500]}
    except requests.exceptions.ConnectionError:
        body = {"error": "flask server unreachable",
                "hint": f"start it with: python app.py  (expected at {flask_url})"}
    except requests.exceptions.Timeout:
        body = {"error": "flask timeout"}
    except Exception as e:
        body = {"error": f"bridge exception: {type(e).__name__}: {e}"}

    # Compact, single-line JSON so ESP32's line-oriented reader never sees a
    # stray newline in the middle of a response.
    return json.dumps(body, separators=(",", ":"))


# ── Serial reader thread ────────────────────────────────────────────────────

def reader_loop(ser: serial.Serial, flask_url: str) -> None:
    """Continuously read lines from ESP32.

    • "TX:..."  → POST to Flask, write "RX:..." back.
    • "READY"   → announce ready to the operator.
    • anything else → echo to operator terminal (ESP_LOG, intent lines, etc.).
    """
    buf = bytearray()
    while True:
        if not ser.is_open:
            return
        try:
            chunk = ser.read(1)
        except serial.SerialException as e:
            # If the main thread closed the port on shutdown, exit quietly.
            if not ser.is_open:
                return
            log_err(f"serial read failed: {e}")
            time.sleep(0.5)
            continue
        except OSError as e:
            # Bad file descriptor on shutdown — port was closed under us.
            if not ser.is_open:
                return
            log_err(f"serial read failed: {e}")
            time.sleep(0.5)
            continue

        if not chunk:
            continue

        # Accumulate until newline, then flush as one line.
        for b in chunk:
            if b in (0x0A, 0x0D):  # \n or \r
                if not buf:
                    continue
                try:
                    line = buf.decode("utf-8", errors="replace").rstrip()
                except Exception:
                    line = repr(bytes(buf))
                buf.clear()
                handle_line(ser, line, flask_url)
            else:
                buf.append(b)


def handle_line(ser: serial.Serial, line: str, flask_url: str) -> None:
    if line.startswith("TX:"):
        body = line[3:].strip()
        log_tx(body)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            resp = json.dumps({"error": f"bridge: invalid JSON from ESP32: {e}"},
                              separators=(",", ":"))
        else:
            resp = post_to_flask(flask_url, payload)
        log_rx(resp)
        # Write the response back so the ESP32's read_line picks it up.
        ser.write(f"RX:{resp}\n".encode("utf-8"))
        ser.flush()
        return

    if line == "READY":
        print(f"{C_GREEN}{C_BOLD}[bridge] ESP32 is ready — type a command and press Enter.{C_RESET}",
              flush=True)
        return

    # Everything else is just ESP32 diagnostic output; show it to the operator.
    print(line, flush=True)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="ESP32 <-> Flask USB serial bridge")
    parser.add_argument("--port", default="/dev/cu.usbmodem101",
                        help="serial port (default: /dev/cu.usbmodem101)")
    parser.add_argument("--baud", type=int, default=115200,
                        help="baud rate (default: 115200)")
    parser.add_argument("--flask", default="http://127.0.0.1:5001/api/intent",
                        help="Flask endpoint URL")
    args = parser.parse_args()

    log_info(f"opening {args.port} @ {args.baud} baud")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        log_err(f"could not open serial port: {e}")
        log_err("hint: unplug/replug the ESP32, or run:  ls /dev/cu.usbmodem*")
        return 1

    # Pulse DTR/RTS to reset the ESP32 so we're guaranteed to see the READY
    # marker that's only emitted once at boot. Without this, if the ESP32 has
    # been running for a while, READY has already scrolled past and the bridge
    # would sit forever waiting for it.
    log_info("pulsing DTR/RTS to reset ESP32...")
    try:
        ser.setDTR(False)
        ser.setRTS(False)
        time.sleep(0.1)
        ser.setDTR(True)
        ser.setRTS(True)
        time.sleep(0.1)
        ser.reset_input_buffer()
    except Exception as e:
        log_err(f"DTR/RTS pulse failed (non-fatal): {e}")

    log_info(f"relaying TX: frames to {args.flask}")
    log_info("waiting for READY from ESP32 (boot takes ~5s after flash)...")

    # Spawn the reader thread.
    t = threading.Thread(target=reader_loop, args=(ser, args.flask), daemon=True)
    t.start()

    # Main thread: forward user keystrokes to the ESP32 line by line.
    try:
        while True:
            try:
                cmd = input("")
            except EOFError:
                break
            ser.write((cmd + "\n").encode("utf-8"))
            ser.flush()
    except KeyboardInterrupt:
        pass
    finally:
        log_info("closing serial port")
        try:
            ser.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
