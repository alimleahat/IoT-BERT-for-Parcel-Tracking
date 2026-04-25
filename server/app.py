"""
app.py
Flask server for the on-device BERT parcel tracker.

Two roles in one process:
  1. Existing JSON API for the original ESP32-pushed-intent flow:
        POST /api/intent  {"intent": "...", "params": {...}}
  2. Demo-friendly web UI at /:
        - User types natural language in the browser
        - Server forwards to the ESP32 over USB serial
        - ESP32 runs BERT, returns the classified intent
        - Server resolves the intent against its own data layer
        - Browser shows: scores, latency, intent, entities, server response

The serial port is owned by a single SerialRunner instance and accesses are
serialised so concurrent UI requests don't race.

Run:
    python server/app.py                     # default /dev/cu.usbmodem1101
    SERIAL_PORT=/dev/tty.usbserial python server/app.py
    DISABLE_SERIAL=1 python server/app.py    # API-only mode (no ESP32 hooked up)
"""

import os
import sys
import threading

from flask import Flask, jsonify, request, send_from_directory

from handlers import dispatch
from serial_runner import RunnerError, SerialRunner

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ── Serial runner (owns the USB port to the ESP32) ────────────────────────
_SERIAL_DISABLED = os.environ.get("DISABLE_SERIAL", "").strip() in ("1", "true", "yes")
_SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/cu.usbmodem1101")
_SERIAL_BAUD = int(os.environ.get("SERIAL_BAUD", "115200"))

runner: SerialRunner | None = None
_runner_lock = threading.Lock()
_runner_error: str | None = None


def _ensure_runner_started() -> None:
    """Lazy-connect to the ESP32 on the first request that needs it."""
    global runner, _runner_error
    if _SERIAL_DISABLED:
        return
    with _runner_lock:
        if runner is not None and runner.connected:
            return
        try:
            r = SerialRunner(port=_SERIAL_PORT, baud=_SERIAL_BAUD)
            r.connect()
            runner = r
            _runner_error = None
            print(f"[app] SerialRunner connected on {_SERIAL_PORT}; "
                  f"arena {r.arena_info[0]}/{r.arena_info[1]} bytes",
                  file=sys.stderr, flush=True)
        except Exception as e:
            _runner_error = f"{type(e).__name__}: {e}"
            print(f"[app] SerialRunner failed to start: {_runner_error}",
                  file=sys.stderr, flush=True)
            raise


# ── UI: index.html ─────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder, "index.html")


# ── New endpoint: full round-trip (browser → ESP32 → server → browser) ────

@app.route("/api/run", methods=["POST"])
def run_command():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    if _SERIAL_DISABLED:
        return jsonify({
            "error": "Server started with DISABLE_SERIAL=1 — no ESP32 connection",
        }), 503

    try:
        _ensure_runner_started()
    except Exception as e:
        return jsonify({
            "error": f"ESP32 not reachable: {type(e).__name__}: {e}",
            "hint": "Check the USB cable and that no other process owns the port.",
        }), 503

    assert runner is not None
    try:
        result = runner.run(text, dispatcher=dispatch)
    except RunnerError as e:
        return jsonify({"error": str(e)}), 504
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    return jsonify(result.to_dict()), 200


@app.route("/api/status", methods=["GET"])
def status():
    info: dict = {
        "serial_disabled": _SERIAL_DISABLED,
        "port": _SERIAL_PORT,
        "baud": _SERIAL_BAUD,
        "connected": runner is not None and runner.connected,
        "last_error": _runner_error,
    }
    if runner is not None:
        used, total = runner.arena_info
        info["arena_used"] = used
        info["arena_total"] = total
    return jsonify(info)


# ── Original API (kept for back-compat with serial_bridge.py + tests) ─────

@app.route("/api/intent", methods=["POST"])
def handle_intent():
    """Legacy endpoint: receive an already-classified intent + params."""
    data = request.get_json(silent=True)
    if not data or "intent" not in data:
        return jsonify({
            "error": "Request must include JSON with 'intent' field",
            "example": {"intent": "VIEW_ALL", "params": {}},
        }), 400

    intent = data["intent"].upper()
    params = data.get("params", {})
    result = dispatch(intent, params)
    status_code = 200 if "error" not in result else 400
    return jsonify(result), status_code


@app.route("/api/depots", methods=["GET"])
def list_depots():
    from data_layer import load_depots
    return jsonify({"depots": load_depots()})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("=" * 60)
    print("  Parcel Tracker — Flask + ESP32 BERT bridge")
    print(f"  Web UI:           http://127.0.0.1:5001/")
    print(f"  Run endpoint:     POST /api/run    {{\"text\": \"...\"}}")
    print(f"  Legacy endpoint:  POST /api/intent {{\"intent\": \"...\"}}")
    print(f"  Status:           GET  /api/status")
    if _SERIAL_DISABLED:
        print("  Serial:           DISABLED (DISABLE_SERIAL=1)")
    else:
        print(f"  Serial:           {_SERIAL_PORT} @ {_SERIAL_BAUD}")
    print("=" * 60)
    # Use threaded=True (Flask default) but disable the reloader so we don't
    # open the serial port twice. debug=True with use_reloader=False keeps
    # nice tracebacks without the double-import.
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
