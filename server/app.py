"""
app.py
Flask server that receives classified intents from the ESP32 and dispatches
them to the parcel tracking handlers.

Single endpoint:
    POST /api/intent
    Body: {"intent": "VIEW_ALL", "params": {"order_id": 123}}

The ESP32 runs BERT on-device to classify the user's natural language into
one of 6 intents, then sends the structured intent + extracted params here.
"""

from flask import Flask, request, jsonify
from handlers import dispatch

app = Flask(__name__)


@app.route("/api/intent", methods=["POST"])
def handle_intent():
    """Main endpoint: receive intent + params, return query results."""
    data = request.get_json(silent=True)

    if not data or "intent" not in data:
        return jsonify({
            "error": "Request must include JSON with 'intent' field",
            "example": {
                "intent": "VIEW_ALL",
                "params": {}
            },
        }), 400

    intent = data["intent"].upper()
    params = data.get("params", {})

    result = dispatch(intent, params)

    status_code = 200 if "error" not in result else 400
    return jsonify(result), status_code


@app.route("/api/depots", methods=["GET"])
def list_depots():
    """Utility endpoint: list all available depots/couriers."""
    from data_layer import load_depots
    depots = load_depots()
    return jsonify({"depots": depots})


@app.route("/api/health", methods=["GET"])
def health():
    """Health check for the ESP32 to verify connectivity."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("=" * 50)
    print("  Parcel Tracker Flask Server")
    print("  POST /api/intent  — main endpoint")
    print("  GET  /api/depots  — list couriers")
    print("  GET  /api/health  — connectivity check")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=True)
