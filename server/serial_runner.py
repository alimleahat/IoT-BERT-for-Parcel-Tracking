"""
serial_runner.py — owns the USB serial port to the ESP32 and runs one
inference at a time on behalf of the Flask app.

Design:
  - One thread (the main Flask thread or anything that calls .run()) at a
    time holds an internal lock; concurrent UI requests are serialised.
  - On startup: open the port, pulse DTR/RTS to reset the ESP32, drain
    the boot log until we see "READY".
  - For each .run(text):
      1. Send text + newline.
      2. Read lines, capturing:
           Intent:           → intent label + score + latency_ms
           Scores:           → 6 per-class scores
           PARCEL: Entity:   → extracted entities (best-effort dict)
           TX:{...}          → the intent payload from firmware
      3. Hand TX: payload to the supplied dispatcher (handlers.dispatch).
      4. Send back RX:{...} so the ESP32 unblocks.
      5. Drain until the trailing "──────" banner.
      6. Return one dict with everything: text, intent, score, scores,
         latency_ms, entities, request_json, response_json.

If anything is missing (e.g. firmware silent), .run() raises RunnerError
with whatever was collected so far attached for debugging.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import serial


# Lines emitted by main.c that we parse. Keep in sync with esp32_firmware/main/main.c.
# Two flavours of the Intent line:
#   Intent: VIEW_ALL (score=2.011) — 3720 ms
#   Intent: SHOW_HISTORY (overridden from SEARCH_ORDER by keyword rule) — 510 ms
_INTENT_NORMAL_RE = re.compile(
    r"Intent:\s+(\w+)\s+\(score=([-\d.]+)\)\s+—\s+(\d+)\s+ms"
)
_INTENT_OVERRIDE_RE = re.compile(
    r"Intent:\s+(\w+)\s+\(overridden from (\w+) by keyword rule\)\s+—\s+(\d+)\s+ms"
)
_SCORES_LINE_RE = re.compile(r"Scores:\s+(.*)")
_SCORE_PAIR_RE = re.compile(r"(\w+)=([-\d.]+)")
_ENTITY_RE = re.compile(r"PARCEL:\s+Entity:\s+(.*)")


class RunnerError(RuntimeError):
    pass


@dataclass
class RunResult:
    text: str
    intent: str | None = None
    score: float | None = None
    latency_ms: int | None = None
    scores: dict[str, float] = field(default_factory=dict)
    entities: dict[str, Any] = field(default_factory=dict)
    request_json: dict[str, Any] | None = None
    response_json: dict[str, Any] | None = None
    raw_log: list[str] = field(default_factory=list)
    overridden_from: str | None = None  # set if firmware applied keyword fallback

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "intent": self.intent,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "scores": self.scores,
            "entities": self.entities,
            "request": self.request_json,
            "response": self.response_json,
            "raw_log": self.raw_log,
            "overridden_from": self.overridden_from,
        }


class SerialRunner:
    def __init__(
        self,
        port: str = "/dev/cu.usbmodem1101",
        baud: int = 115200,
        ready_timeout: float = 25.0,
        run_timeout: float = 30.0,
    ) -> None:
        self.port = port
        self.baud = baud
        self.ready_timeout = ready_timeout
        self.run_timeout = run_timeout
        self._ser: serial.Serial | None = None
        self._lock = threading.Lock()
        self._connected = False
        self._arena_used = -1
        self._arena_total = -1

    # ── lifecycle ──────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open port, reset device, wait for READY. Idempotent."""
        with self._lock:
            if self._connected:
                return
            self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self._pulse_reset()
            ok = self._wait_for_ready()
            if not ok:
                self._ser.close()
                self._ser = None
                raise RunnerError(
                    f"ESP32 never sent READY within {self.ready_timeout}s on {self.port}"
                )
            self._connected = True

    def close(self) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
            self._ser = None
            self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def arena_info(self) -> tuple[int, int]:
        return self._arena_used, self._arena_total

    # ── internal serial helpers ────────────────────────────────────────

    def _pulse_reset(self) -> None:
        assert self._ser is not None
        self._ser.setDTR(False)
        self._ser.setRTS(False)
        time.sleep(0.1)
        self._ser.setDTR(True)
        self._ser.setRTS(True)
        time.sleep(0.1)
        self._ser.reset_input_buffer()

    def _read_line(self, deadline: float) -> str | None:
        assert self._ser is not None
        buf = bytearray()
        while time.time() < deadline:
            chunk = self._ser.read(1)
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

    def _wait_for_ready(self) -> bool:
        deadline = time.time() + self.ready_timeout
        arena_re = re.compile(r"Arena used:\s+(\d+)\s*/\s*(\d+)\s+bytes")
        while time.time() < deadline:
            line = self._read_line(deadline)
            if line is None:
                continue
            m = arena_re.search(line)
            if m:
                self._arena_used = int(m.group(1))
                self._arena_total = int(m.group(2))
            if line == "READY":
                return True
        return False

    # ── main entry point ───────────────────────────────────────────────

    def run(
        self,
        text: str,
        dispatcher: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> RunResult:
        """Send text to the ESP32, drive the round-trip, return all artefacts."""
        if not self._connected or self._ser is None:
            raise RunnerError("SerialRunner not connected. Call connect() first.")

        text = text.strip()
        if not text:
            raise RunnerError("Empty command text")

        with self._lock:
            return self._run_locked(text, dispatcher)

    def _run_locked(
        self,
        text: str,
        dispatcher: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> RunResult:
        assert self._ser is not None
        self._ser.reset_input_buffer()
        self._ser.write((text + "\n").encode("utf-8"))
        self._ser.flush()

        result = RunResult(text=text)
        deadline = time.time() + self.run_timeout
        sent_rx = False

        while time.time() < deadline:
            line = self._read_line(deadline)
            if line is None:
                continue
            result.raw_log.append(line)

            # Parse Intent line — either the normal "score=X" form or the
            # keyword-override form. Score may be filled in later from the
            # Scores: line if this was an override.
            m = _INTENT_NORMAL_RE.search(line)
            if m:
                result.intent = m.group(1)
                result.score = float(m.group(2))
                result.latency_ms = int(m.group(3))
                continue
            mo = _INTENT_OVERRIDE_RE.search(line)
            if mo:
                result.intent = mo.group(1)
                result.overridden_from = mo.group(2)
                result.latency_ms = int(mo.group(3))
                # score will be backfilled when we see the Scores: line
                continue

            # Parse Scores line — there are 6 KEY=VAL pairs after "Scores:".
            # If this followed an override, backfill the per-class score so
            # the UI's score field reflects the chosen intent.
            sm = _SCORES_LINE_RE.search(line)
            if sm:
                for k, v in _SCORE_PAIR_RE.findall(sm.group(1)):
                    try:
                        result.scores[k] = float(v)
                    except ValueError:
                        pass
                if result.score is None and result.intent in result.scores:
                    result.score = result.scores[result.intent]
                continue

            # Parse Entity lines — appear as "I (..) PARCEL: Entity: KEY=VAL".
            em = _ENTITY_RE.search(line)
            if em:
                # KEY=VAL or KEY=VAL (id=N)
                pairs = re.findall(r"(\w+)=([\w.]+)", em.group(1))
                for k, v in pairs:
                    result.entities[k] = _coerce(v)
                continue

            # Parse TX:{...} → send to dispatcher → write RX:{...}.
            if line.startswith("TX:"):
                body = line[3:].strip()
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError as e:
                    raise RunnerError(f"ESP32 sent invalid TX JSON: {e}: {body!r}")
                result.request_json = payload
                intent = (payload.get("intent") or "").upper()
                params = payload.get("params") or {}
                try:
                    response = dispatcher(intent, params)
                except Exception as e:
                    response = {"error": f"dispatcher: {type(e).__name__}: {e}"}
                result.response_json = response
                rx_body = json.dumps(response, separators=(",", ":"))
                self._ser.write(f"RX:{rx_body}\n".encode("utf-8"))
                self._ser.flush()
                sent_rx = True
                continue

            # Trailing banner of the SERVER RESPONSE block — round complete.
            if sent_rx and line.startswith("──────"):
                return result

        # Timed out. Return whatever we have, but flag it.
        raise RunnerError(
            f"Timed out after {self.run_timeout}s waiting for ESP32 response. "
            f"Got: intent={result.intent}, request={result.request_json is not None}, "
            f"response={result.response_json is not None}"
        )


def _coerce(s: str) -> Any:
    """Best-effort string → int/float/string."""
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    try:
        return float(s)
    except ValueError:
        return s
