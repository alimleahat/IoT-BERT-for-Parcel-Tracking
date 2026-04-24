#!/usr/bin/env python3
"""
bench_latency.py — measure on-device BERT inference latency across many runs.

Reuses the existing test_intent_e2e.py plumbing (reset → READY → send command
→ parse "— XXXX ms" line → record → repeat).  After N runs it prints a
summary (count, mean, median, stddev, min, max, p95) per phrase and overall,
and also dumps a CSV for the KPI slide.

Usage:
  python server/bench_latency.py --repeats 5           # 5 × 6 phrases = 30 runs
  python server/bench_latency.py --repeats 10 --only "view order 101"
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
import time
from pathlib import Path

import serial


PHRASES = [
    "show me all orders",
    "view order 101",
    "filter by depot FadEx",
    "show history",
    "calculate the cost of order 101",
    "search for order with ID 123",
]

LAT_RE = re.compile(r"Intent:\s+(\w+)\s+\(score=[-\d.]+\)\s+—\s+(\d+)\s+ms")
ARENA_RE = re.compile(r"Arena used:\s+(\d+)\s*/\s*(\d+)\s+bytes")


def pulse_reset(ser: serial.Serial) -> None:
    ser.setDTR(False); ser.setRTS(False)
    time.sleep(0.1)
    ser.setDTR(True); ser.setRTS(True)
    time.sleep(0.1)
    ser.reset_input_buffer()


def read_line(ser: serial.Serial, deadline: float) -> str | None:
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


def wait_for_ready(ser, timeout=25.0) -> tuple[bool, int, int]:
    arena_used, arena_total = -1, -1
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = read_line(ser, deadline)
        if line is None:
            continue
        m = ARENA_RE.search(line)
        if m:
            arena_used = int(m.group(1))
            arena_total = int(m.group(2))
        if line == "READY":
            return True, arena_used, arena_total
    return False, arena_used, arena_total


def run_one(ser: serial.Serial, phrase: str, timeout: float = 20.0) -> tuple[int, str] | None:
    """Send one phrase; return (latency_ms, intent) or None on timeout."""
    ser.write((phrase + "\n").encode("utf-8"))
    ser.flush()
    deadline = time.time() + timeout
    saw_tx = False
    latency_ms = None
    intent = None
    while time.time() < deadline:
        line = read_line(ser, deadline)
        if line is None:
            continue
        m = LAT_RE.search(line)
        if m:
            intent = m.group(1)
            latency_ms = int(m.group(2))
        if line.startswith("TX:"):
            # Fake the flask reply so the ESP32 advances (we don't care about
            # the server response here — only the on-device latency).
            ser.write(b'RX:{"intent":"BENCH","note":"no-op"}\n')
            ser.flush()
            saw_tx = True
            continue
        # Terminator: trailing separator of the SERVER RESPONSE banner.
        if saw_tx and latency_ms is not None and line.startswith("──────"):
            return latency_ms, intent
    return None


def pct(vals, p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(label: str, vals: list[int]) -> dict:
    if not vals:
        print(f"{label:<40} (no samples)")
        return {}
    d = {
        "label": label,
        "n": len(vals),
        "mean_ms": statistics.mean(vals),
        "median_ms": statistics.median(vals),
        "stdev_ms": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min_ms": min(vals),
        "max_ms": max(vals),
        "p95_ms": pct(vals, 0.95),
    }
    print(
        f"{label:<40} "
        f"n={d['n']:<3} "
        f"mean={d['mean_ms']:7.1f}ms  "
        f"median={d['median_ms']:7.1f}ms  "
        f"σ={d['stdev_ms']:5.1f}ms  "
        f"min={d['min_ms']}ms  "
        f"max={d['max_ms']}ms  "
        f"p95={d['p95_ms']:7.1f}ms"
    )
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbmodem1101")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--repeats", type=int, default=5, help="runs per phrase")
    ap.add_argument("--only", default=None, help="single phrase to repeat")
    ap.add_argument("--csv", default="kpi_latency.csv",
                    help="output CSV (relative to CWD)")
    args = ap.parse_args()

    phrases = [args.only] if args.only else PHRASES

    print(f"[bench] opening {args.port} @ {args.baud}")
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    try:
        print("[bench] pulsing DTR/RTS...")
        pulse_reset(ser)
        print("[bench] waiting for READY...")
        ready, arena_used, arena_total = wait_for_ready(ser, timeout=25.0)
        if not ready:
            print("[bench] FATAL: never saw READY", file=sys.stderr)
            return 1
        print(f"[bench] READY. Arena used: {arena_used}/{arena_total} bytes "
              f"({100.0 * arena_used / arena_total:.1f}%)")
        print(f"[bench] running {len(phrases)} × {args.repeats} = "
              f"{len(phrases) * args.repeats} inferences\n")

        rows = []
        per_phrase: dict[str, list[int]] = {}
        for phrase in phrases:
            per_phrase[phrase] = []
            for rep in range(args.repeats):
                res = run_one(ser, phrase, timeout=20.0)
                if res is None:
                    print(f"  [{phrase}] rep {rep+1}: TIMEOUT")
                    continue
                ms, intent = res
                per_phrase[phrase].append(ms)
                rows.append({
                    "phrase": phrase, "rep": rep + 1,
                    "intent": intent, "latency_ms": ms,
                })
                print(f"  [{phrase[:30]:<30}] rep {rep+1}: {ms} ms → {intent}")
                time.sleep(0.2)

        print("\n─── Per-phrase summary ───")
        summaries = [summarize(p[:40], per_phrase[p]) for p in phrases]

        all_vals = [ms for vs in per_phrase.values() for ms in vs]
        print("\n─── Overall ───")
        overall = summarize("ALL INFERENCES", all_vals)
        overall_row = {
            "phrase": "__overall__", "rep": 0,
            "intent": "ALL", "latency_ms": -1,
            **overall,
        }

        csv_path = Path(args.csv)
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phrase", "rep", "intent", "latency_ms"])
            for r in rows:
                w.writerow([r["phrase"], r["rep"], r["intent"], r["latency_ms"]])
        print(f"\n[bench] wrote {csv_path} ({len(rows)} rows)")

        print(f"\n[bench] KPI snapshot:")
        print(f"  arena_used = {arena_used} bytes "
              f"({arena_used / 1024:.1f} KB, "
              f"{100.0 * arena_used / arena_total:.1f}% of {arena_total / (1024*1024):.1f} MB)")
        if all_vals:
            print(f"  latency p50 = {statistics.median(all_vals):.0f} ms")
            print(f"  latency p95 = {pct(all_vals, 0.95):.0f} ms")
            print(f"  latency σ   = {statistics.stdev(all_vals) if len(all_vals)>1 else 0:.0f} ms")
        return 0
    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
