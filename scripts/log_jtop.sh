#!/usr/bin/env bash
set -euo pipefail

# Log Jetson stats from jtop as JSON Lines.
#
# Usage:
#   scripts/log_jtop.sh [duration_seconds] [interval_seconds] [output_file]
#
# Examples:
#   scripts/log_jtop.sh
#   scripts/log_jtop.sh 300 1
#   scripts/log_jtop.sh 120 0.5 /tmp/jtop_run.jsonl

DURATION_SEC="${1:-60}"
INTERVAL_SEC="${2:-1}"
OUT_FILE="${3:-}"

if [[ -z "${OUT_FILE}" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  OUT_FILE="/home/forecr/pilot_core/logs/jtop_${ts}.jsonl"
fi

mkdir -p "$(dirname "$OUT_FILE")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found." >&2
  exit 1
fi

python3 - "$DURATION_SEC" "$INTERVAL_SEC" "$OUT_FILE" <<'PY'
import json
import os
import sys
import time
from datetime import datetime, timezone

try:
    from jtop import jtop
except Exception as exc:
    print(f"Failed to import jtop Python module: {exc}", file=sys.stderr)
    print("Install with: sudo -H pip3 install -U jetson-stats", file=sys.stderr)
    sys.exit(2)


def to_float(name: str, value: str) -> float:
    try:
        return float(value)
    except Exception:
        print(f"Invalid {name}: {value}", file=sys.stderr)
        sys.exit(2)


duration = to_float("duration_seconds", sys.argv[1])
interval = to_float("interval_seconds", sys.argv[2])
out_path = sys.argv[3]

if duration <= 0:
    print("duration_seconds must be > 0", file=sys.stderr)
    sys.exit(2)
if interval <= 0:
    print("interval_seconds must be > 0", file=sys.stderr)
    sys.exit(2)

start = time.time()
count = 0

with open(out_path, "w", encoding="utf-8") as f:
    with jtop() as jetson:
        while jetson.ok() and (time.time() - start) < duration:
            now = time.time()
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "unix_ts": now,
                "uptime_sec": now - start,
                "stats": jetson.stats,
            }
            f.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
            f.flush()
            count += 1
            time.sleep(interval)

print(f"samples={count}")
print(f"output={out_path}")
PY

echo "Saved jtop log to: $OUT_FILE"
