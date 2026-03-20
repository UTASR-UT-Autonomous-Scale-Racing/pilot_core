#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/export_last_recording_throttle_steering.sh [output_csv_path]
#
# If output_csv_path is omitted, prints CSV to stdout.

RECORDINGS_ROOT="${RECORDINGS_ROOT:-$HOME/recordings}"
OUTPUT_PATH="${1:--}"

latest_recording="$(ls -dt "$RECORDINGS_ROOT"/recording_* 2>/dev/null | head -n 1 || true)"
if [[ -z "$latest_recording" ]]; then
  echo "No recording directories found under: $RECORDINGS_ROOT" >&2
  exit 1
fi

imu_aux_file="$(ls -t "$latest_recording"/imu_aux_*.bin 2>/dev/null | head -n 1 || true)"
if [[ -z "$imu_aux_file" ]]; then
  echo "No imu_aux_*.bin found in: $latest_recording" >&2
  echo "Tip: run pipeline with secondary I2C IMU recording enabled." >&2
  exit 1
fi

/home/forecr/pilot_core/bin/python - "$imu_aux_file" "$OUTPUT_PATH" <<'PY'
import os
import struct
import sys

imu_path = sys.argv[1]
out_path = sys.argv[2]

FILE_HEADER_SIZE = 64
REC_HDR_FMT = "<QddIHH"
REC_HDR_SIZE = struct.calcsize(REC_HDR_FMT)

rows = []
with open(imu_path, "rb") as f:
    f.seek(FILE_HEADER_SIZE)
    while True:
        hdr = f.read(REC_HDR_SIZE)
        if len(hdr) < REC_HDR_SIZE:
            break
        seq, produced_ts, _read_ts, payload_size, _schema_ver, _dtype_tag = struct.unpack(REC_HDR_FMT, hdr)
        payload = f.read(payload_size)
        if len(payload) < payload_size:
            break

        # IMU payload layout is 16 float32 values:
        # [w, x, y, z, ax, ay, az, gx, gy, gz, mx, my, mz, throttle, steering, timestamp]
        if len(payload) >= 16 * 4:
            vals = struct.unpack_from("<16f", payload, 0)
            throttle = float(vals[13])
            steering = float(vals[14])
        else:
            throttle = float("nan")
            steering = float("nan")

        rows.append((seq, produced_ts, throttle, steering))

lines = ["seq,produced_ts,throttle,steering"]
for seq, produced_ts, throttle, steering in rows:
    lines.append(f"{seq},{produced_ts:.6f},{throttle:.6f},{steering:.6f}")
text = "\n".join(lines) + "\n"

if out_path == "-":
    sys.stdout.write(text)
else:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(text)
    print(f"saved_csv={out_path}")
    print(f"rows={len(rows)}")
    print(f"source_imu_aux={imu_path}")
PY
