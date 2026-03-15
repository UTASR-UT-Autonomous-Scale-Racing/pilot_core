#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${ROOT_DIR}/run/pilot_pipeline.pid"

stop_pid() {
  local pid="$1"
  if ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi

  echo "Sending SIGINT to PID ${pid}"
  kill -INT "${pid}" || true

  for _ in $(seq 1 20); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done

  echo "SIGINT timeout; sending SIGTERM to PID ${pid}"
  kill -TERM "${pid}" || true

  for _ in $(seq 1 10); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done

  echo "SIGTERM timeout; sending SIGKILL to PID ${pid}"
  kill -KILL "${pid}" || true
  sleep 1
}

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${pid}" ]]; then
    stop_pid "${pid}"
  fi
  rm -f "${PID_FILE}"
  echo "Shutdown complete (from PID file)"
  exit 0
fi

echo "No PID file found at ${PID_FILE}."
echo "If needed, stop manually with: pkill -f 'start.py.*--oakd-imu'"
