#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/bin/python"
PID_FILE="${ROOT_DIR}/run/pilot_pipeline.pid"
LOG_DIR="${ROOT_DIR}/logs"
RUN_DIR="${ROOT_DIR}/run"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/pipeline_${TS}.log"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python venv not found at ${PYTHON_BIN}"
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Pipeline already running with PID ${existing_pid}"
    echo "Log file: ${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ROOT_DIR}/ORB_SLAM3/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

cd "${ROOT_DIR}"
nohup "${PYTHON_BIN}" "${ROOT_DIR}/start.py" \
  --config "${ROOT_DIR}/config.yaml" \
  --enable-slam \
  --oakd-imu \
  --recording \
  > "${LOG_FILE}" 2>&1 &

pid=$!
echo "${pid}" > "${PID_FILE}"

echo "Started production pipeline"
echo "PID: ${pid}"
echo "Log: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
