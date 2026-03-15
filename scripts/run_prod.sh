#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
PID_FILE="${ROOT_DIR}/run/pilot_pipeline.pid"
LOG_DIR="${ROOT_DIR}/logs"
RUN_DIR="${ROOT_DIR}/run"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/pipeline_${TS}.log"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  # Backward-compat fallback for older setups.
  if [[ -x "${ROOT_DIR}/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/bin/python"
  else
    echo "Python venv not found at ${ROOT_DIR}/.venv/bin/python"
    exit 1
  fi
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
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ROOT_DIR}/ORB_SLAM3/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

cd "${ROOT_DIR}"

START_ARGS=(
  --config "${ROOT_DIR}/config.yaml"
  --enable-slam
  --oakd-imu
  --recording
)

# Optional camera tuning passthrough (maps env vars to start.py flags).
[[ -n "${PILOT_DEPTH_PRESET:-}" ]] && START_ARGS+=(--depth-preset "${PILOT_DEPTH_PRESET}")
[[ -n "${PILOT_STEREO_RESOLUTION:-}" ]] && START_ARGS+=(--stereo-resolution "${PILOT_STEREO_RESOLUTION}")
[[ -n "${PILOT_SYNC_THRESHOLD_MS:-}" ]] && START_ARGS+=(--sync-threshold-ms "${PILOT_SYNC_THRESHOLD_MS}")
[[ -n "${PILOT_IR_DOT_INTENSITY:-}" ]] && START_ARGS+=(--ir-dot-intensity "${PILOT_IR_DOT_INTENSITY}")
[[ -n "${PILOT_IR_FLOOD_INTENSITY:-}" ]] && START_ARGS+=(--ir-flood-intensity "${PILOT_IR_FLOOD_INTENSITY}")
[[ -n "${PILOT_XLINK_CHUNK_SIZE:-}" ]] && START_ARGS+=(--xlink-chunk-size "${PILOT_XLINK_CHUNK_SIZE}")
[[ -n "${PILOT_IR_UPDATE_INTERVAL_S:-}" ]] && START_ARGS+=(--ir-update-interval-s "${PILOT_IR_UPDATE_INTERVAL_S}")
if [[ -n "${PILOT_NO_ACTUATOR_I2C:-}" ]]; then
  if [[ "${PILOT_NO_ACTUATOR_I2C}" == "1" || "${PILOT_NO_ACTUATOR_I2C}" == "true" || "${PILOT_NO_ACTUATOR_I2C}" == "TRUE" ]]; then
    START_ARGS+=(--no-actuator-i2c)
  fi
fi

if [[ -n "${PILOT_ADAPTIVE_IR:-}" ]]; then
  if [[ "${PILOT_ADAPTIVE_IR}" == "1" || "${PILOT_ADAPTIVE_IR}" == "true" || "${PILOT_ADAPTIVE_IR}" == "TRUE" ]]; then
    START_ARGS+=(--adaptive-ir)
  else
    START_ARGS+=(--no-adaptive-ir)
  fi
fi

if [[ -n "${PILOT_RGB_UNDISTORT:-}" ]]; then
  if [[ "${PILOT_RGB_UNDISTORT}" == "1" || "${PILOT_RGB_UNDISTORT}" == "true" || "${PILOT_RGB_UNDISTORT}" == "TRUE" ]]; then
    START_ARGS+=(--rgb-undistort)
  else
    START_ARGS+=(--no-rgb-undistort)
  fi
fi

nohup "${PYTHON_BIN}" "${ROOT_DIR}/start.py" "${START_ARGS[@]}" > "${LOG_FILE}" 2>&1 &

pid=$!
echo "${pid}" > "${PID_FILE}"

echo "Started production pipeline"
echo "PID: ${pid}"
echo "Log: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
