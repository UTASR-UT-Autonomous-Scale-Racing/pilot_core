#!/usr/bin/env bash
# Workstation launch: config_cpu.yaml (real OAK-D, no actuator) with debug logging.
#   scripts/dev_run.sh [extra start.py args, e.g. --enable-slam]
set -euo pipefail
exec "$(dirname "$0")/run.sh" --config config_cpu.yaml --debug "$@"
