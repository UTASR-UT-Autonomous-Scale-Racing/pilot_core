#!/usr/bin/env bash
# workstation launch: config_cpu.yaml (real OAK-D, no actuator), debug logging on
#   scripts/dev_run.sh [extra start.py args, e.g. --enable-slam]
set -euo pipefail
exec "$(dirname "$0")/run.sh" --config config_cpu.yaml --debug "$@"
