#!/usr/bin/env bash
# Dev launch: CPU-only profile with debug logging.  Extra args go to start.py.
set -euo pipefail
exec "$(dirname "$0")/run.sh" --config config_cpu.yaml --debug "$@"
