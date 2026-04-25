#!/usr/bin/env bash
# init all submodules (pilot_vision, firmware, ...)
#   scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
git submodule update --init --recursive
