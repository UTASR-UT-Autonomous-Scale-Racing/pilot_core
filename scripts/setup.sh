#!/usr/bin/env bash
# Initialise all git submodules.
set -euo pipefail
cd "$(dirname "$0")/.."
git submodule update --init --recursive
