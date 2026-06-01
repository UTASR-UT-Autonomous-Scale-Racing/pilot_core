#!/bin/bash
# Stock AutoDRIVE RoboRacer sim on :99. Settings come from $AUTODRIVE_ROOT/autodrive_sim/sim.env
# (SIM_VARIANT=practice|compete, SIM_W, SIM_H) so they can be changed without touching supervisor.
ROOT="${AUTODRIVE_ROOT:-/workspaces}"
SIM_VARIANT=practice; SIM_W=640; SIM_H=360
[ -f "$ROOT/autodrive_sim/sim.env" ] && . "$ROOT/autodrive_sim/sim.env"
export DISPLAY=:99 XDG_RUNTIME_DIR=/tmp/xdg-autodrive
mkdir -p "$XDG_RUNTIME_DIR"
until xdpyinfo >/dev/null 2>&1; do sleep 0.5; done
cd "$ROOT/autodrive_sim/$SIM_VARIANT/autodrive_simulator"
echo "starting $SIM_VARIANT sim at ${SIM_W}x${SIM_H}"
# exec through a pipe: the Unity log goes through the filter (drops stack traces,
# rate-limits the reconnect storm); stdbuf keeps it line-buffered
./"AutoDRIVE Simulator.x86_64" -force-vulkan -screen-fullscreen 0 \
    -screen-width "$SIM_W" -screen-height "$SIM_H" -logFile - 2>&1 \
    | python3 -u "$(dirname "$(readlink -f "$0")")/log_filter.py"
