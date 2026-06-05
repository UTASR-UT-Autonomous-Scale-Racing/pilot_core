#!/bin/bash
# Stock AutoDRIVE RoboRacer sim on :99. Settings come from $AUTODRIVE_ROOT/autodrive_sim/sim.env
# (SIM_VARIANT=practice|compete, SIM_W, SIM_H, SIM_GFX, SIM_ARGS, SIM_NICE, SIM_VK_ICD)
# so they can be changed without touching supervisor. The stock build ships Vulkan
# shaders only (-force-glcore fails); SIM_VK_ICD picks the Vulkan driver. Mesa lavapipe
# (/usr/share/vulkan/icd.d/lvp_icd.json) renders the stock build magenta, so the sim needs
# the NVIDIA driver (docs/autodrive.md).
ROOT="${AUTODRIVE_ROOT:-/workspaces}"
SIM_VARIANT=practice; SIM_W=640; SIM_H=360; SIM_GFX=vulkan; SIM_ARGS=""; SIM_NICE=0; SIM_VK_ICD=""
[ -f "$ROOT/autodrive_sim/sim.env" ] && . "$ROOT/autodrive_sim/sim.env"
export DISPLAY=:99 XDG_RUNTIME_DIR=/tmp/xdg-autodrive
mkdir -p "$XDG_RUNTIME_DIR"
[ -n "$SIM_VK_ICD" ] && export VK_ICD_FILENAMES="$SIM_VK_ICD"
until xdpyinfo >/dev/null 2>&1; do sleep 0.5; done
cd "$ROOT/autodrive_sim/$SIM_VARIANT/autodrive_simulator"
echo "starting $SIM_VARIANT sim at ${SIM_W}x${SIM_H}, gfx=$SIM_GFX ${SIM_VK_ICD:+(ICD $SIM_VK_ICD)}, nice=$SIM_NICE, args='$SIM_ARGS'"
# exec through a pipe: the Unity log goes through the filter (drops stack traces,
# rate-limits the reconnect storm); stdbuf keeps it line-buffered
nice -n "$SIM_NICE" ./"AutoDRIVE Simulator.x86_64" -force-"$SIM_GFX" -screen-fullscreen 0 \
    -screen-width "$SIM_W" -screen-height "$SIM_H" $SIM_ARGS -logFile - 2>&1 \
    | python3 -u "$(dirname "$(readlink -f "$0")")/log_filter.py"
