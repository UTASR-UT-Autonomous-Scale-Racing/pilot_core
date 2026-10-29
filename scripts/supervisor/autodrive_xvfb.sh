#!/bin/bash
# Virtual X display for the sim. GLX here is Mesa llvmpipe; the sim renders with
# NVIDIA Vulkan (-force-vulkan) and presents into this display. Size from
# $AUTODRIVE_ROOT/autodrive_sim/sim.env (SCREEN_W, SCREEN_H); it must hold the sim window
# (SIM_W x SIM_H) or the HUD buttons autodrive_sim_ctl.py clicks fall off-screen.
ROOT="${AUTODRIVE_ROOT:-/workspaces}"
SCREEN_W=1280; SCREEN_H=720
[ -f "$ROOT/autodrive_sim/sim.env" ] && . "$ROOT/autodrive_sim/sim.env"
rm -f /tmp/.X99-lock
exec Xvfb :99 -screen 0 "${SCREEN_W}x${SCREEN_H}x24" +extension GLX +render -noreset
