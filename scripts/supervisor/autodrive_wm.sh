#!/bin/bash
# Unity only receives (xdotool) input in a focused window, which needs a window manager.
export DISPLAY=:99
until xdpyinfo >/dev/null 2>&1; do sleep 0.5; done
exec openbox --config-file "${AUTODRIVE_ROOT:-/workspaces}/autodrive_sim/openbox/rc.xml"
