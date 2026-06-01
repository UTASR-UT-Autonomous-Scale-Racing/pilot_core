#!/bin/bash
# VNC on localhost only; reach it through noVNC (6080) over an SSH tunnel.
until DISPLAY=:99 xdpyinfo >/dev/null 2>&1; do sleep 0.5; done
exec x11vnc -display :99 -localhost -rfbport 5900 -forever -shared -nopw -quiet
