#!/bin/bash
# Virtual X display for the sim. GLX here is Mesa llvmpipe; the sim renders with
# NVIDIA Vulkan (-force-vulkan) and presents into this display.
rm -f /tmp/.X99-lock
exec Xvfb :99 -screen 0 1280x720x24 +extension GLX +render -noreset
