#!/bin/bash
# AutoDRIVE RoboRacer sim on a vast.ai box (unprivileged container, no Docker):
# apt packages, the stock 2026-iros Linux builds, and supervisor services for
# Xvfb (:99) -> openbox -> x11vnc (127.0.0.1:5900) -> noVNC (127.0.0.1:6080) -> the sim.
#
#   scripts/vast_autodrive_setup.sh            # install + register + start the display stack
#   SIM_VARIANT=compete SIM_W=960 SIM_H=540 supervisorctl restart autodrive_sim   (see sim wrapper)
#
# Watch it from your machine: ssh -L 6080:127.0.0.1:6080 ... then http://localhost:6080/vnc.html
# The sim does not connect on its own: once pilot_core listens on :4567, run
#   python scripts/autodrive_sim_ctl.py connect
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${AUTODRIVE_ROOT:-/workspaces}"
SIM_DIR="$ROOT/autodrive_sim"
LOG_DIR="${AUTODRIVE_LOG_DIR:-$ROOT/logs}"
TAG="${AUTODRIVE_TAG:-2026-iros}"
URL="https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing/releases/download/$TAG"

mkdir -p "$SIM_DIR" "$LOG_DIR"

echo "== apt packages"
if ! command -v Xvfb >/dev/null || ! command -v openbox >/dev/null || ! command -v websockify >/dev/null; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        xvfb openbox x11vnc novnc websockify xdotool x11-utils imagemagick ffmpeg tmux unzip mesa-utils vulkan-tools
fi

echo "== GL / Vulkan userspace"
if command -v vast-capabilities >/dev/null; then
    if [ "$(vast-capabilities | jq -r '.hardware.gpu.render.vulkan')" != "true" ]; then
        install-display-drivers
    fi
fi

echo "== sim builds ($TAG) in $SIM_DIR"
for variant in practice compete; do
    if [ ! -x "$SIM_DIR/$variant/autodrive_simulator/AutoDRIVE Simulator.x86_64" ]; then
        dl="$ROOT/downloads"; mkdir -p "$dl"
        curl -sSL -o "$dl/${variant}_linux.zip" "$URL/autodrive_simulator_${variant}_linux.zip"
        unzip -q -o "$dl/${variant}_linux.zip" -d "$SIM_DIR/$variant"
        chmod +x "$SIM_DIR/$variant/autodrive_simulator/AutoDRIVE Simulator.x86_64"
    fi
done

echo "== openbox config (no decorations, windows at 0,0 so screen grabs are just the sim)"
mkdir -p "$SIM_DIR/openbox"
cat > "$SIM_DIR/openbox/rc.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<openbox_config xmlns="http://openbox.org/3.4/rc">
  <applications>
    <application class="*">
      <decor>no</decor>
      <position force="yes"><x>0</x><y>0</y></position>
      <focus>yes</focus>
    </application>
  </applications>
</openbox_config>
EOF

echo "== supervisor services"
conf() {   # name command [autostart]
    cat > "/etc/supervisor/conf.d/$1.conf" <<EOF
[program:$1]
environment=PROC_NAME="%(program_name)s",AUTODRIVE_ROOT="$ROOT",AUTODRIVE_LOG_DIR="$LOG_DIR"
command=$2
autostart=${3:-true}
autorestart=true
startsecs=3
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=10
stdout_logfile=$LOG_DIR/$1.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=2
redirect_stderr=true
EOF
}
chmod +x "$REPO"/scripts/supervisor/*.sh "$REPO"/scripts/supervisor/*.py
conf autodrive_xvfb   "$REPO/scripts/supervisor/autodrive_xvfb.sh"
conf autodrive_wm     "$REPO/scripts/supervisor/autodrive_wm.sh"
conf autodrive_vnc    "$REPO/scripts/supervisor/autodrive_vnc.sh"
conf autodrive_novnc  "$REPO/scripts/supervisor/autodrive_novnc.sh"
conf autodrive_sim    "$REPO/scripts/supervisor/autodrive_sim.sh"
supervisorctl reread
supervisorctl update
supervisorctl status | grep autodrive_
