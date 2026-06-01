#!/bin/bash
# noVNC web client on 127.0.0.1:6080 (http://localhost:6080/vnc.html through the tunnel).
exec websockify --web /usr/share/novnc 127.0.0.1:6080 127.0.0.1:5900
