#!/bin/bash
set -e

GEOMETRY="${VNC_RESOLUTION}"
DISPLAY_NUM="0"
WEBSOCKET_PORT="6901"
export DISPLAY=:${DISPLAY_NUM}

# Configuration du mot de passe VNC
mkdir -p $HOME/.vnc
echo "$VNC_PW" | tigervncpasswd -f > $HOME/.vnc/passwd
chmod 600 $HOME/.vnc/passwd

# Démarrage de TigerVNC (Xvnc)
# -ResizeRemote=1 permet le redimensionnement dynamique depuis noVNC
Xvnc :${DISPLAY_NUM} \
    -geometry ${GEOMETRY} \
    -depth 24 \
    -rfbauth $HOME/.vnc/passwd \
    -rfbport 5900 \
    -pn \
    -localhost no \
    -desktop VisiOS \
    -SecurityTypes VncAuth \
    &
sleep 2

# Démarre le bureau XFCE
startxfce4 &
sleep 2

# Lance le pont web noVNC
websockify -v --web=/usr/share/novnc/ ${WEBSOCKET_PORT} localhost:5900