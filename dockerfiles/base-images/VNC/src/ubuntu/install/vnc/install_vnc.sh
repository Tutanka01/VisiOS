#!/bin/bash
set -e
apt-get update

# Installe le nouveau serveur VNC (TigerVNC) et le client web
apt-get install -y --no-install-recommends tigervnc-standalone-server tigervnc-common tigervnc-tools novnc websockify

# Crée le lien symbolique pour noVNC
ln -s /usr/share/novnc/vnc.html /usr/share/novnc/index.html