#!/usr/bin/env bash
# VisiOS desktop entrypoint: Xvnc (X server + VNC) -> XFCE session -> noVNC bridge.
# Runs under tini, so orphaned apps launched by the agent are reaped.
set -euo pipefail

export DISPLAY=:0
RUNTIME=/tmp/visios
RESOLUTION="${VNC_RESOLUTION:-1600x900}"

mkdir -p "$RUNTIME" "$HOME/.vnc" "$HOME/Desktop" "$HOME/Documents" "$HOME/Downloads"
export XDG_RUNTIME_DIR="$RUNTIME/xdg"
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"

# Leftovers from an unclean stop (a restarted container keeps /tmp and $HOME).
rm -f /tmp/.X0-lock /tmp/.X11-unix/X0 "$RUNTIME/session.env"
rm -f "$HOME"/.mozilla/firefox*/*/lock "$HOME"/.mozilla/firefox*/*/.parentlock 2>/dev/null || true

printf '%s\n' "${VNC_PW:-password}" | tigervncpasswd -f > "$HOME/.vnc/passwd"
chmod 600 "$HOME/.vnc/passwd"

# Fixed geometry: the agent maps model coordinates onto this exact size.
Xtigervnc :0 \
  -geometry "$RESOLUTION" -depth 24 \
  -rfbport 5900 -rfbauth "$HOME/.vnc/passwd" -SecurityTypes VncAuth \
  -AlwaysShared -AcceptSetDesktopSize=0 \
  -desktop VisiOS -nolisten tcp \
  >"$RUNTIME/xvnc.log" 2>&1 &

for _ in $(seq 1 100); do
  xset q >/dev/null 2>&1 && break
  sleep 0.1
done
xset s off -dpms s noblank >/dev/null 2>&1 || true

# One D-Bus session shared by the desktop and every `docker exec` (see
# /etc/profile.d/visios.sh): Firefox uses it to find its running instance.
eval "$(dbus-launch --sh-syntax)"
printf "export DBUS_SESSION_BUS_ADDRESS='%s'\n" "$DBUS_SESSION_BUS_ADDRESS" > "$RUNTIME/session.env"

# startxfce4 reuses the running X server and our D-Bus session, and sets the XDG
# variables (menu prefix, current desktop) the Applications menu relies on.
startxfce4 >"$RUNTIME/xfce.log" 2>&1 &

websockify --web=/usr/share/novnc 6901 localhost:5900 >"$RUNTIME/novnc.log" 2>&1 &

echo "VisiOS desktop ready on :0 (${RESOLUTION}), noVNC on port 6901"

# If a core process dies, exit so Docker's restart policy brings the desktop back.
wait -n
echo "A core desktop process exited; stopping container." >&2
exit 1
