# Desktop session environment for login shells, including `docker exec bash -lc`.
# Without it, apps launched by the agent would not share the desktop's D-Bus
# session (Firefox then fails with "already running, but is not responding").
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR=/tmp/visios/xdg
export NO_AT_BRIDGE=1
if [ -r /tmp/visios/session.env ]; then
  . /tmp/visios/session.env
fi
