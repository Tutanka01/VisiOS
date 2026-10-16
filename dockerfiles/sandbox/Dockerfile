# syntax=docker/dockerfile:1.6
#
# VisiOS sandbox desktop: Debian + minimal XFCE + Firefox ESR, served over VNC/noVNC.
# Built for the agent: native multi-arch (no emulation on Apple Silicon), no
# compositor, no blinking caret, no first-run dialogs, one shared D-Bus session.
#
#   docker build -t visios-sandbox:local dockerfiles/sandbox
#
FROM debian:bookworm-slim

ARG DESKTOP_LANG=en_US.UTF-8
ARG TZ=Europe/Paris
ARG USER_NAME=visios-user
ARG USER_UID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=${DESKTOP_LANG} \
    LC_ALL=${DESKTOP_LANG} \
    TZ=${TZ} \
    DISPLAY=:0 \
    VNC_RESOLUTION=1600x900 \
    VNC_PW=password \
    NO_AT_BRIDGE=1 \
    MOZ_ENABLE_WAYLAND=0 \
    MOZ_CRASHREPORTER_DISABLE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tini dbus-x11 locales tzdata ca-certificates sudo procps \
      tigervnc-standalone-server tigervnc-tools novnc python3-websockify \
      xfce4-session xfwm4 xfce4-panel xfdesktop4 xfce4-settings xfconf \
      thunar xfce4-terminal mousepad galculator adwaita-icon-theme \
      fonts-dejavu-core fonts-liberation2 \
      firefox-esr \
      xdotool wmctrl scrot x11-apps x11-xserver-utils xdg-utils \
      curl nano less unzip python3 \
 && sed -i -e "s/^# *\(${DESKTOP_LANG} \)/\1/" -e "s/^# *\(fr_FR.UTF-8 \)/\1/" /etc/locale.gen \
 && locale-gen \
 && ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime && echo "${TZ}" > /etc/timezone \
 && ln -sf vnc.html /usr/share/novnc/index.html \
 && find /usr/share/locale -mindepth 1 -maxdepth 1 ! -name 'en*' ! -name 'fr*' ! -name 'locale.alias' -exec rm -rf {} + \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /usr/share/doc/* /usr/share/man/* /usr/share/info/*

COPY rootfs/ /

# User with passwordless sudo (the container itself is the sandbox), X socket dir,
# Firefox enterprise policies and agent-friendly default prefs.
RUN useradd --create-home --uid "${USER_UID}" --shell /bin/bash "${USER_NAME}" \
 && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/visios \
 && chmod 0440 /etc/sudoers.d/visios \
 && chmod 0755 /opt/visios/start.sh /usr/local/bin/firefox \
 && mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix \
 && mkdir -p /usr/lib/firefox-esr/distribution \
 && cp /etc/firefox/policies/policies.json /usr/lib/firefox-esr/distribution/policies.json \
 && cat /opt/visios/firefox-prefs.js >> /etc/firefox-esr/firefox-esr.js

COPY --chown=${USER_UID}:${USER_UID} home/ /home/${USER_NAME}/

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}
RUN mkdir -p Desktop Documents Downloads

EXPOSE 5900 6901

HEALTHCHECK --interval=10s --timeout=4s --start-period=20s --retries=3 \
  CMD xset -display :0 q >/dev/null 2>&1 && curl -fsS -o /dev/null http://127.0.0.1:6901/vnc_lite.html || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/visios/start.sh"]
