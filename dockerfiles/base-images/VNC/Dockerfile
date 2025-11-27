# ÉTAPE 1: Construction de la base
ARG BASE_IMAGE="ubuntu:24.04"
FROM $BASE_IMAGE AS base_layer

# Arguments de construction pour la personnalisation
ARG BG_IMG=bg_visios.png

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive \
    LANG='fr_FR.UTF-8' \
    LANGUAGE='fr_FR:fr' \
    LC_ALL='fr_FR.UTF-8' \
    TZ='Europe/Paris' \
    HOME=/home/visios-user \
    STARTUPDIR=/dockerstartup \
    INST_SCRIPTS=/dockerstartup/install

# Création de l'utilisateur
RUN groupmod -n visios-user $(getent group 1000 | cut -d: -f1) && \
    usermod -l visios-user -d /home/visios-user -m $(getent passwd 1000 | cut -d: -f1) && \
    mkdir -p ${HOME}/Desktop && \
    chown -R visios-user:visios-user ${HOME}

WORKDIR $HOME

# Installation des dépendances et locales
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales && \
    locale-gen fr_FR.UTF-8

# Installation des outils et du bureau XFCE
COPY src/ubuntu/install/xfce/ $INST_SCRIPTS/xfce/
RUN find $INST_SCRIPTS/xfce/ -name "*.sh" -exec sed -i 's/\r$//' {} + && \
    bash $INST_SCRIPTS/xfce/install_xfce_ui.sh && \
    rm -rf $INST_SCRIPTS/xfce/

# Installation des outils supplémentaires (Firefox, etc.)
COPY src/ubuntu/install/tools/ $INST_SCRIPTS/tools/
RUN find $INST_SCRIPTS/tools/ -name "*.sh" -exec sed -i 's/\r$//' {} + && \
    bash $INST_SCRIPTS/tools/install_tools.sh && \
    rm -rf $INST_SCRIPTS/tools/

# --- MODIFIÉ : Installation de TigerVNC et noVNC ---
COPY src/ubuntu/install/vnc/ $INST_SCRIPTS/vnc/
RUN find $INST_SCRIPTS/vnc/ -name "*.sh" -exec sed -i 's/\r$//' {} + && \
    bash $INST_SCRIPTS/vnc/install_vnc.sh && \
    rm -rf $INST_SCRIPTS/vnc/

# Configuration du bureau et des icônes
COPY src/common/resources/config/ $HOME/.config/
RUN chown -R visios-user:visios-user $HOME/.config
ADD src/common/resources/images/$BG_IMG /usr/share/backgrounds/bg_default.png
ADD src/common/resources/images/icon_visios.png /usr/share/icons/icon_default.png

# Configuration des scripts de démarrage
COPY src/common/startup_scripts/ $STARTUPDIR/
RUN find $STARTUPDIR -name "*.sh" -exec sed -i 's/\r$//' {} + && \
    chmod +x -R $STARTUPDIR && \
    chown -R visios-user:visios-user $STARTUPDIR

# Script de nettoyage final
COPY src/ubuntu/install/cleanup/ $INST_SCRIPTS/cleanup/
RUN find $INST_SCRIPTS/cleanup/ -name "*.sh" -exec sed -i 's/\r$//' {} + && \
    bash $INST_SCRIPTS/cleanup/cleanup.sh && \
    rm -rf $INST_SCRIPTS/cleanup/

# ---
# ÉTAPE 2: Création de l'image finale
FROM scratch
COPY --from=base_layer / /

# Labels de l'image
LABEL org.opencontainers.image.authors="visios.fr" \
      org.opencontainers.image.title="Desktop VisiOS" \
      org.opencontainers.image.description="Environnement de bureau distant pour VisiOS."

# Variables d'environnement pour l'exécution
ENV HOME=/home/visios-user \
    VNC_PW=password \
    VNC_RESOLUTION=1600x900 \
    LANG='fr_FR.UTF-8' \
    TZ='Europe/Paris'

EXPOSE 6901

USER visios-user
WORKDIR $HOME

ENTRYPOINT [ "/dockerstartup/startup.sh" ]