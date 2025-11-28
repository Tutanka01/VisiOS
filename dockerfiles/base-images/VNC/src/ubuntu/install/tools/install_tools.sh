#!/bin/bash
set -e

# Installation des pré-requis pour ajouter des dépôts
apt-get update
apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    unzip \
    mousepad \
    gnome-terminal \
    xdotool \
    imagemagick \
    xfce4-screenshooter \
    x11-apps

# --- Installation de Firefox (Version .deb via PPA Mozilla Team) ---
# Ubuntu 24.04 force l'installation via Snap par défaut, ce qui ne fonctionne pas bien dans Docker.
# On utilise le PPA officiel de l'équipe Mozilla pour avoir la version .deb classique.

add-apt-repository -y ppa:mozillateam/ppa

# Prioriser le PPA par rapport à Snap
echo '
Package: *
Pin: release o=LP-PPA-mozillateam
Pin-Priority: 1001
' > /etc/apt/preferences.d/mozilla-firefox

# Configurer les mises à jour automatiques pour qu'elles n'écrasent pas avec Snap
echo 'Unattended-Upgrade::Allowed-Origins:: "LP-PPA-mozillateam:${distro_codename}";' > /etc/apt/apt.conf.d/51unattended-upgrades-firefox

apt-get update
apt-get install -y --no-install-recommends firefox