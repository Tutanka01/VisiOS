# Dockerfile pour l'application VisiOS (Streamlit)
FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Exposition du port du cockpit
EXPOSE 8501
ENV VISIOS_HOST=0.0.0.0 \
    VISIOS_PORT=8501

# Commande de démarrage
CMD ["python", "server.py"]
