# visiOS MVP

visiOS est un agent IA local capable de piloter un environnement de bureau Linux conteneurisé via la vision. Ce MVP combine un backend Docker, un agent de vision basé sur Ollama et une interface Streamlit pour orchestrer les interactions.

## Architecture
- **app.py** : interface Streamlit (mode large) affichant l'iframe noVNC, un chat et les logs.
- **backend/container_manager.py** : classe `SandboxManager` pour démarrer/arrêter le conteneur `tutanka01/agentic:osbase`, capturer des captures d'écran et injecter des actions via `xdotool`.
- **backend/vision_agent.py** : classe `VisionAgent` qui envoie la capture d'écran et le prompt utilisateur à Ollama (`qwen2.5-vl`) et renvoie l'action (clic ou saisie) à exécuter.

## Prérequis
1. Python 3.10+
2. Docker en local + image `tutanka01/agentic:osbase` (`docker pull tutanka01/agentic:osbase`).
3. Ollama installé et le modèle multimodal `qwen2.5-vl` téléchargé :
   ```bash
   ollama pull qwen2.5-vl
   ollama serve  # si ce n'est pas déjà lancé
   ```
4. Dépendances Python :
   ```bash
   pip install -r requirements.txt
   ```

## Configuration (.env)
Ajoutez un fichier `.env` à la racine pour personnaliser l'environnement sans modifier le code :

```
# Paramètres conteneur
VISIOS_SANDBOX_IMAGE=tutanka01/agentic:osbase
VISIOS_SANDBOX_NAME=visios-sandbox
VISIOS_SANDBOX_PORT=6901
VISIOS_SANDBOX_WIDTH=1600
VISIOS_SANDBOX_HEIGHT=900

# Endpoints et modèles IA
VISIOS_OLLAMA_URL=http://localhost:11434
VISIOS_OLLAMA_MODEL=qwen2.5-vl
VISIOS_OLLAMA_TEMPERATURE=0.1
VISIOS_OLLAMA_TIMEOUT=120

# Interface
VISIOS_IFRAME_URL=http://localhost:6901/vnc.html
```

Tous les paramètres sont optionnels : si une clé est absente, la valeur par défaut indiquée ci-dessus est utilisée.

## Démarrage
1. Lancez Streamlit :
   ```bash
   streamlit run app.py
   ```
2. Dans l'interface web :
   - Cliquez sur **Start Sandbox** pour démarrer le conteneur (le port noVNC interne est exposé sur `localhost:6901`).
   - La colonne de gauche affiche l'iframe `http://localhost:6901/vnc.html` permettant de visualiser le bureau distant.
   - Si vous préférez lancer manuellement le bureau avant Streamlit, utilisez :
     ```bash
     docker run -d -p 6901:6901 --name visios-desktop tutanka01/agentic:osbase
     ```
     (dans ce cas, vous pouvez renseigner `VISIOS_SANDBOX_NAME=visios-desktop` pour que l'application se connecte au conteneur existant).
   - La colonne de droite contient le chat et les logs d'exécution.
3. Utilisez le champ de chat pour décrire l'action souhaitée (ex : « ouvre le navigateur et va sur example.com »). Le système :
   - prend une capture du bureau,
   - interroge l'agent de vision via Ollama,
   - convertit la réponse en clic ou saisie via `xdotool` dans le conteneur.

## Arrêt/Nettoyage
- Utilisez le bouton **Stop Sandbox** dans Streamlit ou arrêtez manuellement :
  ```bash
  docker stop visios-sandbox && docker rm visios-sandbox
  ```
- Pour arrêter Streamlit, stoppez le processus (`Ctrl+C`).

## Notes
- Les captures d'écran nécessitent qu'au moins un des outils suivants soit présent dans l'image : `import` (ImageMagick), `xfce4-screenshooter` ou `xwd + convert`.
- `xdotool` doit être disponible dans le conteneur (présent dans l'image cible).
- Le système suppose que noVNC répond sur le port 6901 ; adaptez `host_vnc_port` dans `SandboxManager` si nécessaire.
