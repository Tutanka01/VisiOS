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
VISIOS_API_MODE=ollama  # ou "openai" pour LM Studio / API compatibles OpenAI
VISIOS_OPENAI_ENDPOINT=responses  # valeurs possibles: responses, chat_completions, completions
VISIOS_AGENT_MAX_STEPS=3
VISIOS_ACTION_DELAY=1.0  # secondes entre chaque action de l'agent
VISIOS_REPEAT_TOLERANCE=2  # nombre de répétitions identiques autorisées avant arrêt
VISIOS_INPUT_BACKEND=xdotool  # ou "vncdotool"
VISIOS_VNC_HOST=127.0.0.1
VISIOS_VNC_PORT=5901
Ajustez `VISIOS_AGENT_MAX_STEPS`, `VISIOS_ACTION_DELAY` et `VISIOS_REPEAT_TOLERANCE` pour contrôler le nombre d'actions consécutives effectuées par l'agent, le délai entre deux actions et le seuil avant de considérer que le modèle boucle.

# Interface
VISIOS_IFRAME_URL=http://localhost:6901/vnc.html
```

Tous les paramètres sont optionnels : si une clé est absente, la valeur par défaut indiquée ci-dessus est utilisée. `VISIOS_API_MODE=openai` permet de cibler LM Studio ou toute API compatible OpenAI ; utilisez `VISIOS_OPENAI_ENDPOINT=chat_completions` pour les serveurs qui implémentent principalement `/v1/chat/completions` (comme LM Studio) ou `responses` pour l'API `/v1/responses`. Ajustez `VISIOS_AGENT_MAX_STEPS` et `VISIOS_ACTION_DELAY` pour contrôler le nombre d'actions consécutives effectuées par l'agent et le délai entre deux actions.

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
- Si vous préférez piloter l'affichage via `vncdotool`, définissez `VISIOS_INPUT_BACKEND=vncdotool` et exposez le serveur VNC (127.0.0.1:5901 par défaut).
- Les clics sont recalibrés à chaque action via `xdotool getdisplaygeometry`, puis clampés à la surface détectée afin d'éviter les décalages lorsque la résolution change à chaud.
- Les actions `scroll` produisent désormais des événements de molette (boutons 4/5) et interprètent la valeur de `coordinates.y` ainsi que les indications textuelles (`scroll down`, `vers le haut`, `page down`, etc.) pour ajuster sens et intensité.
- Le système suppose que noVNC répond sur le port 6901 ; adaptez `host_vnc_port` dans `SandboxManager` si nécessaire.
- Pour simuler des raccourcis clavier, incluez des tokens entre chevrons dans vos réponses (ex. `<CTRL+L>`, `<ENTER>`). Ils sont interprétés par l'agent et envoyés via `xdotool key` avant/après les chaînes tapées.
- Les modèles qui exposent un mode *reasoning* ou des balises `<think>` sont détectés automatiquement. Les réflexions sont affichées dans les logs Streamlit et dans l'historique des actions, sans configuration supplémentaire.
