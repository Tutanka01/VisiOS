# VisiOS

VisiOS est un agent IA local qui pilote un bureau Linux conteneurisé par la vision : tu écris ce que tu veux, il regarde l'écran, agit avec la souris et le clavier, vérifie l'effet de chaque action et continue jusqu'à ce que l'objectif soit atteint.

## Architecture

- **server.py** : cockpit FastAPI. Sert l'interface web et diffuse en direct (WebSocket) chaque événement de l'agent.
- **web/** : interface (HTML/CSS/JS sans build). Bureau noVNC en direct avec curseur fantôme de l'agent, narration de sa pensée en streaming, fil de conversation, confirmations, précisions en cours de tâche, bouton d'arrêt.
- **backend/agent_runner.py** : boucle `observer → décider → agir → attendre la stabilité → constater l'effet`, avec anti-boucle, confirmations humaines, notes utilisateur, arrêt, et trajectoires dans `trajectories/<run_id>/`.
- **backend/vision_agent.py** : construit le prompt, prépare la capture, appelle le modèle (Ollama ou API compatible OpenAI, en streaming) et convertit la réponse en action en pixels écran.
- **backend/container_manager.py** : `SandboxManager` démarre le conteneur, capture l'écran, calcule une empreinte d'écran et injecte les actions via `xdotool`.
- **backend/demo.py** : faux bureau et faux modèle (`VISIOS_DEMO=1`) pour essayer l'interface sans Docker ni modèle.
- **app.py** : ancien cockpit Streamlit, conservé mais remplacé par `server.py`.

## Démarrage rapide

```bash
pip install -r requirements.txt
python server.py            # puis ouvre http://localhost:8501
```

Pour essayer l'interface sans Docker ni modèle :

```bash
VISIOS_DEMO=1 python server.py
```

Avec Docker Compose (cockpit + bureau) :

```bash
docker compose up --build   # http://localhost:8501
```

Prérequis du mode réel :
1. Docker, avec l'image du bureau : `docker compose build visios-sandbox` (ou `docker build -t visios-sandbox:local dockerfiles/sandbox`).
2. Un modèle de vision servi localement, par exemple LM Studio avec `qwen/qwen3-vl-8b` (`VISIOS_API_MODE=openai`, `VISIOS_OPENAI_ENDPOINT=chat_completions`, `VISIOS_OLLAMA_URL=http://localhost:1234`) ou Ollama (`VISIOS_API_MODE=ollama`, `VISIOS_OLLAMA_URL=http://localhost:11434`).

## Image du bureau (`dockerfiles/sandbox`)

- **Debian bookworm-slim, native amd64 et arm64** : sur un Mac Apple Silicon, le bureau tourne sans émulation, ce qui le rend nettement plus rapide et évite les plantages de Firefox.
- **XFCE minimal** sans compositeur, ni écran de veille, ni animations : l'écran ne bouge que quand quelque chose se passe vraiment. Le panneau du haut contient le menu Applications, des lanceurs (Firefox, Terminal, Fichiers, Éditeur), les fenêtres ouvertes et l'horloge.
- **Firefox ESR préconfiguré pour un agent** : aucune page de premier lancement ni proposition de navigateur par défaut, DuckDuckGo par défaut (pas de mur de consentement), bannières cookies refusées automatiquement, pas de restauration après plantage, défilement non lissé, curseur texte fixe, rendu logiciel.
- **Une seule session D-Bus**, partagée par le bureau et les `docker exec` de l'agent (`/etc/profile.d/visios.sh`). Firefox retrouve ainsi toujours sa fenêtre ouverte au lieu d'afficher « déjà en cours d'exécution ». Le lanceur `/usr/local/bin/firefox` supprime aussi les verrous de profil laissés par un plantage.
- `tini` comme init (les applications lancées par l'agent sont nettoyées), un healthcheck, et un redémarrage automatique si Xvnc, XFCE ou noVNC s'arrête.
- Outils de l'agent : `xdotool`, `wmctrl`, `scrot` (captures rapides), `xwd` (empreinte d'écran), plus `python3`, `curl` et `sudo` sans mot de passe pour `run_command`.
- Interface en anglais par défaut, qui correspond aux interfaces sur lesquelles les modèles de vision sont surtout entraînés. Pour un bureau en français : `VISIOS_DESKTOP_LANG=fr_FR.UTF-8 docker compose build visios-sandbox`.

L'ancienne image (`dockerfiles/base-images/VNC`, `Dockerfile.sandbox`) n'est plus utilisée par le compose.

## Ce qui rend l'agent fiable

- **Un seul système de coordonnées, choisi selon le modèle** (`VISIOS_COORD_MODE=auto`). Qwen3-VL, UI-TARS et la plupart des VLM récents raisonnent sur une grille 0-1000. Qwen2.5-VL répond en pixels de l'image reçue : la capture lui est envoyée à une taille multiple de 28, pour que le serveur ne la redimensionne pas une seconde fois. L'agent convertit ensuite en pixels écran absolus.
- **Image maîtrisée** : la capture est redimensionnée à `VISIOS_MODEL_IMAGE_WIDTH` (1280 par défaut). Cela accélère l'inférence et rend la conversion des coordonnées exacte.
- **Retour d'effet** : après chaque action, VisiOS attend que l'écran réagisse puis se stabilise (empreinte `xwd | md5sum` dans le conteneur) au lieu de dormir une durée fixe. Il dit ensuite au modèle si l'écran a changé. Une action sans effet est signalée, et une répétition sans effet est interdite avant l'arrêt anti-boucle.
- **Historique raisonné** : à chaque tour, le modèle reçoit ses actions précédentes, leur intention, leur effet, la sortie des commandes et les précisions envoyées en cours de route.
- **Actions plus riches et plus directes** : `open_url` (navigateur directement sur l'URL), `run_command` (shell dans le bac à sable, sortie relue au tour suivant), `double_click`, `right_click`, `type` avec focus et `submit`, séquences de touches.
- **Réponses robustes** : format JSON `observation / thought / action`, parsing tolérant (bloc de code, virgule finale, texte autour, `coordinate: [x, y]`, bbox), et une nouvelle tentative automatique si la réponse est illisible.

## Modèles thinking

La réflexion du modèle s'affiche en direct sous chaque étape, puis se replie (« A réfléchi 6,2 s · 412 mots »).

- Dès que le modèle est identifié comme thinking, VisiOS passe à température 0.6, top_p 0.95, top_k 20 : un décodage quasi glouton (0.1) fait tourner ces modèles en boucle.
- En mode **Limitée** (par défaut), une réflexion qui dépasse le budget ou se répète est coupée. Le modèle reçoit alors ses notes et doit répondre directement, sans réfléchir.
- En mode **Désactivée**, VisiOS demande au serveur de sauter la réflexion (`think: false` pour Ollama, `chat_template_kwargs.enable_thinking=false` et `/no_think` pour Qwen). Selon le serveur, cette demande peut être ignorée.

## Actions disponibles

| Action | Exemple |
| --- | --- |
| `click`, `double_click`, `right_click`, `move` | `{"action":"click","x":500,"y":300}` |
| `type` | `{"action":"type","text":"météo Paris","x":500,"y":80,"submit":true}` |
| `key` | `{"action":"key","keys":"ctrl+l"}` ou `"ctrl+a BackSpace"` |
| `scroll` | `{"action":"scroll","x":500,"y":600,"direction":"down","amount":5}` |
| `drag` | `{"action":"drag","x":100,"y":100,"to_x":400,"to_y":400}` |
| `open_app` | `browser`, `terminal`, `files`, `text_editor`, `settings`, `calculator` |
| `open_url` | `{"action":"open_url","url":"https://fr.wikipedia.org"}` |
| `run_command` | `{"action":"run_command","command":"df -h"}` |
| `wait`, `done`, `fail` | `{"action":"done","summary":"…"}` |

## Prudence

- **Équilibrée** (par défaut) : l'agent demande ton accord dans l'interface avant toute action qu'il juge sensible (paiement, envoi, suppression, mot de passe) et avant une commande shell destructrice.
- **Supervisée** : les demandes contenant des mots sensibles sont bloquées et chaque commande shell attend ton accord.
- **Rapide** : aucune confirmation.

`VISIOS_ALLOW_SHELL=false` désactive complètement `run_command`.

## Configuration (.env)

Toutes les clés sont optionnelles.

```
# Bureau
VISIOS_SANDBOX_IMAGE=visios-sandbox:local
VISIOS_DESKTOP_LANG=en_US.UTF-8     # langue du bureau (au build de l'image)
VISIOS_SANDBOX_NAME=visios-sandbox
VISIOS_SANDBOX_PORT=6901
VISIOS_SANDBOX_WIDTH=1600
VISIOS_SANDBOX_HEIGHT=900
VISIOS_SANDBOX_PLATFORM=            # vide = architecture native (recommandé)
VISIOS_VNC_PASSWORD=password

# Modèle
VISIOS_API_MODE=openai              # ou ollama
VISIOS_OPENAI_ENDPOINT=chat_completions   # chat_completions, responses, completions
VISIOS_OLLAMA_URL=http://localhost:1234
VISIOS_OLLAMA_MODEL=qwen/qwen3-vl-8b
VISIOS_OLLAMA_TEMPERATURE=0.1
VISIOS_OLLAMA_TIMEOUT=120
VISIOS_OLLAMA_NUM_CTX=16384         # Ollama : contexte suffisant pour l'image
VISIOS_API_KEY=                     # si l'API compatible OpenAI en exige une
VISIOS_COORD_MODE=auto              # auto, norm1000, pixel
VISIOS_MODEL_IMAGE_WIDTH=1280
VISIOS_MAX_TOKENS=                  # limite de génération (vide = pas de limite)
VISIOS_THINKING=budget              # modèles thinking : full, budget (coupe si trop long ou en boucle), off
VISIOS_THINKING_BUDGET=6000         # caractères de réflexion max par étape (~1500 tokens)
VISIOS_THINKING_TEMPERATURE=0.6     # température utilisée quand le modèle réfléchit (recommandation Qwen)
VISIOS_MODEL_THINKS=auto            # auto (déduit du nom puis des réponses), true, false

# Agent
VISIOS_AGENT_MAX_STEPS=25
VISIOS_ACTION_DELAY=0.3
VISIOS_REPEAT_TOLERANCE=2           # répétitions sans effet tolérées avant arrêt
VISIOS_SAFETY_MODE=balanced         # balanced, strict, fast
VISIOS_ALLOW_SHELL=true
VISIOS_TYPE_DELAY_MS=12
VISIOS_COMMAND_TIMEOUT=30
VISIOS_TRAJECTORY_DIR=trajectories
VISIOS_SAVE_SCREENSHOTS=true
VISIOS_INPUT_BACKEND=xdotool
VISIOS_VNC_PORT=5900

# Cockpit
VISIOS_HOST=127.0.0.1
VISIOS_PORT=8501
VISIOS_IFRAME_URL=http://localhost:6901/vnc_lite.html
VISIOS_DISPLAY_MODE=vnc             # vnc (bureau en direct) ou screenshots
VISIOS_DEMO=0
```

## Tests

```bash
python -m unittest discover -s tests
```

## Arrêt

Bouton **Éteindre** dans les réglages, ou `docker stop visios-sandbox && docker rm visios-sandbox`.
