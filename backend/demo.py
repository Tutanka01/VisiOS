"""Offline stand-ins for the sandbox and the vision model.

Enabled with ``VISIOS_DEMO=1``: the UI and the agent loop run end to end
without Docker or a model server, which is handy for UI work and tests.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from typing import List, Optional, Tuple

from backend.vision_agent import AgentAction, VisionAgent

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = ImageDraw = None


class DemoSandbox:
    """Draws a fake desktop and mutates it when actions are applied."""

    def __init__(self, screen_size: Tuple[int, int] = (1600, 900)) -> None:
        self.screen_size = screen_size
        self.container_name = "visios-demo"
        self.host_vnc_port = 0
        self.running = False
        self.windows: List[str] = []
        self.typed = ""
        self.scroll = 0
        self.pointer = (screen_size[0] // 2, screen_size[1] // 2)
        self.version = 0

    # lifecycle ---------------------------------------------------------
    def start_container(self):
        if not self.running:
            time.sleep(0.4)
        self.running = True
        return self

    def stop_container(self) -> None:
        self.running = False
        self.windows.clear()
        self.typed = ""
        self.version += 1

    def status(self) -> str:
        return "running" if self.running else "exited"

    def is_running(self) -> bool:
        return self.running

    # observation -------------------------------------------------------
    def get_screenshot(self):
        time.sleep(0.15)
        return self._render(), self.screen_size

    def screen_fingerprint(self) -> str:
        return hashlib.md5(f"{self.version}".encode()).hexdigest()

    def get_desktop_context(self) -> str:
        active = self.windows[-1] if self.windows else "Bureau"
        listing = "\n".join(f"- {name}" for name in self.windows) or "- (aucune)"
        return f"active_window: {active}\nopen_windows:\n{listing}"

    # actions -----------------------------------------------------------
    def send_click(self, x, y, absolute=False, button=1, repeat=1):
        self.pointer = (int(x), int(y))
        return self.pointer

    def move_pointer(self, x, y, absolute=False):
        self.pointer = (int(x), int(y))
        return self.pointer

    def send_text(self, text: str) -> None:
        time.sleep(min(0.02 * len(text), 1.5))
        self.typed = text.replace("<ENTER>", "")
        self.version += 1

    def send_hotkey(self, keys: str) -> None:
        self.version += 1

    def send_drag(self, x, y, end_x, end_y, duration=0.4, absolute=False):
        self.pointer = (int(end_x), int(end_y))
        self.version += 1
        return self.pointer

    def send_scroll(self, delta, hint=None) -> bool:
        self.scroll += int(delta or 3)
        self.version += 1
        return True

    def open_application(self, app_name: str) -> bool:
        time.sleep(0.6)
        self.windows.append(app_name.title())
        self.version += 1
        return True

    def open_url(self, url: str) -> bool:
        time.sleep(0.8)
        self.windows.append(f"{url} — Mozilla Firefox")
        self.version += 1
        return True

    def run_shell(self, command: str, timeout: int = 30):
        time.sleep(0.3)
        return 0, f"(demo) {command}\nDesktop  Documents  Downloads"

    # rendering ---------------------------------------------------------
    def _render(self) -> bytes:
        if Image is None:
            return b""
        width, height = self.screen_size
        img = Image.new("RGB", (width, height), (38, 52, 74))
        draw = ImageDraw.Draw(img)
        for index in range(0, height, 6):
            shade = 38 + int(30 * index / height)
            draw.line([(0, index), (width, index)], fill=(shade, 52 + index // 40, 90))
        draw.rectangle([0, 0, width, 34], fill=(24, 26, 33))
        draw.text((16, 10), "Applications    Bureau VisiOS (démo)", fill=(220, 224, 232))
        for offset, name in enumerate(self.windows[-3:]):
            x0, y0 = 180 + offset * 60, 110 + offset * 50
            draw.rectangle([x0, y0, x0 + 1000, y0 + 600], fill=(245, 246, 250), outline=(90, 96, 110))
            draw.rectangle([x0, y0, x0 + 1000, y0 + 36], fill=(222, 225, 232))
            draw.text((x0 + 14, y0 + 12), name, fill=(30, 32, 40))
            if self.typed and offset == len(self.windows[-3:]) - 1:
                draw.rectangle([x0 + 40, y0 + 70, x0 + 960, y0 + 110], outline=(120, 130, 150))
                draw.text((x0 + 54, y0 + 84), self.typed, fill=(20, 20, 30))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()


DEMO_SCRIPT = [
    {
        "observation": "Le bureau XFCE est vide, aucune fenêtre ouverte.",
        "thought": "Le plus fiable pour aller sur un site est d'ouvrir directement l'URL dans le navigateur.",
        "action": "open_url",
        "url": "https://fr.wikipedia.org",
    },
    {
        "observation": "Firefox affiche la page d'accueil de Wikipédia.",
        "thought": "Je clique dans le champ de recherche en haut de la page pour y saisir la requête.",
        "action": "type",
        "x": 520,
        "y": 205,
        "text": "Intelligence artificielle",
        "submit": True,
    },
    {
        "observation": "Les résultats sont en cours de chargement.",
        "thought": "Je descends un peu pour vérifier que l'article est bien affiché.",
        "action": "scroll",
        "x": 500,
        "y": 600,
        "direction": "down",
        "amount": 4,
    },
    {
        "observation": "L'article « Intelligence artificielle » est affiché.",
        "thought": "L'objectif est atteint : l'article est ouvert et visible.",
        "action": "done",
        "summary": "J'ai ouvert Wikipédia et affiché l'article « Intelligence artificielle ».",
    },
]


class DemoVision(VisionAgent):
    """Replays a scripted trajectory, streaming it token by token."""

    def __init__(self) -> None:
        super().__init__(model="demo-vl", api_mode="demo", coord_mode="norm1000")
        self._cursor = 0

    def decide_action(
        self,
        screenshot,
        user_prompt,
        screen_size=None,
        desktop_context=None,
        recent_actions=None,
        context=None,
        on_stream=None,
        should_cancel=None,
    ) -> AgentAction:
        if context is not None and context.step == 1:
            self._cursor = 0
        payload = DEMO_SCRIPT[min(self._cursor, len(DEMO_SCRIPT) - 1)]
        self._cursor += 1
        thinking = (
            f"L'utilisateur veut : {user_prompt}.\n\n"
            f"Je regarde la capture. {payload['observation']}\n\n"
            "Quelles options ai-je ? Je pourrais passer par la souris, mais un chemin plus direct "
            "réduit le risque de rater une cible.\n\n"
            f"{payload['thought']} Je vérifierai le résultat sur la prochaine capture."
        )
        for word in thinking.split(" "):
            if should_cancel and should_cancel():
                break
            if on_stream:
                on_stream("thinking", word + " ")
            time.sleep(0.03)
        raw = json.dumps(payload, ensure_ascii=False)
        for index in range(0, len(raw), 12):
            if on_stream:
                on_stream("text", raw[index : index + 12])
            time.sleep(0.01)
        action = self._interpret(raw, thinking)
        image = self.prepare_image(screenshot, screen_size)
        return self.to_screen_space(action, image)

    def check_backend(self) -> Tuple[bool, str]:
        return True, "demo"
