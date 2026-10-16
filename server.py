"""VisiOS cockpit server: FastAPI + WebSocket live event stream.

Run with ``python server.py`` (or ``uvicorn server:app``) and open
http://localhost:8501. Set ``VISIOS_DEMO=1`` to try the UI without Docker
or a model server.
"""
from __future__ import annotations

import asyncio
import base64
import datetime as dt
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from backend.agent_runner import AgentRunConfig, AgentRunner  # noqa: E402
from backend.vision_agent import AgentAction, VisionAgent  # noqa: E402

LOGGER = logging.getLogger("visios.server")
logging.basicConfig(level=os.getenv("VISIOS_LOG_LEVEL", "INFO"))

WEB_DIR = Path(__file__).parent / "web"
DEMO_MODE = os.getenv("VISIOS_DEMO", "").lower() in {"1", "true", "yes", "on"}
DEFAULT_VNC_URL = "http://localhost:6901/vnc_lite.html"
VNC_PASSWORD = os.getenv("VISIOS_VNC_PASSWORD", os.getenv("VNC_PW", "password"))
DISPLAY_MODE = os.getenv("VISIOS_DISPLAY_MODE", "screenshots" if DEMO_MODE else "vnc")
MAX_HISTORY_EVENTS = 1500
CONVERSATION_TURNS = 4


def build_vnc_url(raw_url: str) -> str:
    """Add noVNC defaults that make the desktop feel embedded, not bolted on."""

    parts = urlsplit(raw_url or DEFAULT_VNC_URL)
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    defaults = {
        "autoconnect": "true",
        "resize": "scale",
        "scale": "true",
        "reconnect": "true",
        "reconnect_delay": "1000",
        "quality": "9",
        "compression": "2",
        "show_dot": "true",
        "view_only": "false",
    }
    if VNC_PASSWORD and "password" not in params:
        defaults["password"] = VNC_PASSWORD
    merged = {**defaults, **params}
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(merged), parts.fragment))


def _frame_from_png(png: bytes, width: int = 1280) -> Optional[str]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(png)) as img:
            img = img.convert("RGB")
            if img.size[0] > width:
                img = img.resize((width, int(img.size[1] * width / img.size[0])), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


class Hub:
    """Owns the sandbox, the current run and the connected browsers."""

    def __init__(self) -> None:
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.clients: Set[WebSocket] = set()
        self.history: List[Dict[str, Any]] = []
        self.conversation: List[Dict[str, str]] = []
        self.runner: Optional[AgentRunner] = None
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self._confirm_event = threading.Event()
        self._confirm_result = False
        self._sandbox = None
        self._sandbox_error = ""
        self.vision = self._make_vision()
        self.settings: Dict[str, Any] = {
            "max_steps": AgentRunConfig().max_steps,
            "safety_mode": AgentRunConfig().safety_mode,
            "display_mode": DISPLAY_MODE,
            "thinking": self.vision.thinking_mode,
        }

    # -- components ------------------------------------------------------
    @staticmethod
    def _make_vision() -> VisionAgent:
        if DEMO_MODE:
            from backend.demo import DemoVision

            return DemoVision()
        return VisionAgent()

    @property
    def sandbox(self):
        if self._sandbox is None:
            try:
                if DEMO_MODE:
                    from backend.demo import DemoSandbox

                    self._sandbox = DemoSandbox()
                else:
                    from backend.container_manager import SandboxManager

                    self._sandbox = SandboxManager()
                self._sandbox_error = ""
            except Exception as exc:
                self._sandbox_error = f"Docker indisponible : {exc}"
                raise RuntimeError(self._sandbox_error) from exc
        return self._sandbox

    @property
    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def status(self) -> Dict[str, Any]:
        try:
            sandbox_status = self.sandbox.status()
        except Exception:
            sandbox_status = "unavailable"
        backend_ok, backend_detail = self.vision.check_backend()
        return {
            "sandbox": sandbox_status,
            "sandbox_error": self._sandbox_error,
            "model_ok": backend_ok,
            "model_detail": backend_detail,
            "running": self.running,
        }

    def config(self) -> Dict[str, Any]:
        screen = getattr(self._sandbox, "screen_size", None) or (
            int(os.getenv("VISIOS_SANDBOX_WIDTH", 1600)),
            int(os.getenv("VISIOS_SANDBOX_HEIGHT", 900)),
        )
        return {
            "demo": DEMO_MODE,
            "vnc_url": build_vnc_url(os.getenv("VISIOS_IFRAME_URL", DEFAULT_VNC_URL)),
            "model": self.vision.model,
            "api_mode": self.vision.api_mode,
            "coord_mode": self.vision.coord_mode,
            "screen": list(screen),
            "settings": self.settings,
        }

    # -- broadcasting ----------------------------------------------------
    def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Thread-safe: may be called from the agent thread."""

        event = {"type": event_type, "ts": time.time(), **payload}
        # Streaming deltas are numerous; keep them out of the replay buffer.
        if event_type not in {"model_stream", "frame"}:
            with self.lock:
                self.history.append(event)
                if len(self.history) > MAX_HISTORY_EVENTS:
                    self.history = self.history[-MAX_HISTORY_EVENTS:]
        if self.loop is not None:
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._broadcast(event)))

    async def _broadcast(self, event: Dict[str, Any]) -> None:
        dead = []
        for client in list(self.clients):
            try:
                await client.send_json(event)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)

    # -- runs ------------------------------------------------------------
    def start_run(self, prompt: str, settings: Dict[str, Any]) -> Optional[str]:
        if self.running:
            return "Une tâche est déjà en cours."
        try:
            sandbox = self.sandbox
        except RuntimeError as exc:
            return str(exc)

        self.update_settings(settings)
        config = AgentRunConfig(
            max_steps=max(1, min(int(self.settings["max_steps"]), 100)),
            safety_mode=str(self.settings["safety_mode"]),
        )
        stream_frames = self.settings.get("display_mode") == "screenshots"

        def on_event(event_type: str, payload: Dict[str, Any]) -> None:
            self.publish(event_type, payload)

        def on_step(record) -> None:
            if stream_frames and record.screenshot:
                frame = _frame_from_png(record.screenshot)
                if frame:
                    self.publish("frame", {"image": frame, "step": record.step})

        runner = AgentRunner(
            sandbox=sandbox,
            vision=self.vision,
            config=config,
            on_event=on_event,
            on_step=on_step,
            confirm=self._wait_for_confirmation,
        )
        self.runner = runner
        conversation = self._conversation_summary()
        self.publish("user_message", {"text": prompt})

        def target() -> None:
            result = runner.run(prompt, conversation=conversation)
            self.conversation.append({"goal": prompt, "status": result.status, "reason": result.reason})
            self.conversation = self.conversation[-CONVERSATION_TURNS:]
            if stream_frames:
                try:
                    png, _ = sandbox.get_screenshot()
                    frame = _frame_from_png(png)
                    if frame:
                        self.publish("frame", {"image": frame})
                except Exception:
                    pass
            self.publish("status", self.status())

        self.thread = threading.Thread(target=target, name="visios-agent", daemon=True)
        self.thread.start()
        return None

    def update_settings(self, incoming: Dict[str, Any]) -> None:
        self.settings.update({k: v for k, v in incoming.items() if k in self.settings and v is not None})
        if self.settings.get("thinking") in {"full", "budget", "off"}:
            self.vision.thinking_mode = self.settings["thinking"]

    def _conversation_summary(self) -> str:
        lines = []
        for turn in self.conversation[-CONVERSATION_TURNS:]:
            lines.append(f"- User asked: {turn['goal']} → {turn['status']}: {turn['reason']}")
        return "\n".join(lines)

    def stop_run(self) -> None:
        if self.runner:
            self.runner.stop()
        self._confirm_result = False
        self._confirm_event.set()

    def add_note(self, text: str) -> None:
        if self.runner and self.running:
            self.runner.add_note(text)
            self.publish("user_note", {"text": text})

    def _wait_for_confirmation(self, action: AgentAction) -> bool:
        self._confirm_event.clear()
        self._confirm_result = False
        self._confirm_event.wait(timeout=AgentRunConfig().confirmation_timeout)
        return self._confirm_result

    def resolve_confirmation(self, approved: bool) -> None:
        self._confirm_result = bool(approved)
        self._confirm_event.set()

    def reset(self) -> None:
        if self.running:
            return
        with self.lock:
            self.history = []
        self.conversation = []
        self.publish("reset", {})

    def capture_frame(self) -> Optional[str]:
        png, _ = self.sandbox.get_screenshot()
        return _frame_from_png(png)


hub = Hub()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    hub.loop = asyncio.get_running_loop()
    yield
    hub.stop_run()


app = FastAPI(title="VisiOS", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/config")
async def get_config() -> JSONResponse:
    return JSONResponse(hub.config())


@app.get("/api/status")
async def get_status() -> JSONResponse:
    return JSONResponse(await asyncio.to_thread(hub.status))


@app.post("/api/sandbox/{action}")
async def sandbox_action(action: str) -> JSONResponse:
    if hub.running:
        return JSONResponse({"error": "Arrête la tâche en cours d'abord."}, status_code=409)
    try:
        if action == "start":
            await asyncio.to_thread(hub.sandbox.start_container)
        elif action == "stop":
            await asyncio.to_thread(hub.sandbox.stop_container)
        else:
            return JSONResponse({"error": "Action inconnue."}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    status = await asyncio.to_thread(hub.status)
    hub.publish("status", status)
    return JSONResponse(status)


@app.get("/api/frame")
async def get_frame() -> JSONResponse:
    try:
        frame = await asyncio.to_thread(hub.capture_frame)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    return JSONResponse({"image": frame})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    hub.clients.add(ws)
    try:
        with hub.lock:
            history = list(hub.history)
        await ws.send_json({"type": "hello", "config": hub.config(), "history": history, "running": hub.running})
        await ws.send_json({"type": "status", **(await asyncio.to_thread(hub.status))})
        while True:
            message = await ws.receive_json()
            kind = message.get("type")
            if kind == "run":
                prompt = str(message.get("prompt") or "").strip()
                if not prompt:
                    continue
                error = await asyncio.to_thread(hub.start_run, prompt, message.get("settings") or {})
                if error:
                    await ws.send_json({"type": "error", "message": error})
            elif kind == "stop":
                hub.stop_run()
            elif kind == "note":
                hub.add_note(str(message.get("text") or ""))
            elif kind == "confirm":
                hub.resolve_confirmation(bool(message.get("approved")))
            elif kind == "reset":
                hub.reset()
            elif kind == "settings":
                incoming = message.get("settings") or {}
                hub.update_settings(incoming)
                hub.publish("settings", {"settings": hub.settings})
            elif kind == "ping":
                await ws.send_json({"type": "pong", "ts": dt.datetime.now().isoformat()})
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.getenv("VISIOS_HOST", "127.0.0.1"),
        port=int(os.getenv("VISIOS_PORT", "8501")),
        log_level="info",
    )
