"""Vision agent that asks a multimodal model for the next desktop action."""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover - exercised only in minimal test envs
    requests = None  # type: ignore[assignment]

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None

LOGGER = logging.getLogger(__name__)

StreamCallback = Callable[[str, str], None]
CancelCheck = Callable[[], bool]


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):  # pragma: no cover
        return default


def _env_optional_int(key: str) -> Optional[int]:
    raw = os.getenv(key, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:  # pragma: no cover - defensive
        return None


DEFAULT_BASE_URL = os.getenv("VISIOS_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("VISIOS_OLLAMA_MODEL", "qwen2.5-vl")
DEFAULT_TEMPERATURE = _env_float("VISIOS_OLLAMA_TEMPERATURE", 0.1)
DEFAULT_TIMEOUT = _env_int("VISIOS_OLLAMA_TIMEOUT", 120)
DEFAULT_API_MODE = os.getenv("VISIOS_API_MODE", "ollama").lower()
DEFAULT_OPENAI_ENDPOINT = os.getenv("VISIOS_OPENAI_ENDPOINT", "responses").lower()
DEFAULT_API_KEY = os.getenv("VISIOS_API_KEY", "")
# "auto" picks the native grounding format of the model family:
# Qwen2.5-VL answers in absolute pixels of the image it received,
# Qwen3-VL / UI-TARS / most recent VLMs answer on a 0-1000 grid.
DEFAULT_COORD_MODE = os.getenv("VISIOS_COORD_MODE", "auto").lower()
DEFAULT_IMAGE_WIDTH = _env_int("VISIOS_MODEL_IMAGE_WIDTH", 1280)
DEFAULT_MAX_TOKENS = _env_optional_int("VISIOS_MAX_TOKENS")
DEFAULT_NUM_CTX = _env_int("VISIOS_OLLAMA_NUM_CTX", 16384)
# Thinking models: "full" lets them reason freely, "budget" cuts runaway reasoning
# (too long or looping), "off" asks the server to skip it.
DEFAULT_THINKING_MODE = os.getenv("VISIOS_THINKING", "budget").lower()
DEFAULT_THINKING_BUDGET = _env_int("VISIOS_THINKING_BUDGET", 6000)  # characters, ~1500 tokens
DEFAULT_MODEL_THINKS = os.getenv("VISIOS_MODEL_THINKS", "auto").lower()
# Qwen's guidance for thinking mode: near-greedy decoding makes the reasoning loop forever.
THINKING_TEMPERATURE = _env_float("VISIOS_THINKING_TEMPERATURE", 0.6)
THINKING_TOP_P = 0.95
THINKING_TOP_K = 20
THINKING_MODEL_HINTS = ("thinking", "qwen3.5", "qwq", "deepseek-r1", "magistral", "glm-4.1v", "kimi-vl")
PIXEL_GRID = 28  # Qwen2.5-VL patch * merge size: dimensions divisible by 28 are not resized again

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "un": 1,
    "une": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
    "petit": 1,  # fallback to avoid returning 0 when parsing phrases
}

COORD_RULES = {
    "norm1000": (
        "Use a normalized 0-1000 grid, whatever the image resolution: x=0 is the left edge and "
        "x=1000 the right edge, y=0 is the top edge and y=1000 the bottom edge."
    ),
    "pixel": (
        "Use absolute pixel coordinates of the screenshot you receive, which is __W__x__H__ pixels: "
        "x goes from 0 (left) to __W__, y from 0 (top) to __H__."
    ),
}

SYSTEM_PROMPT = """You are VisiOS, an autonomous agent operating a real Linux desktop (XFCE) by looking at screenshots and controlling the mouse and keyboard. You work step by step until the user's goal is fully achieved. The top panel holds the Applications menu, quick launchers (Firefox, Terminal, Files, Text Editor), the open windows and a clock. Firefox starts on DuckDuckGo.

Each turn you receive the user's goal, the history of your previous steps with their outcome, and a fresh screenshot. You answer with exactly ONE next action.

COORDINATES
__COORD_RULES__
Always aim at the visual center of the element you interact with. Never invent coordinates for something you cannot see on the screenshot.

ACTIONS
- click: {"action":"click","x":X,"y":Y}
- double_click: {"action":"double_click","x":X,"y":Y} (open desktop icons and files)
- right_click: {"action":"right_click","x":X,"y":Y}
- move: {"action":"move","x":X,"y":Y} (hover only)
- type: {"action":"type","text":"...","x":X,"y":Y,"submit":true} x/y are optional: when given, that field is clicked first to focus it. submit=true presses Enter after typing.
- key: {"action":"key","keys":"ctrl+l"} one shortcut or a space-separated sequence, e.g. "ctrl+a BackSpace", "Return", "Escape", "alt+F4", "ctrl+t", "Tab".
- scroll: {"action":"scroll","x":X,"y":Y,"direction":"down","amount":5} amount = 1-15 wheel notches; x/y over the area to scroll.
- drag: {"action":"drag","x":X,"y":Y,"to_x":X2,"to_y":Y2}
- open_app: {"action":"open_app","app":"browser|terminal|files|text_editor|settings|calculator"}
- open_url: {"action":"open_url","url":"https://..."} fastest way to visit a website (opens a browser tab).
- run_command: {"action":"run_command","command":"ls ~/Desktop"} runs bash in the sandbox; you will read its output next turn. Great for file and system tasks. Nothing appears on screen.
- wait: {"action":"wait","seconds":2} when something is loading.
- done: {"action":"done","summary":"..."} only when the screenshot proves the goal is achieved. The summary is shown to the user: write it in the user's language.
- fail: {"action":"fail","summary":"..."} when the goal is impossible or you need information from the user (credentials, a choice...).

RESPONSE FORMAT
Reply with ONLY one JSON object, no markdown fence, no text around it:
{"observation":"what matters on the screen right now","thought":"did the previous action work? what is the next step and why?","action":"<name>", ...fields of that action..., "requires_confirmation":false}

RULES
1. Look first: did the previous action have the expected effect? If not, do not repeat it blindly; try another way (keyboard shortcut, other target, scroll, wait).
2. Prefer the most reliable path: open_url for websites, keyboard shortcuts over tiny buttons, run_command for file operations the user does not need to watch.
3. To fill a text field, use type with the field's x/y so it gets focus, and "submit":true to validate a search or URL.
4. Close popups, cookie banners or first-run dialogs that are in the way.
5. Keep observation and thought short: one or two sentences each.
6. Set requires_confirmation=true before anything irreversible or sensitive: payments, sending messages or emails, deleting user data, entering passwords.
"""

THINKING_CUT_PROMPT = (
    "You have already analysed this screenshot at length. Stop deliberating now: decide with what you "
    "know and reply immediately with ONLY the JSON object for the next action. Your notes so far:"
)

JSON_RETRY_PROMPT = (
    "Your previous reply could not be parsed. Answer again with ONLY one valid JSON object "
    'following the RESPONSE FORMAT, for example {"observation":"...","thought":"...","action":"click","x":500,"y":300}.'
)


@dataclass
class AgentAction:
    """Structured decision returned by the vision model."""

    action_type: str
    x: Optional[float] = None
    y: Optional[float] = None
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    text: Optional[str] = None
    app: Optional[str] = None
    raw_response: str = ""
    reasoning: Optional[str] = None
    scroll_amount: Optional[float] = None
    duration: Optional[float] = None
    confidence: Optional[float] = None
    requires_confirmation: bool = False
    thought: Optional[str] = None
    observation: Optional[str] = None
    url: Optional[str] = None
    command: Optional[str] = None
    submit: bool = False
    # Coordinates exactly as the model emitted them (model space), kept for history/debug.
    model_coords: Optional[Tuple[float, float]] = None
    # True once x/y/end_x/end_y have been converted to absolute screen pixels.
    coords_absolute: bool = False
    parse_error: bool = False
    # Raw chain of thought of thinking models (reasoning_content / <think>), kept apart
    # from the short JSON "thought" so the UI can show it on its own.
    thinking: Optional[str] = None

    def is_click(self) -> bool:
        return self._normalized_type() in {"click", "double_click", "right_click"}

    def is_type(self) -> bool:
        return self._normalized_type() == "type"

    def _normalized_type(self) -> str:
        raw = (self.action_type or "").strip().lower()
        if not raw:
            return ""
        normalized = raw.replace("-", "_").replace(" ", "_")
        if normalized in {"tap", "left_click", "leftclick", "mouse_click", "click_element"}:
            return "click"
        if normalized in {"double_click", "doubleclick", "dblclick", "double_tap"}:
            return "double_click"
        if normalized in {"right_click", "rightclick", "context_click", "context_menu"}:
            return "right_click"
        if normalized in {"move", "hover", "mouse_move", "move_mouse", "mousemove"}:
            return "move"
        if normalized in {"key", "keys", "press", "press_key", "shortcut", "keyboard", "keypress", "hotkey", "key_press"}:
            return "hotkey"
        if normalized in {"type_text", "write", "input", "input_text", "fill"}:
            return "type"
        if normalized.startswith("scroll") or normalized in {"wheel", "mousewheel"}:
            return "scroll"
        if normalized in {"swipe", "move_drag", "mouse_drag", "drag_and_drop", "left_click_drag"}:
            return "drag"
        if normalized in {"open_url", "navigate", "goto", "go_to", "go_to_url", "visit", "browse", "open_website", "open_link"}:
            return "open_url"
        if normalized in {"run_command", "command", "shell", "bash", "execute", "run", "exec", "terminal_command"}:
            return "run_command"
        if normalized in {"open", "launch", "launch_app", "open_application"} or normalized.startswith("open_"):
            return "open_app"
        if normalized in {"finish", "finished", "complete", "completed", "success", "stop", "terminate"}:
            return "done"
        if normalized in {"fail", "failed", "give_up", "ask_user", "need_help", "impossible", "error", "call_user"}:
            return "fail"
        if normalized in {"observe", "sleep", "pause"}:
            return "wait"
        return normalized

    def pointer(self) -> Optional[Tuple[float, float]]:
        if self.x is None or self.y is None:
            return None
        return self.x, self.y

    def has_drag_target(self) -> bool:
        return (
            self.x is not None
            and self.y is not None
            and self.end_x is not None
            and self.end_y is not None
        )

    def summary(self) -> str:
        """Short, human-readable label of the action (used in history and UI)."""

        kind = self._normalized_type()
        if kind in {"click", "double_click", "right_click", "move"} and self.model_coords:
            return f"{kind} ({self.model_coords[0]:.0f}, {self.model_coords[1]:.0f})"
        if kind == "type":
            where = f" @ ({self.model_coords[0]:.0f}, {self.model_coords[1]:.0f})" if self.model_coords else ""
            return f'type "{_clip(self.text or "", 80)}"{where}{" + Enter" if self.submit else ""}'
        if kind == "hotkey":
            return f"key {self.text or ''}"
        if kind == "scroll":
            direction = "up" if (self.scroll_amount or 0) < 0 else "down"
            return f"scroll {direction} {abs(self.scroll_amount or 0):.0f}"
        if kind == "drag":
            return "drag"
        if kind == "open_app":
            return f"open_app {self.app or self.text or ''}"
        if kind == "open_url":
            return f"open_url {self.url or self.text or ''}"
        if kind == "run_command":
            return f"run_command `{_clip(self.command or self.text or '', 120)}`"
        if kind == "wait":
            return f"wait {self.duration or 1:.0f}s"
        if kind in {"done", "fail"}:
            return f"{kind}: {_clip(self.text or '', 120)}"
        return kind or "noop"


def _clip(value: str, limit: int) -> str:
    value = " ".join(str(value).split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


class ThinkingGuard:
    """Detects runaway reasoning: over budget, or stuck repeating the same passage."""

    def __init__(self, budget: int) -> None:
        self.budget = max(0, int(budget or 0))
        self.text = ""
        self.reason = ""
        self._checked = 0

    def feed(self, delta: str) -> bool:
        self.text += delta
        if self.budget and len(self.text) > self.budget:
            self.reason = "budget"
            return True
        if len(self.text) - self._checked >= 300 and len(self.text) > 1200:
            self._checked = len(self.text)
            tail = self.text[-160:]
            if self.text.find(tail) < len(self.text) - len(tail):
                self.reason = "loop"
                return True
        return False


@dataclass
class PreparedImage:
    """Screenshot as sent to the model, plus what is needed to map coordinates back."""

    b64: str
    mime: str
    model_size: Tuple[int, int]
    screen_size: Tuple[int, int]


@dataclass
class StepContext:
    """Everything the model needs besides the screenshot."""

    goal: str
    step: int = 1
    max_steps: int = 0
    history: List[str] = field(default_factory=list)
    last_result: str = ""
    desktop_context: str = ""
    user_notes: List[str] = field(default_factory=list)
    conversation: str = ""


class VisionAgent:
    """Wrapper around Ollama / OpenAI-compatible APIs for multimodal decisions."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
        api_mode: str = DEFAULT_API_MODE,
        openai_endpoint: str = DEFAULT_OPENAI_ENDPOINT,
        coord_mode: str = DEFAULT_COORD_MODE,
        image_width: int = DEFAULT_IMAGE_WIDTH,
        max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
        api_key: str = DEFAULT_API_KEY,
        thinking_mode: str = DEFAULT_THINKING_MODE,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.api_mode = api_mode.lower()
        self.openai_endpoint = openai_endpoint.lower()
        self.coord_mode = self._resolve_coord_mode(coord_mode, model)
        self.image_width = max(320, int(image_width or 1280))
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.thinking_mode = thinking_mode if thinking_mode in {"full", "budget", "off"} else "budget"
        self.thinking_budget = thinking_budget
        # Learned from the first answers when the model name does not tell.
        if DEFAULT_MODEL_THINKS in {"true", "1", "yes"}:
            self.model_thinks = True
        elif DEFAULT_MODEL_THINKS in {"false", "0", "no"}:
            self.model_thinks = False
        else:
            self.model_thinks = any(hint in (model or "").lower() for hint in THINKING_MODEL_HINTS)
        self.last_thinking_cut = ""

    @staticmethod
    def _resolve_coord_mode(mode: str, model: str) -> str:
        cleaned = (mode or "auto").strip().lower()
        if cleaned in {"pixel", "pixels", "absolute"}:
            return "pixel"
        if cleaned in {"norm1000", "normalized", "relative", "1000"}:
            return "norm1000"
        name = (model or "").lower().replace("_", "-")
        if "qwen2.5-vl" in name or "qwen2.5vl" in name or "qwen2-5-vl" in name:
            return "pixel"
        return "norm1000"

    # ------------------------------------------------------------------
    def decide_action(
        self,
        screenshot: bytes,
        user_prompt: str,
        screen_size: Optional[Tuple[int, int]] = None,
        desktop_context: Optional[str] = None,
        recent_actions: Optional[str] = None,
        context: Optional[StepContext] = None,
        on_stream: Optional[StreamCallback] = None,
        should_cancel: Optional[CancelCheck] = None,
    ) -> AgentAction:
        """Send the screenshot + context to the backend and return a screen-space action."""

        image = self.prepare_image(screenshot, screen_size)
        if context is None:
            context = StepContext(
                goal=user_prompt,
                desktop_context=desktop_context or "",
                history=[recent_actions] if recent_actions else [],
            )
        system_prompt = self.build_system_prompt(image.model_size)
        user_text = self.build_user_message(context)
        LOGGER.info(
            "Querying %s (%s) with a %sx%s image, coord_mode=%s",
            self.api_mode,
            self.model,
            image.model_size[0],
            image.model_size[1],
            self.coord_mode,
        )

        disable_thinking = self.thinking_mode == "off"
        full_text, backend_reasoning = self._call_backend(
            system_prompt, user_text, image, on_stream, should_cancel, disable_thinking
        )

        if self.last_thinking_cut and not (should_cancel and should_cancel()):
            # Runaway reasoning: keep its tail as notes and ask for a direct answer.
            LOGGER.info("Thinking cut (%s) after %s chars", self.last_thinking_cut, len(backend_reasoning or full_text))
            if on_stream:
                on_stream("status", f"thinking_cut:{self.last_thinking_cut}")
            notes = backend_reasoning or full_text.split("<think>", 1)[-1]
            follow_up = f"{user_text}\n\n{THINKING_CUT_PROMPT}\n{notes[-1500:]}"
            full_text, extra_reasoning = self._call_backend(
                system_prompt, follow_up, image, on_stream, should_cancel, True
            )
            backend_reasoning = self._merge_reasoning_segments(backend_reasoning, extra_reasoning)

        action = self._interpret(full_text, backend_reasoning)

        if action.parse_error and not (should_cancel and should_cancel()):
            LOGGER.info("Unparseable model answer, retrying once: %s", full_text[:300])
            if on_stream:
                on_stream("status", "retry")
            retry_text = f"{user_text}\n\n{JSON_RETRY_PROMPT}"
            full_text, retry_reasoning = self._call_backend(
                system_prompt, retry_text, image, on_stream, should_cancel, True
            )
            action = self._interpret(full_text, self._merge_reasoning_segments(backend_reasoning, retry_reasoning))

        self.to_screen_space(action, image)
        return action

    def _interpret(self, full_text: str, backend_reasoning: Optional[str]) -> AgentAction:
        cleaned_text, inline_reasoning = self._strip_reasoning_sections(full_text)
        LOGGER.debug("Vision backend response: %s", cleaned_text)
        action = self._parse_action(cleaned_text)
        action.reasoning = self._merge_reasoning_segments(action.reasoning, backend_reasoning, inline_reasoning)
        action.thinking = self._merge_reasoning_segments(backend_reasoning, inline_reasoning)
        return action

    def _call_backend(
        self,
        system_prompt: str,
        user_text: str,
        image: PreparedImage,
        on_stream: Optional[StreamCallback],
        should_cancel: Optional[CancelCheck],
        disable_thinking: bool = False,
    ) -> Tuple[str, Optional[str]]:
        self.last_thinking_cut = ""
        if self.api_mode == "openai":
            return self._call_openai_api(image, system_prompt, user_text, on_stream, should_cancel, disable_thinking)
        return self._call_ollama_chat(image, system_prompt, user_text, on_stream, should_cancel, disable_thinking)

    def _sampling(self, disable_thinking: bool) -> Dict[str, Any]:
        if self.model_thinks and not disable_thinking:
            return {
                "temperature": max(self.temperature, THINKING_TEMPERATURE),
                "top_p": THINKING_TOP_P,
                "top_k": THINKING_TOP_K,
            }
        return {"temperature": self.temperature}

    def _guard(self, disable_thinking: bool) -> ThinkingGuard:
        if self.thinking_mode == "full" and not disable_thinking:
            return ThinkingGuard(0)  # loop detection only
        budget = self.thinking_budget if not disable_thinking else min(self.thinking_budget or 2000, 2000)
        return ThinkingGuard(budget)

    def _no_think_suffix(self, disable_thinking: bool) -> str:
        # Qwen3 soft switch; harmless text for other models.
        if disable_thinking and self.model_thinks and "qwen" in (self.model or "").lower():
            return "\n/no_think"
        return ""

    # ------------------------------------------------------------------
    # Image & coordinate handling
    # ------------------------------------------------------------------
    def prepare_image(self, screenshot: bytes, screen_size: Optional[Tuple[int, int]] = None) -> PreparedImage:
        """Resize the screenshot to a size we control so coordinates map back exactly."""

        if Image is None:
            size = screen_size or (0, 0)
            return PreparedImage(base64.b64encode(screenshot).decode("ascii"), "image/png", size, size)

        with Image.open(io.BytesIO(screenshot)) as img:
            img = img.convert("RGB")
            screen_w, screen_h = img.size
            target_w, target_h = self._target_size(screen_w, screen_h)
            if (target_w, target_h) != (screen_w, screen_h):
                img = img.resize((target_w, target_h), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG", optimize=False, compress_level=3)
        return PreparedImage(
            b64=base64.b64encode(buffer.getvalue()).decode("ascii"),
            mime="image/png",
            model_size=(target_w, target_h),
            screen_size=(screen_w, screen_h),
        )

    def _target_size(self, width: int, height: int) -> Tuple[int, int]:
        if width <= 0 or height <= 0:
            return width, height
        scale = min(1.0, self.image_width / float(width))
        target_w = width * scale
        target_h = height * scale
        if self.coord_mode == "pixel":
            # Snap to the model's patch grid so the server does not resize again
            # (which would silently shift every pixel coordinate).
            target_w = max(PIXEL_GRID, round(target_w / PIXEL_GRID) * PIXEL_GRID)
            target_h = max(PIXEL_GRID, round(target_h / PIXEL_GRID) * PIXEL_GRID)
        return int(round(target_w)), int(round(target_h))

    def to_screen_space(self, action: AgentAction, image: PreparedImage) -> AgentAction:
        """Convert model-space coordinates to absolute screen pixels (in place)."""

        if action.x is not None and action.y is not None:
            action.model_coords = (action.x, action.y)
        screen_w, screen_h = image.screen_size
        if screen_w <= 0 or screen_h <= 0:
            return action

        def convert(x: Optional[float], y: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
            if x is None or y is None:
                return x, y
            return self._model_to_screen(x, y, image)

        action.x, action.y = convert(action.x, action.y)
        action.end_x, action.end_y = convert(action.end_x, action.end_y)
        action.coords_absolute = True
        return action

    def _model_to_screen(self, x: float, y: float, image: PreparedImage) -> Tuple[float, float]:
        screen_w, screen_h = image.screen_size
        model_w, model_h = image.model_size
        is_fraction = 0 <= x <= 1 and 0 <= y <= 1 and (x % 1 or y % 1)
        if is_fraction:
            sx, sy = x * screen_w, y * screen_h
        elif self.coord_mode == "norm1000" and x <= 1000 and y <= 1000:
            sx, sy = x / 1000.0 * screen_w, y / 1000.0 * screen_h
        else:
            # Pixel mode, or a norm1000 model that answered in pixels anyway.
            sx = x * screen_w / float(model_w or screen_w)
            sy = y * screen_h / float(model_h or screen_h)
        sx = max(0.0, min(screen_w - 1.0, sx))
        sy = max(0.0, min(screen_h - 1.0, sy))
        return sx, sy

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------
    def build_system_prompt(self, model_size: Tuple[int, int]) -> str:
        rules = COORD_RULES[self.coord_mode]
        width, height = model_size
        rules = rules.replace("__W__", str(max(width, 1))).replace("__H__", str(max(height, 1)))
        return SYSTEM_PROMPT.replace("__COORD_RULES__", rules)

    def build_user_message(self, context: StepContext) -> str:
        lines: List[str] = [f"GOAL: {context.goal}"]
        if context.conversation:
            lines.append(f"EARLIER IN THIS CONVERSATION:\n{context.conversation}")
        if context.max_steps:
            lines.append(f"STEP {context.step} of at most {context.max_steps}.")
        if context.history:
            lines.append("PREVIOUS STEPS:\n" + "\n".join(context.history[-12:]))
        else:
            lines.append("PREVIOUS STEPS: none, this is the first step.")
        if context.last_result:
            lines.append(f"RESULT OF YOUR LAST ACTION:\n{context.last_result}")
        for note in context.user_notes:
            lines.append(f"NEW MESSAGE FROM THE USER (takes priority): {note}")
        if context.desktop_context:
            lines.append(f"DESKTOP STATE:\n{context.desktop_context[:1500]}")
        lines.append("Here is the current screenshot. Reply with the JSON for the next action only.")
        return "\n\n".join(lines)

    # Backwards compatible helper kept for callers/tests of the previous API.
    def _build_prompt(
        self,
        user_prompt: str,
        screen_size: Optional[Tuple[int, int]],
        desktop_context: Optional[str] = None,
        recent_actions: Optional[str] = None,
    ) -> str:
        context = StepContext(
            goal=user_prompt,
            desktop_context=desktop_context or "",
            history=[recent_actions] if recent_actions else [],
        )
        size = screen_size or (1000, 1000)
        return f"{self.build_system_prompt(self._target_size(*size))}\n{self.build_user_message(context)}"

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_action(self, model_text: str) -> AgentAction:
        """Best-effort extraction of a structured action from model output."""

        action = AgentAction(action_type="noop", raw_response=model_text)
        reasoning_parts: List[str] = []

        json_blob = self._extract_json(model_text)
        if json_blob is not None:
            reasoning_parts.extend(self._extract_reasoning_fields(json_blob))
            action = self._from_json(json_blob, model_text)
        else:
            match = re.search(r"click.*?(\d+)\D+(\d+)", model_text, re.IGNORECASE | re.DOTALL)
            if match:
                action = AgentAction(action_type="click", x=float(match.group(1)), y=float(match.group(2)), raw_response=model_text)
            elif "hotkey" in model_text.lower() or "shortcut" in model_text.lower():
                text_match = re.search(r"(?:hotkey|shortcut)[:\s]+(.+)$", model_text, re.IGNORECASE)
                action = AgentAction(action_type="hotkey", text=text_match.group(1).strip() if text_match else "", raw_response=model_text)
            elif re.search(r"\btype\b", model_text, re.IGNORECASE):
                text_match = re.search(r"type[:\s]+(.+)$", model_text, re.IGNORECASE)
                action = AgentAction(action_type="type", text=text_match.group(1).strip() if text_match else "", raw_response=model_text)
            elif "scroll" in model_text.lower():
                amount = self._extract_numeric_from_text(model_text)
                action = AgentAction(action_type="scroll", text=model_text, raw_response=model_text, scroll_amount=amount)
            else:
                action.parse_error = True
        if reasoning_parts:
            action.reasoning = self._merge_reasoning_segments(action.reasoning, *reasoning_parts)
        return action

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        stripped = VisionAgent._sanitize_json_text(text)
        if stripped:
            parsed = VisionAgent._loads_object(stripped)
            if parsed is not None:
                return parsed

        # Scan for balanced {...} blocks and keep the last one that parses
        # (models sometimes echo an example before the real answer).
        found: Optional[Dict[str, Any]] = None
        for candidate in VisionAgent._balanced_objects(text):
            parsed = VisionAgent._loads_object(candidate)
            if parsed is not None and (found is None or "action" in parsed or "type" in parsed):
                found = parsed
        return found

    @staticmethod
    def _loads_object(text: str) -> Optional[Dict[str, Any]]:
        attempts = [text, VisionAgent._loosen_braces(text)]
        # Trailing commas are a frequent small-model mistake.
        attempts.append(re.sub(r",\s*([}\]])", r"\1", attempts[-1]))
        for attempt in attempts:
            try:
                value = json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(value, dict):
                return value
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value[0]
        return None

    @staticmethod
    def _balanced_objects(text: str) -> Iterable[str]:
        depth = 0
        start = -1
        in_string = False
        escaped = False
        for index, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : index + 1]
                    start = -1

    @staticmethod
    def _sanitize_json_text(text: str) -> str:
        stripped = text.strip()

        fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
        if fence:
            stripped = fence.group(1).strip()
        elif stripped.startswith("```"):
            stripped = stripped.strip("`").strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()

        while stripped.startswith("{{") and stripped.endswith("}}"):  # handle double braces
            stripped = stripped[1:-1].strip()

        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        return ""

    @staticmethod
    def _loosen_braces(text: str) -> str:
        if "{{" not in text and "}}" not in text:
            return text
        return text.replace("{{", "{").replace("}}", "}")

    @staticmethod
    def _first(payload: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = payload.get(key)
            if value is not None and value != "":
                return value
        return None

    @staticmethod
    def _xy(value: Any) -> Tuple[Optional[float], Optional[float]]:
        if isinstance(value, dict):
            return (
                VisionAgent._to_number(VisionAgent._first(value, "x", "X")),
                VisionAgent._to_number(VisionAgent._first(value, "y", "Y")),
            )
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            if len(value) >= 4 and all(isinstance(v, (int, float)) for v in value[:4]):
                # Bounding box [x1, y1, x2, y2]: aim at its center.
                return (float(value[0]) + float(value[2])) / 2, (float(value[1]) + float(value[3])) / 2
            return VisionAgent._to_number(value[0]), VisionAgent._to_number(value[1])
        if isinstance(value, str):
            numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
            if len(numbers) >= 2:
                return float(numbers[0]), float(numbers[1])
        return None, None

    @staticmethod
    def _from_json(payload: Dict[str, Any], raw_text: str) -> AgentAction:
        action = payload.get("action") or payload.get("type") or payload.get("name") or "noop"
        if isinstance(action, dict):
            nested = action
            action = nested.get("action") or nested.get("type") or nested.get("name") or "noop"
            payload = {**payload, **nested}
        params = payload.get("parameters") or payload.get("params") or payload.get("args")
        if isinstance(params, dict):
            payload = {**payload, **params}

        coords = VisionAgent._first(
            payload, "coordinates", "coordinate", "coords", "point", "position", "start", "bbox", "box"
        )
        end_coords = VisionAgent._first(
            payload, "to", "end", "target_coordinates", "end_coordinates", "end_coordinate", "destination"
        )
        x, y = VisionAgent._xy(coords) if coords is not None else (None, None)
        if x is None or y is None:
            x = VisionAgent._to_number(payload.get("x"))
            y = VisionAgent._to_number(payload.get("y"))
        end_x, end_y = VisionAgent._xy(end_coords) if end_coords is not None else (None, None)
        if end_x is None or end_y is None:
            end_x = VisionAgent._to_number(VisionAgent._first(payload, "to_x", "end_x", "x2"))
            end_y = VisionAgent._to_number(VisionAgent._first(payload, "to_y", "end_y", "y2"))

        action_text = str(action)
        normalized = AgentAction(action_type=action_text)._normalized_type()

        text = VisionAgent._first(payload, "text", "value", "keys", "key", "hotkey", "query", "content")
        if normalized in {"done", "fail"}:
            text = VisionAgent._first(payload, "summary", "message", "result", "reason", "text", "answer") or text
        url = VisionAgent._first(payload, "url", "link", "href")
        command = VisionAgent._first(payload, "command", "cmd", "script")
        if normalized == "open_url" and not url and text:
            url = text
        if normalized == "run_command" and not command and text:
            command = text
        if isinstance(text, (list, tuple)):
            text = " ".join(str(part) for part in text)

        app = payload.get("app") or payload.get("application") or payload.get("program")
        if normalized == "open_app" and action_text.lower().replace("-", "_").startswith("open_") and not app:
            app = action_text.split("_", 1)[1]

        scroll_amount = VisionAgent._extract_scroll_amount(payload, text)
        direction = str(payload.get("direction") or "").lower()
        if normalized == "scroll" and scroll_amount is not None:
            if direction in {"up", "haut", "left"} or "up" in action_text.lower():
                scroll_amount = -abs(scroll_amount)
            elif direction in {"down", "bas", "right"} or "down" in action_text.lower():
                scroll_amount = abs(scroll_amount)
        elif normalized == "scroll" and direction:
            text = f"{text or ''} {direction}".strip()

        observation = VisionAgent._first(payload, "observation", "screen", "state")
        thought = VisionAgent._first(payload, "thought", "thinking", "reasoning", "rationale", "plan")
        duration = VisionAgent._to_number(VisionAgent._first(payload, "duration", "seconds", "time"))

        return AgentAction(
            action_type=action_text,
            x=x,
            y=y,
            end_x=end_x,
            end_y=end_y,
            text=str(text) if text is not None else None,
            app=str(app) if app else None,
            raw_response=raw_text,
            scroll_amount=scroll_amount,
            duration=duration,
            confidence=VisionAgent._to_number(payload.get("confidence")),
            requires_confirmation=VisionAgent._to_bool(
                payload.get("requires_confirmation")
                or payload.get("needs_confirmation")
                or payload.get("confirm")
            ),
            thought=str(thought).strip() if isinstance(thought, str) else None,
            observation=str(observation).strip() if isinstance(observation, str) else None,
            url=str(url).strip() if url else None,
            command=str(command) if command else None,
            submit=VisionAgent._to_bool(payload.get("submit") or payload.get("enter") or payload.get("press_enter")),
        )

    @staticmethod
    def _extract_scroll_amount(payload: Dict[str, Any], text: Optional[str]) -> Optional[float]:
        candidates = []
        for key in ("scroll", "delta", "amount", "value", "distance", "lines", "magnitude"):
            if key in payload:
                candidates.append(payload[key])
        coords = payload.get("coordinates") or payload.get("coords")
        if isinstance(coords, dict):
            for key in ("delta", "amount"):
                if key in coords:
                    candidates.append(coords[key])
        meta = payload.get("meta") or payload.get("metadata")
        if isinstance(meta, dict):
            for key in ("scroll", "delta", "amount"):
                if key in meta:
                    candidates.append(meta[key])
        for candidate in candidates:
            number = VisionAgent._to_number(candidate)
            if number is not None:
                return number
        return VisionAgent._extract_numeric_from_text(text)

    @staticmethod
    def _extract_numeric_from_text(text: Optional[str]) -> Optional[float]:
        if not text:
            return None
        if isinstance(text, (int, float)):
            return float(text)
        if not isinstance(text, str):
            text = str(text)
        match = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if not match:
            return VisionAgent._word_to_number(text)
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _word_to_number(text: str) -> Optional[float]:
        lowered = text.lower()
        tokens = re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ]+", lowered)
        for token in tokens:
            normalized = VisionAgent._strip_accents(token)
            if normalized in NUMBER_WORDS:
                return float(NUMBER_WORDS[normalized])
        return None

    @staticmethod
    def _strip_accents(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    @staticmethod
    def _to_number(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {"1", "true", "yes", "on", "oui"}

    # ------------------------------------------------------------------
    # Backend adapters
    # ------------------------------------------------------------------
    def _endpoint(self, path: str) -> str:
        cleaned_base = self.base_url.rstrip("/")
        cleaned_path = path.strip("/")
        if cleaned_base.endswith("/v1") and cleaned_path.startswith("v1/"):
            cleaned_path = cleaned_path[3:]
        return f"{cleaned_base}/{cleaned_path}" if cleaned_path else cleaned_base

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _call_ollama_chat(
        self,
        image: PreparedImage,
        system_prompt: str,
        user_text: str,
        on_stream: Optional[StreamCallback] = None,
        should_cancel: Optional[CancelCheck] = None,
        disable_thinking: bool = False,
    ) -> Tuple[str, Optional[str]]:
        self._ensure_requests()
        options: Dict[str, Any] = {**self._sampling(disable_thinking), "num_ctx": DEFAULT_NUM_CTX}
        if self.max_tokens:
            options["num_predict"] = self.max_tokens
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text + self._no_think_suffix(disable_thinking), "images": [image.b64]},
            ],
            "stream": True,
            "options": options,
        }
        if disable_thinking and self.model_thinks:
            payload["think"] = False
        url = self._endpoint("api/chat")
        response = requests.post(url, json=payload, stream=True, timeout=self.timeout)
        response.raise_for_status()

        content: List[str] = []
        thinking: List[str] = []
        guard = self._guard(disable_thinking)
        try:
            for line in response.iter_lines():
                if should_cancel and should_cancel():
                    break
                if not line:
                    continue
                chunk = json.loads(line)
                message = chunk.get("message") or {}
                think_delta = message.get("thinking") or ""
                text_delta = message.get("content") or chunk.get("response") or ""
                if self._consume_deltas(think_delta, text_delta, content, thinking, guard, on_stream):
                    break
                if chunk.get("done"):
                    break
        finally:
            response.close()
        return "".join(content).strip(), ("".join(thinking).strip() or None)

    # Legacy single-prompt endpoint, kept for callers of the previous API.
    def _call_ollama_api(self, encoded_image: str, prompt: str) -> Tuple[str, Optional[str]]:
        self._ensure_requests()
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "images": [encoded_image],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        response = requests.post(self._endpoint("api/generate"), json=payload, timeout=self.timeout)
        response.raise_for_status()
        return str(response.json().get("response", "")).strip(), None

    def _call_openai_api(
        self,
        image: PreparedImage,
        system_prompt: str,
        user_text: str,
        on_stream: Optional[StreamCallback] = None,
        should_cancel: Optional[CancelCheck] = None,
        disable_thinking: bool = False,
    ) -> Tuple[str, Optional[str]]:
        if self.openai_endpoint in {"chat", "chat_completions", "v1/chat/completions"}:
            return self._call_openai_chat(image, system_prompt, user_text, on_stream, should_cancel, disable_thinking)

        if self.openai_endpoint in {"completions", "completion", "v1/completions"}:
            return self._call_openai_completion(image, system_prompt, user_text)

        return self._call_openai_responses(image, system_prompt, user_text)

    def _call_openai_responses(
        self,
        image: PreparedImage,
        system_prompt: str,
        user_text: str,
    ) -> Tuple[str, Optional[str]]:
        self._ensure_requests()
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {"type": "input_image", "image_url": f"data:{image.mime};base64,{image.b64}"},
                    ],
                },
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["max_output_tokens"] = self.max_tokens
        response = requests.post(
            self._endpoint("v1/responses"), json=payload, headers=self._headers(), timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        text, reasoning = self._extract_openai_text(data)
        if text:
            return text.strip(), reasoning
        LOGGER.warning("Unable to parse OpenAI responses payload: %s", json.dumps(data)[:500])
        return "", reasoning

    def _call_openai_chat(
        self,
        image: PreparedImage,
        system_prompt: str,
        user_text: str,
        on_stream: Optional[StreamCallback] = None,
        should_cancel: Optional[CancelCheck] = None,
        disable_thinking: bool = False,
    ) -> Tuple[str, Optional[str]]:
        self._ensure_requests()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text + self._no_think_suffix(disable_thinking)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image.mime};base64,{image.b64}"},
                        },
                    ],
                },
            ],
            "stream": True,
            **self._sampling(disable_thinking),
        }
        if disable_thinking and self.model_thinks:
            # Honoured by llama.cpp / vLLM style servers, ignored elsewhere.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        response = requests.post(
            self._endpoint("v1/chat/completions"),
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            # Server ignored stream=true: parse the regular JSON body.
            data = response.json()
            text, reasoning = self._extract_openai_text(data)
            if on_stream and text:
                on_stream("text", text)
            return text.strip(), reasoning

        content: List[str] = []
        thinking: List[str] = []
        guard = self._guard(disable_thinking)
        stop = False
        try:
            for raw_line in response.iter_lines(decode_unicode=False):
                if stop:
                    break
                if should_cancel and should_cancel():
                    break
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    think_delta = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if isinstance(think_delta, dict):
                        think_delta = think_delta.get("content") or think_delta.get("text") or ""
                    text_delta = delta.get("content") or ""
                    if self._consume_deltas(str(think_delta), str(text_delta), content, thinking, guard, on_stream):
                        stop = True
                        break
        finally:
            response.close()
        return "".join(content).strip(), ("".join(thinking).strip() or None)

    def _consume_deltas(
        self,
        think_delta: str,
        text_delta: str,
        content: List[str],
        thinking: List[str],
        guard: ThinkingGuard,
        on_stream: Optional[StreamCallback],
    ) -> bool:
        """Record one streamed chunk; return True when the reasoning must be cut."""

        cut = False
        if think_delta:
            self.model_thinks = True
            thinking.append(think_delta)
            if on_stream:
                on_stream("thinking", think_delta)
            cut = guard.feed(think_delta)
        if text_delta:
            content.append(text_delta)
            if on_stream:
                on_stream("text", text_delta)
            so_far = "".join(content)
            if "<think>" in so_far:
                self.model_thinks = True
                if "</think>" not in so_far:
                    cut = guard.feed(text_delta) or cut
        if cut:
            self.last_thinking_cut = guard.reason
        return cut

    def _call_openai_completion(
        self,
        image: PreparedImage,
        system_prompt: str,
        user_text: str,
    ) -> Tuple[str, Optional[str]]:
        self._ensure_requests()
        composed_prompt = (
            f"{system_prompt}\n{user_text}\n"
            "Image (base64) follows between <image></image> tags.\n"
            f"<image>{image.b64}</image>"
        )
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": composed_prompt,
            "temperature": self.temperature,
        }
        response = requests.post(
            self._endpoint("v1/completions"), json=payload, headers=self._headers(), timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        text, reasoning = self._extract_openai_text(data)
        if text:
            return text.strip(), reasoning
        LOGGER.warning("Unable to parse OpenAI completions payload: %s", json.dumps(data)[:500])
        return "", reasoning

    def check_backend(self) -> Tuple[bool, str]:
        """Quick reachability probe used by the UI health indicator."""

        self._ensure_requests()
        path = "v1/models" if self.api_mode == "openai" else "api/tags"
        try:
            response = requests.get(self._endpoint(path), headers=self._headers(), timeout=3)
            response.raise_for_status()
            return True, "ok"
        except Exception as exc:
            return False, str(exc)[:200]

    @staticmethod
    def _ensure_requests() -> None:
        if requests is None:
            raise RuntimeError(
                "The `requests` package is not installed. Run `pip install -r requirements.txt` "
                "before calling the vision backend."
            )

    @staticmethod
    def _extract_openai_text(payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        reasoning_segments: List[str] = []

        def _iter_text(blobs: Iterable[Dict[str, Any]]) -> Iterable[str]:
            for blob in blobs:
                if not isinstance(blob, dict):
                    continue
                if blob.get("type") == "reasoning":
                    normalized = VisionAgent._normalize_reasoning_value(
                        blob.get("summary") or blob.get("content") or blob.get("reasoning")
                    )
                    if normalized:
                        reasoning_segments.append(normalized)
                    continue
                if "content" in blob and isinstance(blob["content"], list):
                    for content in blob["content"]:
                        if not isinstance(content, dict):
                            continue
                        segment_type = content.get("type")
                        if segment_type in {"reasoning", "reasoning_text"}:
                            normalized = VisionAgent._normalize_reasoning_value(
                                content.get("reasoning") or content.get("text")
                            )
                            if normalized:
                                reasoning_segments.append(normalized)
                            continue
                        text_value = content.get("text") or content.get("value")
                        if text_value:
                            yield str(text_value)
                else:
                    text_value = blob.get("text") or blob.get("value")
                    if text_value:
                        yield str(text_value)

        if "output" in payload and isinstance(payload["output"], list):
            text_value = "".join(_iter_text(payload["output"]))
            return text_value, VisionAgent._merge_reasoning_segments(*reasoning_segments)

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            message = choice.get("message", {}) or {}
            message_reasoning = message.get("reasoning_content") or message.get("reasoning")
            normalized_reasoning = VisionAgent._normalize_reasoning_value(message_reasoning)
            if normalized_reasoning:
                reasoning_segments.append(normalized_reasoning)
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for segment in content:
                    if not isinstance(segment, dict):
                        continue
                    segment_type = segment.get("type")
                    if segment_type == "reasoning":
                        normalized = VisionAgent._normalize_reasoning_value(segment.get("reasoning"))
                        if normalized:
                            reasoning_segments.append(normalized)
                        continue
                    if "text" in segment:
                        parts.append(str(segment["text"]))
                    elif segment_type in {"output_text", "text"} and "value" in segment:
                        parts.append(str(segment["value"]))
                if parts:
                    return "".join(parts), VisionAgent._merge_reasoning_segments(*reasoning_segments)
            if isinstance(content, str):
                return content, VisionAgent._merge_reasoning_segments(*reasoning_segments)
            if "text" in choice:
                return str(choice["text"]), VisionAgent._merge_reasoning_segments(*reasoning_segments)

        if "output_text" in payload:
            return str(payload["output_text"]), VisionAgent._merge_reasoning_segments(*reasoning_segments)

        return "", VisionAgent._merge_reasoning_segments(*reasoning_segments)

    @staticmethod
    def _strip_reasoning_sections(text: str) -> Tuple[str, Optional[str]]:
        if not text:
            return text, None

        reasoning_parts: List[str] = []
        cleaned = text

        def _store_snippet(snippet: str) -> None:
            snippet = snippet.strip()
            if snippet:
                reasoning_parts.append(snippet)

        for tag in ("think", "thinking", "reasoning", "thought", "thoughts"):
            pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)

            def _replace(match: re.Match[str]) -> str:
                _store_snippet(match.group(1))
                return ""

            cleaned = pattern.sub(_replace, cleaned)

        block_pattern = re.compile(
            r"(?is)(?:^|\n)\s*(?:reasoning|analysis|thoughts?)\s*:\s*(.+?)(?=\n\s*(?:action|result|answer)\s*:|\Z)"
        )

        cleaned = block_pattern.sub(lambda match: (_store_snippet(match.group(1)) or "\n"), cleaned)
        cleaned = cleaned.strip()

        return cleaned, VisionAgent._merge_reasoning_segments(*reasoning_parts)

    @staticmethod
    def _merge_reasoning_segments(*segments: Optional[str]) -> Optional[str]:
        parts: List[str] = []
        for segment in segments:
            if not segment:
                continue
            normalized = segment.strip()
            if normalized and normalized not in parts:
                parts.append(normalized)
        if parts:
            return "\n\n".join(parts)
        return None

    @staticmethod
    def _normalize_reasoning_value(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        if isinstance(value, (list, tuple)):
            parts = [VisionAgent._normalize_reasoning_value(item) for item in value]
            parts = [part for part in parts if part]
            if parts:
                return "\n".join(parts)
            return None
        if isinstance(value, dict):
            for candidate in ("text", "value", "content", "summary"):
                if candidate in value:
                    normalized = VisionAgent._normalize_reasoning_value(value[candidate])
                    if normalized:
                        return normalized
            nested = [VisionAgent._normalize_reasoning_value(v) for v in value.values()]
            nested = [part for part in nested if part]
            if nested:
                return "\n".join(nested)
            return None
        try:
            text = str(value).strip()
        except Exception:  # pragma: no cover - extremely defensive
            return None
        return text or None

    @staticmethod
    def _extract_reasoning_fields(payload: Dict[str, Any]) -> List[str]:
        reasoning_keys = (
            "reasoning",
            "thinking",
            "thoughts",
            "thought",
            "analysis",
            "plan",
            "inner_monologue",
            "chain_of_thought",
        )
        collected: List[str] = []
        for key in reasoning_keys:
            if key in payload:
                normalized = VisionAgent._normalize_reasoning_value(payload[key])
                if normalized:
                    collected.append(normalized)
        for extra_key in ("meta", "metadata"):
            extra = payload.get(extra_key)
            if isinstance(extra, dict):
                collected.extend(VisionAgent._extract_reasoning_fields(extra))
        return collected
