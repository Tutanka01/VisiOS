"""Sandbox management utilities for visiOS.

This module exposes the :class:`SandboxManager` that encapsulates everything
required to start/stop the remote desktop container, collect screenshots from it
and inject mouse/keyboard actions via ``xdotool`` executed inside the container.
"""
from __future__ import annotations

import io
import logging
import os
import re
import shlex
import time
from typing import List, Optional, Tuple

import docker
from docker.errors import NotFound, APIError
from docker.models.containers import Container

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None  # Pillow is only needed when screenshots must be inspected

LOGGER = logging.getLogger(__name__)
INPUT_TOKEN_PATTERN = re.compile(r"(<[^>]+>)")
SPECIAL_KEYS = {
    "ENTER": "Return",
    "RETURN": "Return",
    "TAB": "Tab",
    "ESC": "Escape",
    "ESCAPE": "Escape",
    "BACKSPACE": "BackSpace",
    "DELETE": "Delete",
    "SPACE": "space",
    "UP": "Up",
    "DOWN": "Down",
    "LEFT": "Left",
    "RIGHT": "Right",
    "CTRL": "ctrl",
    "ALT": "alt",
    "SHIFT": "shift",
    "META": "meta",
}


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default


DEFAULT_IMAGE = os.getenv("VISIOS_SANDBOX_IMAGE", "tutanka01/agentic:osbase")
DEFAULT_NAME = os.getenv("VISIOS_SANDBOX_NAME", "visios-sandbox")
DEFAULT_PORT = _env_int("VISIOS_SANDBOX_PORT", 6901)
DEFAULT_WIDTH = _env_int("VISIOS_SANDBOX_WIDTH", 1600)
DEFAULT_HEIGHT = _env_int("VISIOS_SANDBOX_HEIGHT", 900)
INPUT_BACKEND = os.getenv("VISIOS_INPUT_BACKEND", "xdotool").lower()
VNC_HOST = os.getenv("VISIOS_VNC_HOST", "127.0.0.1")
VNC_PORT = _env_int("VISIOS_VNC_PORT", 5901)
MAX_SCREENSHOT_RETRIES = _env_int("VISIOS_SCREENSHOT_RETRIES", 5)
SCREENSHOT_RETRY_DELAY = _env_float("VISIOS_SCREENSHOT_RETRY_DELAY", 2.0)
SCROLL_DEFAULT_MAGNITUDE = 5.0
SCROLL_MIN_STEPS = 1
SCROLL_MAX_STEPS = 30
SCROLL_SMALL_SCALE = 0.5
SCROLL_LARGE_SCALE = 2.0
SCROLL_UP_TOKENS = (
    "up",
    "haut",
    "vers le haut",
    "monte",
    "scroll up",
)
SCROLL_DOWN_TOKENS = (
    "down",
    "bas",
    "vers le bas",
    "descend",
    "scroll down",
)
SCROLL_SMALL_TOKENS = (
    "small",
    "petit",
    "slight",
    "peu",
)
SCROLL_LARGE_TOKENS = (
    "page",
    "large",
    "big",
    "grand",
    "beaucoup",
)


class SandboxManager:
    """Controls the lifecycle of the visiOS sandbox container.

    Parameters
    ----------
    image : str
        Docker image that embeds the remote desktop stack.
    container_name : str
        Deterministic name so we can re-attach on hot reloads.
    host_vnc_port : int
        Host port that will expose the container noVNC endpoint.
    screen_size : tuple[int, int]
        Default screen size (width, height) used when starting the container.
    """

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        container_name: str = DEFAULT_NAME,
        host_vnc_port: int = DEFAULT_PORT,
        screen_size: Tuple[int, int] = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
    ) -> None:
        self.image = image
        self.container_name = container_name
        self.host_vnc_port = host_vnc_port
        self.screen_size = screen_size
        self.client = docker.from_env()
        self.container: Optional[Container] = None
        self.input_backend = INPUT_BACKEND

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def start_container(self) -> Container:
        """Ensure the sandbox container is running and return it."""

        try:
            container = self.client.containers.get(self.container_name)
            if container.status != "running":
                LOGGER.info("Starting existing container %s", self.container_name)
                container.start()
            self.container = container
            self._refresh_display_geometry()
            return container
        except NotFound:
            LOGGER.info("Creating container %s from %s", self.container_name, self.image)

        ports = {"6901/tcp": ("127.0.0.1", self.host_vnc_port)}
        env = {
            "VNC_RESOLUTION": f"{self.screen_size[0]}x{self.screen_size[1]}",
            "DISPLAY": ":0",
        }

        container = self.client.containers.run(
            self.image,
            name=self.container_name,
            detach=True,
            environment=env,
            ports=ports,
            tty=True,
            stdin_open=True,
            auto_remove=False,
        )
        # Give the VNC stack a short moment to boot before returning control.
        time.sleep(5)
        self.container = container
        self._refresh_display_geometry()
        return container

    def stop_container(self) -> None:
        """Stop and remove the sandbox container if it exists."""

        if not self.container:
            try:
                self.container = self.client.containers.get(self.container_name)
            except NotFound:
                LOGGER.info("No sandbox container to stop")
                return

        try:
            LOGGER.info("Stopping container %s", self.container_name)
            self.container.stop(timeout=5)
            self.container.remove()
        except APIError as exc:
            LOGGER.error("Unable to stop container: %s", exc)
        finally:
            self.container = None

    # ------------------------------------------------------------------
    # Runtime utilities
    # ------------------------------------------------------------------
    def get_screenshot(self) -> Tuple[bytes, Tuple[int, int]]:
        """Capture a screenshot from the running container.

        Returns
        -------
        tuple
            Raw PNG bytes and the detected (width, height).
        """

        container = self._require_container()
        capture_cmd = (
            "bash -lc '"
            "if command -v import >/dev/null 2>&1; then DISPLAY=:0 import -window root png:-; "
            "elif command -v xfce4-screenshooter >/dev/null 2>&1; then "
            "rm -f /tmp/visios-shot.png && DISPLAY=:0 xfce4-screenshooter --fullscreen --save /tmp/visios-shot.png --no-popup >/dev/null 2>&1 && cat /tmp/visios-shot.png; "
            "elif command -v xwd >/dev/null 2>&1 && command -v convert >/dev/null 2>&1; then DISPLAY=:0 xwd -root -silent | convert xwd:- png:-; "
            "else exit 1; fi'"
        )
        last_error: Optional[str] = None
        img_bytes: bytes = b""
        for attempt in range(1, MAX_SCREENSHOT_RETRIES + 1):
            result = container.exec_run(capture_cmd, demux=True)
            stdout, stderr = result.output if isinstance(result.output, tuple) else (result.output, None)

            if result.exit_code == 0 and stdout:
                img_bytes = stdout
                break

            error_msg = (stderr or stdout or b"").decode("utf-8", "ignore").strip()
            last_error = error_msg or f"exit_code={result.exit_code}"
            LOGGER.warning(
                "Screenshot attempt %s/%s failed (exit_code=%s): %s",
                attempt,
                MAX_SCREENSHOT_RETRIES,
                result.exit_code,
                last_error,
            )
            if attempt < MAX_SCREENSHOT_RETRIES:
                time.sleep(SCREENSHOT_RETRY_DELAY)
        else:  # pragma: no cover - only hit when loop never breaks
            raise RuntimeError(f"Screenshot command failed after retries: {last_error}")

        if not img_bytes:
            raise RuntimeError("Screenshot command produced no data despite successful exit")
        width, height = self.screen_size
        if Image is not None and img_bytes:
            with Image.open(io.BytesIO(img_bytes)) as img:
                width, height = img.size
        self._update_screen_size(width, height)
        return img_bytes, (width, height)

    def execute_command(self, cmd: str) -> str:
        """Execute an arbitrary shell command inside the container.

        This is primarily used for xdotool mouse/keyboard injections.
        """

        container = self._require_container()
        exec_cmd = f"bash -lc {shlex.quote(cmd)}"
        LOGGER.debug("Executing in container: %s", exec_cmd)
        result = container.exec_run(exec_cmd, demux=True)
        stdout, stderr = result.output if isinstance(result.output, tuple) else (result.output, None)
        if result.exit_code != 0:
            raise RuntimeError((stderr or stdout or b"").decode("utf-8", "ignore"))
        return (stdout or b"").decode("utf-8", "ignore")

    # ------------------------------------------------------------------
    # Convenience helpers for pointer/keyboard interactions
    # ------------------------------------------------------------------
    def send_click(self, x: float, y: float, absolute: bool = False) -> Tuple[int, int]:
        """Click at the provided coordinates using xdotool."""

        target_x, target_y = self._coerce_coordinates(x, y, absolute)
        target_x, target_y = self._clamp_to_screen(target_x, target_y)
        if self.input_backend == "vncdotool":
            self._vncdotool_command(f"move {target_x} {target_y}")
            self._vncdotool_command("click 1")
        else:
            cmd = (
                f"xdotool mousemove --sync {target_x} {target_y} && "
                "xdotool click --clearmodifiers 1"
            )
            self.execute_command(cmd)
        return target_x, target_y

    def move_pointer(self, x: float, y: float, absolute: bool = False) -> Tuple[int, int]:
        """Move the mouse pointer without clicking and return the effective coordinates."""

        target_x, target_y = self._coerce_coordinates(x, y, absolute)
        target_x, target_y = self._clamp_to_screen(target_x, target_y)
        if self.input_backend == "vncdotool":
            self._vncdotool_command(f"move {target_x} {target_y}")
        else:
            self.execute_command(f"xdotool mousemove --sync {target_x} {target_y}")
        return target_x, target_y

    def send_text(self, text: str) -> None:
        """Type arbitrary text inside the sandbox."""

        safe = text.replace("\n", " ")
        if self.input_backend == "vncdotool":
            self._vncdotool_type(safe)
            return

        for token_type, payload in self._tokenize_input(safe):
            if token_type == "text" and payload:
                self.execute_command(
                    f"xdotool type --delay 50 --clearmodifiers {shlex.quote(payload)}"
                )
            elif token_type == "key" and payload:
                self.execute_command(f"xdotool key --clearmodifiers {shlex.quote(payload)}")

    def send_scroll(self, delta_lines: Optional[float], hint: Optional[str] = None) -> bool:
        """Scroll vertically using mouse wheel events."""

        direction, repeat = self._normalize_scroll_request(delta_lines, hint)
        if repeat == 0:
            LOGGER.debug("Ignoring scroll request with no magnitude (delta=%s, hint=%s)", delta_lines, hint)
            return False
        button = 5 if direction > 0 else 4
        if self.input_backend == "vncdotool":
            for _ in range(repeat):
                self._vncdotool_command(f"click {button}")
        else:
            self.execute_command(f"xdotool click --repeat {repeat} --delay 35 {button}")
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _coerce_coordinates(self, x: float, y: float, absolute: bool) -> Tuple[int, int]:
        """Convert relative coordinates into actual pixels."""

        width, height = self._ensure_screen_size()
        if absolute:
            return int(x), int(y)

        if 0 <= x <= 1 and 0 <= y <= 1:
            return int(x * width), int(y * height)

        if 0 <= x <= 1000 and 0 <= y <= 1000:
            return int((x / 1000) * width), int((y / 1000) * height)

        # Fallback: assume already in pixels
        return int(x), int(y)

    def _clamp_to_screen(self, x: int, y: int) -> Tuple[int, int]:
        width, height = self._ensure_screen_size()
        if width <= 0 or height <= 0:
            return x, y
        clamped_x = max(0, min(width - 1, x))
        clamped_y = max(0, min(height - 1, y))
        return clamped_x, clamped_y

    def _ensure_screen_size(self) -> Tuple[int, int]:
        width, height = self.screen_size
        if width > 0 and height > 0:
            return width, height
        return self._refresh_display_geometry()

    def _refresh_display_geometry(self) -> Tuple[int, int]:
        try:
            output = self.execute_command("xdotool getdisplaygeometry").strip()
            parts = output.split()
            if len(parts) >= 2:
                width, height = int(parts[0]), int(parts[1])
                self._update_screen_size(width, height)
        except Exception as exc:  # pragma: no cover - depends on container tools
            LOGGER.debug("Unable to refresh display geometry: %s", exc)
        return self.screen_size

    def _update_screen_size(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        if (width, height) != self.screen_size:
            LOGGER.info("Detected display geometry change: from %s to %s", self.screen_size, (width, height))
        self.screen_size = (width, height)

    def _require_container(self) -> Container:
        if self.container is None:
            try:
                self.container = self.client.containers.get(self.container_name)
            except NotFound as exc:  # pragma: no cover
                raise RuntimeError("Sandbox container is not running") from exc
        return self.container

    # ------------------------------------------------------------------
    # Text helper utilities
    # ------------------------------------------------------------------
    def _tokenize_input(self, text: str) -> List[Tuple[str, str]]:
        tokens: List[Tuple[str, str]] = []
        for chunk in INPUT_TOKEN_PATTERN.split(text):
            if not chunk:
                continue
            if chunk.startswith("<") and chunk.endswith(">"):
                normalized = self._normalize_token(chunk[1:-1])
                if normalized:
                    tokens.append(("key", normalized))
            else:
                tokens.append(("text", chunk))
        return tokens

    def _normalize_token(self, token: str) -> str:
        cleaned = token.strip()
        if not cleaned:
            return ""
        parts = [part.strip() for part in cleaned.replace("-", "+").split("+") if part.strip()]
        if not parts:
            return ""
        normalized: List[str] = []
        for part in parts:
            key = SPECIAL_KEYS.get(part.upper(), part.lower())
            normalized.append(key)
        return "+".join(normalized)

    # ------------------------------------------------------------------
    # vncdotool helpers
    # ------------------------------------------------------------------
    def _vncdotool_command(self, command: str) -> None:
        endpoint = f"{VNC_HOST}:{VNC_PORT}"
        self.execute_command(f"vncdotool -s {shlex.quote(endpoint)} {command}")

    def _vncdotool_type(self, text: str) -> None:
        tokens = self._tokenize_input(text)
        for token_type, payload in tokens:
            if token_type == "text" and payload:
                self._vncdotool_command(f"type {shlex.quote(payload)}")
            elif token_type == "key" and payload:
                self._vncdotool_command(f"key {shlex.quote(payload)}")

    # ------------------------------------------------------------------
    # Scroll helpers
    # ------------------------------------------------------------------
    def _normalize_scroll_request(
        self, delta_lines: Optional[float], hint: Optional[str]
    ) -> Tuple[int, int]:
        direction = 0
        magnitude = 0.0
        magnitude_source = "none"
        if delta_lines is not None:
            try:
                magnitude = abs(float(delta_lines))
                if magnitude > 0:
                    direction = 1 if float(delta_lines) > 0 else -1
                    magnitude_source = "delta"
            except (TypeError, ValueError):
                magnitude = 0.0

        hint_direction = self._parse_scroll_direction_hint(hint)
        if direction == 0 and hint_direction != 0:
            direction = hint_direction

        if magnitude == 0.0:
            magnitude, magnitude_source = self._parse_scroll_magnitude_hint(hint)
        if magnitude == 0.0:
            magnitude = SCROLL_DEFAULT_MAGNITUDE
            magnitude_source = "default"

        magnitude = self._apply_scroll_hint_scaling(magnitude, hint, magnitude_source)

        steps = self._scale_scroll_magnitude(magnitude)
        if steps == 0:
            return direction or 0, 0
        return (direction or 1), steps

    def _parse_scroll_direction_hint(self, hint: Optional[str]) -> int:
        if not hint:
            return 0
        lowered = hint.lower()
        for token in SCROLL_UP_TOKENS:
            if token in lowered:
                return -1
        for token in SCROLL_DOWN_TOKENS:
            if token in lowered:
                return 1
        return 0

    def _parse_scroll_magnitude_hint(self, hint: Optional[str]) -> Tuple[float, str]:
        if not hint:
            return 0.0, "none"
        lowered = hint.lower()
        number_match = re.search(r"(-?\d+(?:\.\d+)?)", lowered)
        if number_match:
            try:
                value = abs(float(number_match.group(1)))
                return value, "hint-number"
            except ValueError:
                pass
        for token in SCROLL_LARGE_TOKENS:
            if token in lowered:
                return 18.0, "hint-large"
        for token in SCROLL_SMALL_TOKENS:
            if token in lowered:
                return 3.0, "hint-small"
        return 0.0, "none"

    def _apply_scroll_hint_scaling(self, magnitude: float, hint: Optional[str], source: str) -> float:
        if magnitude <= 0 or not hint:
            return magnitude
        if source not in {"delta", "hint-number"}:
            return magnitude
        lowered = hint.lower()
        multiplier = 1.0
        if any(token in lowered for token in SCROLL_SMALL_TOKENS):
            multiplier *= SCROLL_SMALL_SCALE
        if any(token in lowered for token in SCROLL_LARGE_TOKENS):
            multiplier *= SCROLL_LARGE_SCALE
        return max(magnitude * multiplier, 0.1)

    def _scale_scroll_magnitude(self, magnitude: float) -> int:
        if magnitude <= 0:
            return 0
        if magnitude <= 1:
            steps = int(round(magnitude * 6))
        elif magnitude <= 5:
            steps = int(round(magnitude * 2))
        elif magnitude <= 100:
            steps = int(round(magnitude / 4))
        else:
            steps = SCROLL_MAX_STEPS
        steps = max(SCROLL_MIN_STEPS, steps)
        return min(steps, SCROLL_MAX_STEPS)
