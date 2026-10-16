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
from typing import Any, Dict, List, Optional, Tuple

try:
    import docker
    from docker.errors import APIError, NotFound
    from docker.models.containers import Container
except ImportError:  # pragma: no cover - exercised only in minimal test envs
    docker = None  # type: ignore[assignment]

    class APIError(Exception):
        pass

    class NotFound(Exception):
        pass

    Container = Any  # type: ignore[misc, assignment]

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
    "DEL": "Delete",
    "SPACE": "space",
    "UP": "Up",
    "DOWN": "Down",
    "LEFT": "Left",
    "RIGHT": "Right",
    "HOME": "Home",
    "END": "End",
    "PAGEUP": "Prior",
    "PAGE_UP": "Prior",
    "PGUP": "Prior",
    "PAGEDOWN": "Next",
    "PAGE_DOWN": "Next",
    "PGDN": "Next",
    "INSERT": "Insert",
    "CTRL": "ctrl",
    "CONTROL": "ctrl",
    "ALT": "alt",
    "SHIFT": "shift",
    "META": "super",
    "SUPER": "super",
    "WIN": "super",
    "CMD": "ctrl",
    "COMMAND": "ctrl",
}
for _index in range(1, 13):
    SPECIAL_KEYS[f"F{_index}"] = f"F{_index}"


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


DEFAULT_IMAGE = os.getenv("VISIOS_SANDBOX_IMAGE", "visios-sandbox:local")
DEFAULT_NAME = os.getenv("VISIOS_SANDBOX_NAME", "visios-sandbox")
DEFAULT_PORT = _env_int("VISIOS_SANDBOX_PORT", 6901)
DEFAULT_WIDTH = _env_int("VISIOS_SANDBOX_WIDTH", 1600)
DEFAULT_HEIGHT = _env_int("VISIOS_SANDBOX_HEIGHT", 900)
DEFAULT_PLATFORM = os.getenv("VISIOS_SANDBOX_PLATFORM", "").strip() or None
DISPLAY = os.getenv("VISIOS_DISPLAY", ":0")
INPUT_BACKEND = os.getenv("VISIOS_INPUT_BACKEND", "xdotool").lower()
VNC_HOST = os.getenv("VISIOS_VNC_HOST", "127.0.0.1")
VNC_PORT = _env_int("VISIOS_VNC_PORT", 5900)
MAX_SCREENSHOT_RETRIES = _env_int("VISIOS_SCREENSHOT_RETRIES", 5)
SCREENSHOT_RETRY_DELAY = _env_float("VISIOS_SCREENSHOT_RETRY_DELAY", 2.0)
TYPE_DELAY_MS = _env_int("VISIOS_TYPE_DELAY_MS", 12)
COMMAND_TIMEOUT = _env_int("VISIOS_COMMAND_TIMEOUT", 30)
BOOT_TIMEOUT = _env_float("VISIOS_BOOT_TIMEOUT", 30.0)
SCROLL_DEFAULT_MAGNITUDE = 5.0
SCROLL_MIN_STEPS = 1
SCROLL_MAX_STEPS = 30
SCROLL_SMALL_SCALE = 0.6
SCROLL_LARGE_SCALE = 1.4
SCROLL_SMALL_CLAMP = 4.0
SCROLL_LARGE_FLOOR = 8.0
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

APP_ALIASES: Dict[str, str] = {
    "browser": "browser",
    "web": "browser",
    "web_browser": "browser",
    "firefox": "browser",
    "chrome": "browser",
    "chromium": "browser",
    "navigateur": "browser",
    "internet": "browser",
    "terminal": "terminal",
    "console": "terminal",
    "shell": "terminal",
    "files": "files",
    "file": "files",
    "file_manager": "files",
    "fichiers": "files",
    "explorer": "files",
    "thunar": "files",
    "text": "text_editor",
    "editor": "text_editor",
    "text_editor": "text_editor",
    "editeur": "text_editor",
    "mousepad": "text_editor",
    "notepad": "text_editor",
    "settings": "settings",
    "parametres": "settings",
    "paramètres": "settings",
    "writer": "writer",
    "document": "writer",
    "word": "writer",
    "calc": "calc",
    "spreadsheet": "calc",
    "tableur": "calc",
    "excel": "calc",
    "calculator": "calculator",
    "calculatrice": "calculator",
}

APP_COMMANDS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "browser": (("firefox",), ("firefox-esr",), ("chromium-browser",), ("chromium",), ("google-chrome",)),
    "terminal": (("xfce4-terminal",), ("gnome-terminal",), ("x-terminal-emulator",), ("xterm",)),
    "files": (("thunar",), ("pcmanfm",), ("nautilus",)),
    "text_editor": (("mousepad",), ("gedit",), ("xed",)),
    "settings": (("xfce4-settings-manager",),),
    "writer": (("libreoffice", "--writer"), ("lowriter",)),
    "calc": (("libreoffice", "--calc"), ("localc",)),
    "calculator": (("galculator",), ("gnome-calculator",), ("kcalc",)),
}


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
        platform: Optional[str] = DEFAULT_PLATFORM,
    ) -> None:
        self.image = image
        self.container_name = container_name
        self.host_vnc_port = host_vnc_port
        self.screen_size = screen_size
        self.platform = platform
        if docker is None:
            raise RuntimeError(
                "Docker SDK is not installed. Run `pip install -r requirements.txt` "
                "before starting the visiOS sandbox."
            )
        self.client = docker.from_env()
        self.container: Optional[Container] = None
        self.input_backend = INPUT_BACKEND
        self._desktop_prepared = False

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
                self._wait_for_display()
            self.container = container
            self._refresh_display_geometry()
            self._prepare_desktop()
            return container
        except NotFound:
            LOGGER.info("Creating container %s from %s", self.container_name, self.image)

        ports = {"6901/tcp": ("127.0.0.1", self.host_vnc_port)}
        env = {
            "VNC_RESOLUTION": f"{self.screen_size[0]}x{self.screen_size[1]}",
            "DISPLAY": DISPLAY,
        }

        run_kwargs = {
            "image": self.image,
            "name": self.container_name,
            "detach": True,
            "environment": env,
            "ports": ports,
            "tty": True,
            "stdin_open": True,
            "auto_remove": False,
            "shm_size": "512m",  # Firefox crashes tabs with the 64MB default
        }
        if self.platform:
            run_kwargs["platform"] = self.platform

        container = self.client.containers.run(**run_kwargs)
        self.container = container
        self._wait_for_display()
        self._refresh_display_geometry()
        self._prepare_desktop()
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
            self._desktop_prepared = False

    def status(self) -> str:
        """Return the docker status of the sandbox (running, exited, missing...)."""

        try:
            container = self.client.containers.get(self.container_name)
            self.container = container
            return (container.status or "unknown").lower()
        except NotFound:
            self.container = None
            return "missing"
        except Exception as exc:  # pragma: no cover - docker daemon unavailable
            LOGGER.debug("Unable to read sandbox status: %s", exc)
            return "unavailable"

    def is_running(self) -> bool:
        return self.status() == "running"

    def _prepare_desktop(self) -> None:
        """Best-effort tweaks that make screen-change detection reliable.

        A blinking text caret changes the framebuffer every ~500ms, which would
        make the screen look like it never settles. GTK apps (Firefox, terminal,
        editors) follow the XSETTINGS value set here.
        """

        if self._desktop_prepared:
            return
        script = (
            'pid=$(pgrep -u "$(id -u)" -x xfce4-session | head -n1); '
            'if [ -n "$pid" ]; then '
            "export $(tr '\\0' '\\n' < /proc/$pid/environ | grep '^DBUS_SESSION_BUS_ADDRESS=' | head -n1); "
            "fi; "
            "xfconf-query -c xsettings -p /Net/CursorBlink -n -t bool -s false 2>/dev/null || "
            "xfconf-query -c xsettings -p /Net/CursorBlink -s false 2>/dev/null || true"
        )
        try:
            self.execute_command(script)
            self._desktop_prepared = True
        except Exception as exc:  # pragma: no cover - depends on the desktop image
            LOGGER.debug("Unable to prepare desktop: %s", exc)

    def _wait_for_display(self) -> None:
        """Poll the X server until it answers instead of sleeping blindly."""

        deadline = time.monotonic() + BOOT_TIMEOUT
        while time.monotonic() < deadline:
            try:
                output = self.execute_command("xdotool getdisplaygeometry").strip()
                if output:
                    # The desktop session needs a moment after X is up.
                    time.sleep(1.0)
                    return
            except Exception:
                pass
            time.sleep(0.5)
        LOGGER.warning("Display did not become ready within %.0fs", BOOT_TIMEOUT)

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
            f"export DISPLAY={DISPLAY}; "
            "if command -v scrot >/dev/null 2>&1; then "
            "f=/tmp/visios-shot-$$.png; scrot --overwrite --silent \"$f\" && cat \"$f\"; rc=$?; rm -f \"$f\"; exit $rc; "
            "elif command -v import >/dev/null 2>&1; then import -silent -window root png:-; "
            "elif command -v xwd >/dev/null 2>&1 && command -v convert >/dev/null 2>&1; then xwd -root -silent | convert xwd:- png:-; "
            "elif command -v xfce4-screenshooter >/dev/null 2>&1; then "
            "rm -f /tmp/visios-shot.png && xfce4-screenshooter --fullscreen --save /tmp/visios-shot.png --no-popup >/dev/null 2>&1 && cat /tmp/visios-shot.png; "
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

    def screen_fingerprint(self) -> str:
        """Cheap hash of the framebuffer, computed inside the container.

        Only a 32-char digest crosses the docker boundary, which makes it cheap
        enough to poll while waiting for the UI to settle.
        """

        probe = (
            "if command -v xwd >/dev/null 2>&1; then xwd -root -silent | md5sum; "
            "else import -silent -window root rgb:- | md5sum; fi"
        )
        try:
            return self.execute_command(probe).split()[0]
        except Exception as exc:  # pragma: no cover - depends on container tools
            LOGGER.debug("Unable to fingerprint screen: %s", exc)
            return ""

    def wait_until_stable(
        self,
        reference: Optional[str] = None,
        timeout: float = 4.0,
        interval: float = 0.35,
        min_wait: float = 0.15,
    ) -> Tuple[bool, str]:
        """Wait until two consecutive fingerprints match.

        Returns ``(changed, fingerprint)`` where ``changed`` tells whether the
        settled screen differs from ``reference`` (usually taken before acting).
        """

        time.sleep(max(min_wait, 0.0))
        deadline = time.monotonic() + max(timeout, 0.0)
        previous = self.screen_fingerprint()
        if not previous:
            # Fingerprinting unavailable: fall back to a fixed pause.
            time.sleep(min(timeout, 1.0))
            return True, ""
        while time.monotonic() < deadline:
            time.sleep(interval)
            current = self.screen_fingerprint()
            if current == previous:
                break
            previous = current
        changed = reference is None or not reference or previous != reference
        return changed, previous

    def execute_command(self, cmd: str) -> str:
        """Execute an arbitrary shell command inside the container.

        This is primarily used for xdotool mouse/keyboard injections.
        """

        container = self._require_container()
        exec_cmd = f"bash -lc {shlex.quote(cmd)}"
        LOGGER.debug("Executing in container: %s", exec_cmd)
        result = container.exec_run(exec_cmd, demux=True, environment={"DISPLAY": DISPLAY})
        stdout, stderr = result.output if isinstance(result.output, tuple) else (result.output, None)
        if result.exit_code != 0:
            raise RuntimeError((stderr or stdout or b"").decode("utf-8", "ignore"))
        return (stdout or b"").decode("utf-8", "ignore")

    def run_shell(self, command: str, timeout: int = COMMAND_TIMEOUT) -> Tuple[int, str]:
        """Run a user-level shell command in the sandbox and capture its output.

        Unlike :meth:`execute_command`, failures are returned instead of raised so
        the agent can read the error and adapt.
        """

        container = self._require_container()
        wrapped = f"cd ~ && timeout {int(timeout)} bash -lc {shlex.quote(command)} 2>&1"
        result = container.exec_run(
            f"bash -lc {shlex.quote(wrapped)}",
            demux=False,
            environment={"DISPLAY": DISPLAY},
        )
        output = (result.output or b"").decode("utf-8", "ignore")
        if result.exit_code == 124:
            output += f"\n[timeout after {timeout}s]"
        return int(result.exit_code or 0), output.strip()

    def get_desktop_context(self) -> str:
        """Return lightweight desktop metadata to augment screenshots."""

        probe = (
            "printf 'active_window: '; "
            "xdotool getactivewindow getwindowname 2>/dev/null || echo '(none)'; "
            "printf 'open_windows:\\n'; "
            "if command -v wmctrl >/dev/null 2>&1; then "
            "wmctrl -l 2>/dev/null | awk '{$1=\"\"; $2=\"\"; $3=\"\"; sub(/^ +/, \"\"); print \"- \" $0}' | head -n 15; "
            "else "
            "xdotool search --onlyvisible --name . getwindowname %@ 2>/dev/null | sed 's/^/- /' | head -n 15; "
            "fi"
        )
        try:
            return self.execute_command(probe).strip()[:2000]
        except Exception as exc:  # pragma: no cover - desktop tools vary by image
            LOGGER.debug("Unable to collect desktop context: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Convenience helpers for pointer/keyboard interactions
    # ------------------------------------------------------------------
    def send_click(
        self,
        x: float,
        y: float,
        absolute: bool = False,
        button: int = 1,
        repeat: int = 1,
    ) -> Tuple[int, int]:
        """Click at the provided coordinates using xdotool."""

        target_x, target_y = self._coerce_coordinates(x, y, absolute)
        target_x, target_y = self._clamp_to_screen(target_x, target_y)
        button = button if button in (1, 2, 3) else 1
        repeat = max(1, min(int(repeat), 3))
        if self.input_backend == "vncdotool":
            self._vncdotool_command(f"move {target_x} {target_y}")
            for _ in range(repeat):
                self._vncdotool_command(f"click {button}")
        else:
            cmd = (
                f"xdotool mousemove --sync {target_x} {target_y} "
                f"click --clearmodifiers --repeat {repeat} --delay 90 {button}"
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
        """Type arbitrary text inside the sandbox.

        Newlines become Return key presses; ``<KEY>`` tokens are sent as keys.
        All chunks are chained in a single ``docker exec`` to keep typing fluid.
        """

        safe = text.replace("\r\n", "\n").replace("\n", "<ENTER>")
        if self.input_backend == "vncdotool":
            self._vncdotool_type(safe)
            return

        commands: List[str] = []
        for token_type, payload in self._tokenize_input(safe):
            if token_type == "text" and payload:
                commands.append(
                    f"xdotool type --delay {TYPE_DELAY_MS} --clearmodifiers {shlex.quote(payload)}"
                )
            elif token_type == "key" and payload:
                commands.append(f"xdotool key --clearmodifiers {shlex.quote(payload)}")
        if commands:
            self.execute_command(" && ".join(commands))

    def send_hotkey(self, keys: str) -> None:
        """Send a keyboard shortcut such as CTRL+L, <ALT+F4> or a sequence "ctrl+a Delete"."""

        sequence = self.normalize_key_sequence(keys)
        if not sequence:
            raise ValueError("Hotkey is empty")
        if self.input_backend == "vncdotool":
            for combo in sequence:
                self._vncdotool_command(f"key {shlex.quote(combo)}")
            return
        quoted = " ".join(shlex.quote(combo) for combo in sequence)
        self.execute_command(f"xdotool key --clearmodifiers --delay 60 {quoted}")

    def normalize_key_sequence(self, keys: str) -> List[str]:
        cleaned = (keys or "").strip()
        if not cleaned:
            return []
        if INPUT_TOKEN_PATTERN.search(cleaned):
            chunks = [chunk[1:-1] for chunk in INPUT_TOKEN_PATTERN.findall(cleaned)]
        else:
            chunks = re.split(r"[\s,]+", cleaned)
        sequence = [self._normalize_token(chunk) for chunk in chunks]
        return [combo for combo in sequence if combo]

    def send_drag(
        self,
        x: float,
        y: float,
        end_x: float,
        end_y: float,
        duration: float = 0.4,
        absolute: bool = False,
    ) -> Tuple[int, int]:
        """Drag from one coordinate to another."""

        start_x, start_y = self._clamp_to_screen(*self._coerce_coordinates(x, y, absolute))
        target_x, target_y = self._clamp_to_screen(*self._coerce_coordinates(end_x, end_y, absolute))
        pause = max(0.05, min(float(duration or 0.4), 5.0))

        if self.input_backend == "vncdotool":
            self._vncdotool_command(f"move {start_x} {start_y}")
            self._vncdotool_command("mousedown 1")
            time.sleep(pause)
            self._vncdotool_command(f"move {target_x} {target_y}")
            self._vncdotool_command("mouseup 1")
        else:
            mid_x = (start_x + target_x) // 2
            mid_y = (start_y + target_y) // 2
            self.execute_command(
                f"xdotool mousemove --sync {start_x} {start_y} mousedown 1 "
                f"sleep {pause / 2:.2f} mousemove --sync {mid_x} {mid_y} "
                f"sleep {pause / 2:.2f} mousemove --sync {target_x} {target_y} mouseup 1"
            )
        return target_x, target_y

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

    def open_application(self, app_name: str) -> bool:
        """Launch an allowlisted desktop application by semantic alias."""

        alias = self._normalize_app_alias(app_name)
        if not alias:
            return False
        for argv in APP_COMMANDS.get(alias, ()):
            if self._try_launch_command(argv):
                return True
        return False

    def open_url(self, url: str) -> bool:
        """Open a URL in the sandbox browser (new tab if it is already running)."""

        cleaned = (url or "").strip()
        if not cleaned:
            return False
        if not re.match(r"^[a-z][a-z0-9+.-]*://", cleaned, re.IGNORECASE):
            cleaned = f"https://{cleaned}"
        for argv in APP_COMMANDS["browser"]:
            if self._try_launch_command((*argv, cleaned)):
                return True
        return self._try_launch_command(("xdg-open", cleaned))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _coerce_coordinates(self, x: float, y: float, absolute: bool) -> Tuple[int, int]:
        """Convert legacy relative coordinates into actual pixels.

        The agent runner now always passes ``absolute=True`` with screen pixels;
        the heuristics below only remain for backwards compatibility.
        """

        width, height = self._ensure_screen_size()
        if absolute:
            return int(round(x)), int(round(y))

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
        if len(cleaned) > 1:
            cleaned = cleaned.replace("-", "+")
        parts = [part.strip() for part in cleaned.split("+") if part.strip()]
        if not parts:
            return "plus" if "+" in cleaned else ""
        normalized: List[str] = []
        for part in parts:
            key = SPECIAL_KEYS.get(part.upper().replace(" ", ""))
            if key is None:
                key = part.lower() if len(part) == 1 else part
            normalized.append(key)
        return "+".join(normalized)

    def _normalize_app_alias(self, app_name: str) -> str:
        cleaned = (app_name or "").strip().lower().replace("-", "_")
        if not cleaned:
            return ""
        cleaned = cleaned.replace(" ", "_")
        return APP_ALIASES.get(cleaned, "")

    def _try_launch_command(self, argv: Tuple[str, ...]) -> bool:
        if not argv:
            return False
        binary = argv[0]
        command = " ".join(shlex.quote(part) for part in argv)
        probe = (
            f"if command -v {shlex.quote(binary)} >/dev/null 2>&1; then "
            f"(setsid nohup {command} >/tmp/visios-launch.log 2>&1 &) ; "
            "echo launched; "
            "else exit 127; fi"
        )
        try:
            return "launched" in self.execute_command(probe)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # vncdotool helpers
    # ------------------------------------------------------------------
    def _vncdotool_command(self, command: str) -> None:
        endpoint = f"{VNC_HOST}::{VNC_PORT}"
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
        lowered = hint.lower()
        has_small = any(token in lowered for token in SCROLL_SMALL_TOKENS)
        has_large = any(token in lowered for token in SCROLL_LARGE_TOKENS)

        adjusted = magnitude
        if has_small:
            adjusted = min(adjusted, SCROLL_SMALL_CLAMP)
            adjusted = max(adjusted * SCROLL_SMALL_SCALE, 0.4)
        elif has_large:
            adjusted = max(adjusted, SCROLL_LARGE_FLOOR)
            adjusted = adjusted * SCROLL_LARGE_SCALE
        elif source == "default":
            adjusted = max(min(adjusted, 5.0), 1.5)
        return adjusted

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
