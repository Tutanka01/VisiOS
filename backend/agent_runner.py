"""Reusable agent execution loop for visiOS.

The UI should stay a cockpit. This module owns the operational loop:
observe the sandbox, ask the vision agent for the next action, execute it,
wait for the screen to settle, tell the model what changed, and stop on
success, loops, safety blocks or a user stop request.

Everything the loop does is published as structured events (``on_event``) so a
UI can render the run live.
"""
from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.container_manager import SandboxManager
from backend.vision_agent import AgentAction, StepContext, VisionAgent

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None


LogCallback = Callable[[str], None]
StepCallback = Callable[["AgentStepRecord"], None]
EventCallback = Callable[[str, Dict[str, Any]], None]
ConfirmCallback = Callable[[AgentAction], bool]


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "oui"}


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


RISK_KEYWORDS = (
    "password",
    "mot de passe",
    "credential",
    "secret",
    "token",
    "delete",
    "supprimer",
    "remove",
    "rm ",
    "format",
    "wipe",
    "payment",
    "paiement",
    "purchase",
    "acheter",
    "buy",
    "bank",
    "banque",
    "send email",
    "envoyer",
    "login",
    "connexion",
    "sudo",
)

DANGEROUS_COMMAND = re.compile(
    r"(rm\s+-[a-z]*r[a-z]*f?\s+(/|~|\$HOME)(\s|$)|mkfs|dd\s+if=|:\(\)\s*\{|shutdown|reboot|halt\b|"
    r"chmod\s+-R\s+777\s+/|>\s*/dev/sd|curl[^|]*\|\s*(sudo\s+)?(ba)?sh|wget[^|]*\|\s*(sudo\s+)?(ba)?sh)",
    re.IGNORECASE,
)

# How long to wait for the screen to react, per action kind (seconds).
# (time allowed for a first visible change, time allowed for it to settle)
SETTLE_PROFILES: Dict[str, Tuple[float, float]] = {
    "click": (1.2, 3.0),
    "double_click": (2.5, 4.0),
    "right_click": (1.0, 2.0),
    "move": (0.4, 1.0),
    "type": (0.6, 2.5),
    "hotkey": (1.5, 3.5),
    "scroll": (0.8, 1.5),
    "drag": (1.0, 2.0),
    "open_app": (10.0, 5.0),
    "open_url": (8.0, 6.0),
    "run_command": (0.3, 1.0),
    "wait": (0.0, 1.0),
}

FINAL_STATUSES = {"done", "failed", "fail", "blocked", "loop_guard", "max_steps", "stopped", "error"}


@dataclass
class AgentRunConfig:
    """Runtime knobs for one autonomous run."""

    max_steps: int = _env_int("VISIOS_AGENT_MAX_STEPS", 25)
    action_delay_seconds: float = _env_float("VISIOS_ACTION_DELAY", 0.3)
    heavy_action_delay_seconds: float = _env_float("VISIOS_HEAVY_ACTION_DELAY", 3.0)
    repeat_tolerance: int = _env_int("VISIOS_REPEAT_TOLERANCE", 2)
    safety_mode: str = os.getenv("VISIOS_SAFETY_MODE", "balanced")
    trajectory_dir: str = os.getenv("VISIOS_TRAJECTORY_DIR", "trajectories")
    save_screenshots: bool = _env_bool("VISIOS_SAVE_SCREENSHOTS", True)
    allow_shell: bool = _env_bool("VISIOS_ALLOW_SHELL", True)
    max_invalid_responses: int = 3
    confirmation_timeout: float = 300.0


@dataclass
class AgentStepRecord:
    """One step in an agent trajectory."""

    run_id: str
    step: int
    action: AgentAction
    screenshot: bytes
    screen_size: Tuple[int, int]
    desktop_context: str
    status: str
    detail: str = ""
    executed: bool = False
    pointer: Optional[Tuple[int, int]] = None
    screenshot_path: Optional[str] = None
    started_at: str = ""
    duration_ms: int = 0
    screen_changed: Optional[bool] = None


@dataclass
class AgentRunResult:
    """Summary returned after the run finishes."""

    run_id: str
    status: str
    steps: int
    reason: str
    trajectory_path: Optional[str] = None


@dataclass
class SafetyDecision:
    allowed: bool
    reason: str = ""
    needs_confirmation: bool = False


@dataclass
class ExecutionOutcome:
    executed: bool
    status: str
    detail: str
    output: str = ""
    kind: str = ""


class TrajectoryRecorder:
    """Append-only local trajectory recorder."""

    def __init__(self, base_dir: str, run_id: str, save_screenshots: bool = True) -> None:
        self.base_dir = Path(base_dir)
        self.run_id = run_id
        self.save_screenshots = save_screenshots
        self.run_dir = self.base_dir / run_id
        self.events_path = self.run_dir / "events.jsonl"
        self._enabled = True
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def write_header(self, user_prompt: str, config: AgentRunConfig) -> None:
        self.write_event(
            "run_started",
            {
                "prompt": user_prompt,
                "config": asdict(config),
            },
        )

    def write_step(self, record: AgentStepRecord) -> Optional[str]:
        if not self.enabled:
            return None
        screenshot_path = None
        if self.save_screenshots and record.screenshot:
            screenshot_path = self.run_dir / f"step_{record.step:03d}.png"
            try:
                screenshot_path.write_bytes(record.screenshot)
            except OSError:
                screenshot_path = None

        action_payload = asdict(record.action)
        action_payload.pop("raw_response", None)
        payload = {
            "step": record.step,
            "status": record.status,
            "detail": record.detail,
            "executed": record.executed,
            "screen_changed": record.screen_changed,
            "screen_size": list(record.screen_size),
            "desktop_context": record.desktop_context,
            "pointer": list(record.pointer) if record.pointer else None,
            "duration_ms": record.duration_ms,
            "started_at": record.started_at,
            "screenshot_path": str(screenshot_path) if screenshot_path else None,
            "action": action_payload,
            "raw_response": record.action.raw_response,
        }
        self.write_event("step", payload)
        return str(screenshot_path) if screenshot_path else None

    def write_result(self, result: AgentRunResult) -> None:
        self.write_event("run_finished", asdict(result))

    def write_event(self, event_type: str, payload: Dict[str, object]) -> None:
        if not self.enabled:
            return
        event = {
            "type": event_type,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            **payload,
        }
        try:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except OSError:
            self._enabled = False


class RunCancelled(Exception):
    """Raised internally when the user stops the run."""


class AgentRunner:
    """Coordinates observe → decide → act → settle cycles."""

    def __init__(
        self,
        sandbox: SandboxManager,
        vision: VisionAgent,
        config: Optional[AgentRunConfig] = None,
        log: Optional[LogCallback] = None,
        on_step: Optional[StepCallback] = None,
        on_event: Optional[EventCallback] = None,
        confirm: Optional[ConfirmCallback] = None,
    ) -> None:
        self.sandbox = sandbox
        self.vision = vision
        self.config = config or AgentRunConfig()
        self._log = log or (lambda _message: None)
        self.on_step = on_step
        self.on_event = on_event
        self.confirm = confirm
        self.pointer: Optional[Tuple[int, int]] = None
        self._recent_actions: List[str] = []
        self._history: List[str] = []
        self._notes: List[str] = []
        self._notes_lock = threading.Lock()
        self._stop = threading.Event()
        self.run_id: Optional[str] = None

    # ------------------------------------------------------------------
    # External controls (thread-safe)
    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def add_note(self, text: str) -> None:
        cleaned = (text or "").strip()
        if cleaned:
            with self._notes_lock:
                self._notes.append(cleaned)

    def _take_notes(self) -> List[str]:
        with self._notes_lock:
            notes, self._notes = self._notes, []
        return notes

    # ------------------------------------------------------------------
    def log(self, message: str) -> None:
        self._log(message)
        self._emit("log", {"message": message})

    def run(self, user_prompt: str, conversation: str = "") -> AgentRunResult:
        run_id = self._new_run_id()
        self.run_id = run_id
        recorder = TrajectoryRecorder(
            self.config.trajectory_dir,
            run_id,
            save_screenshots=self.config.save_screenshots,
        )
        recorder.write_header(user_prompt, self.config)
        self._emit("run_started", {"run_id": run_id, "goal": user_prompt, "max_steps": self.config.max_steps})
        self.log(f"Run {run_id}: objectif reçu")

        final_status = "stopped"
        final_reason = "Aucune action exécutée."
        steps_done = 0
        try:
            final_status, final_reason, steps_done = self._loop(run_id, user_prompt, conversation, recorder)
        except RunCancelled:
            final_status, final_reason = "stopped", "Arrêté à ta demande."
        except Exception as exc:  # pragma: no cover - depends on runtime desktop/model
            final_status, final_reason = "error", f"{type(exc).__name__}: {exc}"

        result = AgentRunResult(
            run_id=run_id,
            status=final_status,
            steps=steps_done,
            reason=final_reason,
            trajectory_path=str(recorder.events_path) if recorder.enabled else None,
        )
        recorder.write_result(result)
        self.log(f"Run {run_id}: {final_status} - {final_reason}")
        self._emit("run_finished", asdict(result))
        return result

    def _loop(
        self,
        run_id: str,
        user_prompt: str,
        conversation: str,
        recorder: TrajectoryRecorder,
    ) -> Tuple[str, str, int]:
        self._phase("booting", "Préparation du bureau")
        self.sandbox.start_container()

        last_signature: Optional[Tuple[object, ...]] = None
        no_effect_repeats = 0
        invalid_streak = 0
        last_result = ""
        steps_done = 0

        for step in range(1, self.config.max_steps + 1):
            self._check_cancel()
            steps_done = step
            started = time.monotonic()
            started_at = dt.datetime.now(dt.timezone.utc).isoformat()
            self._emit("step_started", {"step": step, "max_steps": self.config.max_steps})

            # ---- observe -------------------------------------------------
            self._phase("observing", "Regarde l'écran", step)
            screenshot, screen_size = self.sandbox.get_screenshot()
            desktop_context = self.sandbox.get_desktop_context()
            self._emit(
                "observation",
                {"step": step, "screen_size": list(screen_size), "thumbnail": self._thumbnail(screenshot)},
            )

            # ---- decide --------------------------------------------------
            notes = self._take_notes()
            for note in notes:
                self.log(f"Message pris en compte: {note}")
            context = StepContext(
                goal=user_prompt,
                step=step,
                max_steps=self.config.max_steps,
                history=list(self._history),
                last_result=last_result,
                desktop_context=desktop_context,
                user_notes=notes,
                conversation=conversation,
            )
            self._phase("thinking", "Réfléchit", step)
            think_started = time.monotonic()
            action = self.vision.decide_action(
                screenshot,
                user_prompt,
                screen_size=screen_size,
                desktop_context=desktop_context,
                recent_actions=self._recent_action_summary(),
                context=context,
                on_stream=lambda kind, delta: self._emit("model_stream", {"step": step, "kind": kind, "delta": delta}),
                should_cancel=self._stop.is_set,
            )
            think_ms = int((time.monotonic() - think_started) * 1000)
            self._check_cancel()
            if action.reasoning:
                self.log(f"Reasoning: {action.reasoning[:500]}")
            self.log(f"Action proposée: {action.summary()}")
            self._emit(
                "action_planned",
                {"step": step, "think_ms": think_ms, "action": self._action_payload(action, screen_size)},
            )

            # ---- invalid answers ----------------------------------------
            kind = action._normalized_type()
            if action.parse_error or kind in {"", "noop"}:
                invalid_streak += 1
                detail = "Réponse du modèle inexploitable." if action.parse_error else "Aucune action proposée."
                self._finish_step(
                    recorder, run_id, step, action, screenshot, screen_size, desktop_context,
                    "invalid", detail, started_at, started,
                )
                self._remember(step, action, "invalid answer, no action executed")
                last_result = (
                    "Your last reply did not contain a usable action. Reply with ONE JSON object "
                    "containing a valid \"action\" field."
                )
                if invalid_streak >= self.config.max_invalid_responses:
                    return "failed", "Le modèle n'a pas renvoyé d'action exploitable.", step
                continue
            invalid_streak = 0

            # ---- safety --------------------------------------------------
            safety = self._check_safety(user_prompt, action)
            if safety.needs_confirmation and safety.allowed:
                approved = self._ask_confirmation(step, action, safety.reason)
                if not approved:
                    safety = SafetyDecision(False, "Action refusée par l'utilisateur.")
            if not safety.allowed:
                self._finish_step(
                    recorder, run_id, step, action, screenshot, screen_size, desktop_context,
                    "blocked", safety.reason, started_at, started,
                )
                self.log(f"Blocage sécurité: {safety.reason}")
                return "blocked", safety.reason, step

            # ---- terminal actions ----------------------------------------
            if kind in {"done", "fail"}:
                status = "done" if kind == "done" else "failed"
                detail = action.text or ("Objectif terminé." if kind == "done" else "Impossible de continuer.")
                self._finish_step(
                    recorder, run_id, step, action, screenshot, screen_size, desktop_context,
                    status, detail, started_at, started,
                )
                return status, detail, step

            # ---- act -----------------------------------------------------
            self._phase("acting", action.summary(), step)
            reference = self._fingerprint()
            try:
                outcome = self._execute(action)
            except Exception as exc:  # pragma: no cover - depends on runtime desktop
                outcome = ExecutionOutcome(False, "failed", f"{type(exc).__name__}: {exc}", kind=kind)

            changed: Optional[bool] = None
            if outcome.executed:
                self._phase("settling", "Attend que l'écran se stabilise", step)
                changed = self._settle(kind, reference, action)

            self._finish_step(
                recorder, run_id, step, action, screenshot, screen_size, desktop_context,
                outcome.status, outcome.detail, started_at, started,
                executed=outcome.executed, pointer=self.pointer, changed=changed,
            )

            # ---- feedback for the next turn ------------------------------
            last_result, effect = self._describe_outcome(action, outcome, changed)
            self._remember(step, action, effect)

            signature = self._action_signature(action)
            if changed is False and signature == last_signature and kind not in {"wait", "run_command"}:
                no_effect_repeats += 1
                if no_effect_repeats >= self.config.repeat_tolerance:
                    detail = (
                        f"La même action a été répétée {no_effect_repeats + 1} fois sans effet; "
                        "arrêt pour éviter une boucle."
                    )
                    self.log(detail)
                    return "loop_guard", detail, step
                last_result += (
                    "\nWARNING: you repeated exactly the same action and nothing changed. "
                    "You MUST try a different approach now."
                )
            elif changed is not False:
                no_effect_repeats = 0
            last_signature = signature

        return (
            "max_steps",
            "Nombre maximal d'étapes atteint. Augmente la limite ou donne une instruction plus ciblée.",
            steps_done,
        )

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def _execute(self, action: AgentAction) -> ExecutionOutcome:
        kind = action._normalized_type()
        absolute = action.coords_absolute

        if kind in {"click", "double_click", "right_click"} and action.x is not None and action.y is not None:
            button = 3 if kind == "right_click" else 1
            repeat = 2 if kind == "double_click" else 1
            self.pointer = self.sandbox.send_click(action.x, action.y, absolute=absolute, button=button, repeat=repeat)
            return ExecutionOutcome(True, "executed", f"{kind} @ {self.pointer}", kind=kind)

        if kind == "move" and action.x is not None and action.y is not None:
            self.pointer = self.sandbox.move_pointer(action.x, action.y, absolute=absolute)
            return ExecutionOutcome(True, "executed", f"Pointeur @ {self.pointer}", kind=kind)

        if kind == "type" and (action.text or action.submit):
            if action.x is not None and action.y is not None:
                self.pointer = self.sandbox.send_click(action.x, action.y, absolute=absolute)
                time.sleep(0.2)
            text = action.text or ""
            if action.submit and not text.rstrip().upper().endswith("<ENTER>"):
                text += "<ENTER>"
            self.sandbox.send_text(text)
            return ExecutionOutcome(True, "executed", f"Saisi: {action.text or ''}", kind=kind)

        if kind == "hotkey" and action.text:
            self.sandbox.send_hotkey(action.text)
            return ExecutionOutcome(True, "executed", f"Touches: {action.text}", kind=kind)

        if kind == "drag" and action.has_drag_target():
            self.pointer = self.sandbox.send_drag(
                action.x,
                action.y,
                action.end_x,
                action.end_y,
                duration=action.duration or 0.4,
                absolute=absolute,
            )
            return ExecutionOutcome(True, "executed", f"Glissé vers {self.pointer}", kind=kind)

        if kind == "scroll":
            anchor = self._move_scroll_anchor(action)
            scroll_value = action.scroll_amount
            if scroll_value is None and action.x is None and action.y is not None:
                scroll_value = action.y
            hint = " ".join(
                chunk
                for chunk in (action.action_type, action.text)
                if isinstance(chunk, str) and chunk.strip()
            )
            if self.sandbox.send_scroll(scroll_value, hint or None):
                return ExecutionOutcome(True, "executed", f"Défilement @ {anchor} ({scroll_value})", kind=kind)
            return ExecutionOutcome(False, "noop", "Défilement ignoré: amplitude manquante.", kind=kind)

        if kind == "wait":
            wait_seconds = min(max(action.duration or 1.5, 0.2), 10.0)
            self._sleep(wait_seconds)
            return ExecutionOutcome(True, "executed", f"Attente {wait_seconds:.1f}s", kind=kind)

        if kind == "open_app":
            app_name = action.app or action.text or ""
            if self.sandbox.open_application(app_name):
                return ExecutionOutcome(True, "executed", f"Application lancée: {app_name}", kind=kind)
            return ExecutionOutcome(False, "failed", f"Application inconnue: {app_name}", kind=kind)

        if kind == "open_url":
            url = action.url or action.text or ""
            if self.sandbox.open_url(url):
                return ExecutionOutcome(True, "executed", f"Ouverture de {url}", kind=kind)
            return ExecutionOutcome(False, "failed", f"Impossible d'ouvrir {url}", kind=kind)

        if kind == "run_command":
            command = action.command or action.text or ""
            if not command.strip():
                return ExecutionOutcome(False, "failed", "Commande vide.", kind=kind)
            exit_code, output = self.sandbox.run_shell(command)
            status = "executed" if exit_code == 0 else "failed"
            return ExecutionOutcome(
                True, status, f"$ {command} (code {exit_code})", output=output, kind=kind
            )

        return ExecutionOutcome(False, "noop", "Action incomplète (coordonnées ou texte manquant).", kind=kind)

    def _move_scroll_anchor(self, action: AgentAction) -> Optional[Tuple[int, int]]:
        pointer_coords = action.pointer()
        if pointer_coords:
            self.pointer = self.sandbox.move_pointer(*pointer_coords, absolute=action.coords_absolute)
        elif self.pointer:
            self.pointer = self.sandbox.move_pointer(*self.pointer, absolute=True)
        else:
            width, height = self.sandbox.screen_size
            self.pointer = self.sandbox.move_pointer(width / 2, height / 2, absolute=True)
        return self.pointer

    def _fingerprint(self) -> str:
        probe = getattr(self.sandbox, "screen_fingerprint", None)
        if probe is None:
            return ""
        try:
            return probe() or ""
        except Exception:
            return ""

    def _settle(self, kind: str, reference: str, action: AgentAction) -> Optional[bool]:
        """Wait for the UI to react and then to stop moving.

        Returns whether the screen differs from ``reference`` (None if unknown).
        """

        first_change, stable_budget = SETTLE_PROFILES.get(kind, (1.0, 2.0))
        base_delay = max(self.config.action_delay_seconds, 0.0)
        if not reference:
            self._sleep(max(base_delay, 1.0 if kind not in {"open_app", "open_url"} else 3.0))
            return None

        interval = 0.3
        current = reference
        deadline = time.monotonic() + first_change
        self._sleep(min(base_delay, 0.5))
        current = self._fingerprint()
        while current == reference and time.monotonic() < deadline:
            self._sleep(interval)
            current = self._fingerprint()
        if current == reference:
            return False

        deadline = time.monotonic() + stable_budget
        while time.monotonic() < deadline:
            self._sleep(interval)
            latest = self._fingerprint()
            if latest == current:
                break
            current = latest
        return current != reference

    def _describe_outcome(
        self, action: AgentAction, outcome: ExecutionOutcome, changed: Optional[bool]
    ) -> Tuple[str, str]:
        """Return (message for the model, short effect label for history)."""

        kind = outcome.kind or action._normalized_type()
        if kind == "run_command":
            output = outcome.output or "(no output)"
            if len(output) > 1800:
                output = output[:900] + "\n…\n" + output[-800:]
            status = "succeeded" if outcome.status == "executed" else "FAILED"
            return f"The command {status}. Output:\n{output}", f"command {status}"
        if not outcome.executed:
            return f"Your action could not be executed: {outcome.detail}", f"not executed ({outcome.detail})"
        if changed is False:
            if kind in {"wait", "move"}:
                return "Nothing changed on screen.", "no visible change"
            return (
                "The screen did NOT change after this action: it probably missed its target or had no "
                "effect. Look carefully at the screenshot and try something different.",
                "NO visible change",
            )
        if changed is None:
            return "Action executed.", "executed"
        return "Action executed and the screen changed. Check on the screenshot that it did what you expected.", "screen changed"

    def _remember(self, step: int, action: AgentAction, effect: str) -> None:
        thought = " ".join((action.thought or "").split())
        if len(thought) > 140:
            thought = thought[:139] + "…"
        line = f"{step}. {action.summary()} → {effect}"
        if thought:
            line += f" (intent: {thought})"
        self._history.append(line)
        self._history = self._history[-15:]
        self._recent_actions.append(f"- {line}"[:400])
        self._recent_actions = self._recent_actions[-8:]

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------
    def _check_safety(self, user_prompt: str, action: AgentAction) -> SafetyDecision:
        mode = (self.config.safety_mode or "balanced").lower()
        kind = action._normalized_type()

        if kind == "run_command" and not self.config.allow_shell:
            return SafetyDecision(False, "Les commandes shell sont désactivées (VISIOS_ALLOW_SHELL=false).")

        if mode in {"off", "fast", "rapide"}:
            return SafetyDecision(True)

        if mode in {"strict", "supervised", "supervisé"}:
            if kind == "run_command":
                return SafetyDecision(True, "Commande shell en mode supervisé.", needs_confirmation=True)
            risky_surface = " ".join(
                value
                for value in (user_prompt, action.text or "", action.app or "")
                if value
            ).lower()
            for keyword in RISK_KEYWORDS:
                if keyword in risky_surface:
                    return SafetyDecision(False, f"Action sensible détectée: {keyword}")

        if action.requires_confirmation:
            return SafetyDecision(
                True if self.confirm else False,
                "Le modèle demande une confirmation humaine avant cette action.",
                needs_confirmation=bool(self.confirm),
            )

        if kind == "run_command" and DANGEROUS_COMMAND.search(action.command or action.text or ""):
            return SafetyDecision(
                True if self.confirm else False,
                "Commande potentiellement destructrice.",
                needs_confirmation=bool(self.confirm),
            )

        return SafetyDecision(True)

    def _ask_confirmation(self, step: int, action: AgentAction, reason: str) -> bool:
        if not self.confirm:
            return False
        self._phase("waiting_user", "Attend ta confirmation", step)
        self._emit("confirmation_required", {"step": step, "reason": reason, "action": self._action_payload(action)})
        approved = bool(self.confirm(action))
        self._emit("confirmation_resolved", {"step": step, "approved": approved})
        self._check_cancel()
        return approved

    # ------------------------------------------------------------------
    # Events & records
    # ------------------------------------------------------------------
    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.on_event:
            return
        try:
            self.on_event(event_type, {"run_id": self.run_id, **payload})
        except Exception:  # pragma: no cover - a UI failure must not kill the run
            pass

    def _phase(self, phase: str, label: str, step: Optional[int] = None) -> None:
        self._emit("phase", {"phase": phase, "label": label, "step": step})

    def _finish_step(
        self,
        recorder: TrajectoryRecorder,
        run_id: str,
        step: int,
        action: AgentAction,
        screenshot: bytes,
        screen_size: Tuple[int, int],
        desktop_context: str,
        status: str,
        detail: str,
        started_at: str,
        started_monotonic: float,
        executed: bool = False,
        pointer: Optional[Tuple[int, int]] = None,
        changed: Optional[bool] = None,
    ) -> AgentStepRecord:
        record = self._record(
            run_id, step, action, screenshot, screen_size, desktop_context, status, detail,
            started_at, started_monotonic, executed=executed, pointer=pointer,
        )
        record.screen_changed = changed
        record.screenshot_path = recorder.write_step(record)
        if self.on_step:
            self.on_step(record)
        self._emit(
            "step_finished",
            {
                "step": step,
                "status": status,
                "detail": detail,
                "executed": executed,
                "screen_changed": changed,
                "duration_ms": record.duration_ms,
            },
        )
        return record

    def _record(
        self,
        run_id: str,
        step: int,
        action: AgentAction,
        screenshot: bytes,
        screen_size: Tuple[int, int],
        desktop_context: str,
        status: str,
        detail: str,
        started_at: str,
        started_monotonic: float,
        executed: bool = False,
        pointer: Optional[Tuple[int, int]] = None,
    ) -> AgentStepRecord:
        return AgentStepRecord(
            run_id=run_id,
            step=step,
            action=action,
            screenshot=screenshot,
            screen_size=screen_size,
            desktop_context=desktop_context,
            status=status,
            detail=detail,
            executed=executed,
            pointer=pointer,
            started_at=started_at,
            duration_ms=int((time.monotonic() - started_monotonic) * 1000),
        )

    @staticmethod
    def _action_payload(action: AgentAction, screen_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": action._normalized_type(),
            "summary": action.summary(),
            "thought": action.thought,
            "observation": action.observation,
            "reasoning": action.reasoning,
            "thinking": AgentRunner._clip_thinking(action.thinking),
            "text": action.text,
            "url": action.url,
            "command": action.command,
            "app": action.app,
            "requires_confirmation": action.requires_confirmation,
            "raw": action.raw_response[:2000] if action.raw_response else "",
        }
        # Fractions of the screen let the UI place markers without knowing the resolution.
        if screen_size and action.coords_absolute and screen_size[0] and screen_size[1]:
            width, height = screen_size
            if action.x is not None and action.y is not None:
                payload["point"] = [action.x / width, action.y / height]
            if action.end_x is not None and action.end_y is not None:
                payload["end_point"] = [action.end_x / width, action.end_y / height]
        return payload

    @staticmethod
    def _clip_thinking(text: Optional[str], limit: int = 20000) -> Optional[str]:
        if not text or len(text) <= limit:
            return text
        return "…" + text[-limit:]

    @staticmethod
    def _thumbnail(screenshot: bytes, width: int = 480) -> Optional[str]:
        if Image is None or not screenshot:
            return None
        try:
            with Image.open(io.BytesIO(screenshot)) as img:
                img = img.convert("RGB")
                ratio = width / float(img.size[0])
                img = img.resize((width, max(1, int(img.size[1] * ratio))), Image.BILINEAR)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=70)
            return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_cancel(self) -> None:
        if self._stop.is_set():
            raise RunCancelled()

    def _sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if self._stop.wait(seconds):
            raise RunCancelled()

    def _recent_action_summary(self) -> str:
        if not self._recent_actions:
            return ""
        return "\n".join(self._recent_actions[-5:])

    def _remember_action(self, action: AgentAction, status: str, detail: str) -> None:
        summary = f"- {action._normalized_type()} status={status} detail={detail}"
        self._recent_actions.append(summary[:400])
        self._recent_actions = self._recent_actions[-8:]

    @staticmethod
    def _action_signature(action: AgentAction) -> Tuple[object, ...]:
        def bucket(value: Optional[float]) -> Optional[int]:
            # ~1% of a 1600px screen: near-identical clicks count as the same action.
            return int(round(value / 16.0)) if value is not None else None

        return (
            action._normalized_type(),
            bucket(action.x),
            bucket(action.y),
            bucket(action.end_x),
            bucket(action.end_y),
            (action.text or "").strip(),
            (action.app or "").strip(),
            (action.url or "").strip(),
            (action.command or "").strip(),
        )

    @staticmethod
    def _new_run_id() -> str:
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}-{uuid.uuid4().hex[:8]}"
