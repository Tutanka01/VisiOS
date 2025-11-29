"""Streamlit MVP for visiOS."""
from __future__ import annotations

import datetime as dt
import html
import os
import re
import time
from typing import List, Optional, Tuple, Union

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from backend.container_manager import SandboxManager
from backend.vision_agent import AgentAction, VisionAgent

st.set_page_config(page_title="visiOS - Local Agent", layout="wide")

VNC_IFRAME_URL = os.getenv("VISIOS_IFRAME_URL", "http://localhost:6901/vnc.html")
MAX_AGENT_STEPS = int(os.getenv("VISIOS_AGENT_MAX_STEPS", "3"))
ACTION_DELAY_SECONDS = float(os.getenv("VISIOS_ACTION_DELAY", "1.0"))
REPEAT_TOLERANCE = int(os.getenv("VISIOS_REPEAT_TOLERANCE", "1"))

PALETTE = {
    "bg": "#0f1117",
    "panel": "#171b24",
    "card": "#1f2430",
    "accent": "#4c8bf5",
    "accent_soft": "#9dbdff",
    "success": "#43c383",
    "warning": "#f5c26b",
    "danger": "#f07178",
    "muted": "#8a92a6",
}


def _inject_branding() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        html, body, [data-testid="stAppViewContainer"] {{
            background: {PALETTE["bg"]};
            color: #f4f5fb;
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        [data-testid="stAppViewContainer"] > .main {{
            padding-top: 0.1rem;
            padding-bottom: 1.25rem;
        }}
        section.main > div.block-container {{
            padding-top: 0rem;
            margin-top: -0.35rem;
        }}
        [data-testid="stHeader"] {{
            background: {PALETTE["bg"]};
            border-bottom: 1px solid rgba(255,255,255,0.03);
        }}
        .visi-headline {{
            padding: 12px 4px 6px;
        }}
        .visi-headline h1 {{
            margin: 0;
            font-size: 1.9rem;
            font-weight: 600;
            color: #f4f5fb;
        }}
        .visi-headline p {{
            margin: 4px 0 0;
            color: {PALETTE["muted"]};
            font-size: 0.95rem;
        }}
        .visi-stat {{
            background: {PALETTE["card"]};
            border-radius: 12px;
            padding: 14px 16px;
            border: 1px solid rgba(255,255,255,0.03);
            min-height: 100px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .visi-stat span {{ display: block; }}
        .visi-stat .label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {PALETTE["muted"]};
        }}
        .visi-stat .value {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 6px;
        }}
        .visi-stat .sub {{
            font-size: 0.9rem;
            color: rgba(255,255,255,0.6);
        }}
        .visi-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .visi-pill.success {{ background-color: rgba(67,195,131,0.15); color: {PALETTE["success"]}; }}
        .visi-pill.warn {{ background-color: rgba(245,194,107,0.16); color: {PALETTE["warning"]}; }}
        .visi-pill.idle {{ background-color: rgba(255,255,255,0.08); color: #d8dbe6; }}
        .visi-panel {{
            background: {PALETTE["panel"]};
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 14px;
            padding: 20px;
            height: 100%;
        }}
        .visi-frame iframe {{
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.05);
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }}
        .visi-log {{
            background: {PALETTE["card"]};
            border-radius: 10px;
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.03);
            max-height: 360px;
            overflow-y: auto;
            font-family: 'Inter', monospace;
            font-size: 0.8rem;
        }}
        .visi-log-line {{
            margin-bottom: 4px;
            opacity: 0.9;
        }}
        .visi-log-line strong {{ color: {PALETTE["accent_soft"]}; }}
        .visi-timeline {{
            border-left: 1px solid rgba(255,255,255,0.08);
            margin-left: 6px;
            padding-left: 16px;
            margin-bottom: 12px;
        }}
        .visi-timeline-card {{
            background: {PALETTE["panel"]};
            border-radius: 10px;
            padding: 10px 12px;
            border: 1px solid rgba(255,255,255,0.04);
        }}
        .visi-timeline-card h4 {{
            margin: 0;
            font-size: 0.95rem;
        }}
        .visi-timeline-card p {{
            margin: 3px 0;
            color: {PALETTE["muted"]};
        }}
        button[data-baseweb="button"] {{
            border-radius: 6px !important;
            padding: 0.6rem 1rem;
            font-weight: 600;
            background: {PALETTE["card"]};
            border: 1px solid rgba(255,255,255,0.08);
            color: #f4f5fb;
        }}
        button[data-baseweb="button"]:hover {{
            border: 1px solid {PALETTE["accent"]};
            color: {PALETTE["accent"]};
        }}
        .stTabs [data-baseweb="tab"] {{
            border-bottom: 1px solid rgba(255,255,255,0.06);
            padding: 0.6rem 1.2rem;
            color: {PALETTE["muted"]};
        }}
        .stTabs [aria-selected="true"] {{
            color: #f4f5fb;
            border-bottom: 2px solid {PALETTE["accent"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _stat_card(title: str, value: str, subtitle: str = "", tone: str = "idle") -> str:
    pill = f"<span class='visi-pill {tone}'>{subtitle}</span>" if subtitle else ""
    return (
        f"<div class='visi-stat'>"
        f"<span class='label'>{title}</span>"
        f"<span class='value'>{value}</span>"
        f"<span class='sub'>{pill}</span>"
        f"</div>"
    )


def _sandbox_state_label(manager: SandboxManager) -> Tuple[str, str]:
    try:
        container = manager.container or manager.client.containers.get(manager.container_name)
        container.reload()
        status = (container.status or "unknown").lower()
    except Exception:
        status = "stopped"
    tone = "success" if status == "running" else "warn" if status not in {"created", "exited"} else "idle"
    label = "Actif" if status == "running" else "En veille" if status == "exited" else status.title()
    return label, tone


def _format_timestamp(value: Optional[dt.datetime]) -> str:
    if not value:
        return "—"
    return value.strftime("%H:%M:%S")


def _truncate(text: Optional[str], limit: int = 68) -> str:
    if not text:
        return "—"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _render_timeline(entries: List[dict]) -> None:
    if not entries:
        st.caption("Aucune action récente.")
        return
    for entry in reversed(entries):
        action = entry["action"]
        text = _display_text(action.text).strip()
        coords = (
            f"→ ({round(action.x, 1)}, {round(action.y, 1)})"
            if action.x is not None and action.y is not None
            else ""
        )
        st.markdown(
            f"""
            <div class='visi-timeline'>
                <div class='visi-timeline-card'>
                    <h4>Étape {entry['step']} · {action.action_type.upper()}</h4>
                    <p>{html.escape(text) or '—'} {coords}</p>
                    {f"<p><em>{html.escape(action.reasoning)}</em></p>" if action.reasoning else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if entry["screenshot"]:
            st.image(entry["screenshot"], use_column_width=True, caption="Capture associée")


_inject_branding()


def _init_state() -> None:
    if "sandbox" not in st.session_state:
        st.session_state.sandbox = SandboxManager()
    if "vision" not in st.session_state:
        st.session_state.vision = VisionAgent()
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "trace" not in st.session_state:
        st.session_state.trace = []
    if "latest_screenshot" not in st.session_state:
        st.session_state.latest_screenshot = None
    if "pointer" not in st.session_state:
        st.session_state.pointer = None
    if "screen_size" not in st.session_state:
        st.session_state.screen_size = None
    if "last_capture_ts" not in st.session_state:
        st.session_state.last_capture_ts = None
    if "last_action_label" not in st.session_state:
        st.session_state.last_action_label = "IDLE"
    if "agent_steps" not in st.session_state:
        st.session_state.agent_steps = 0
    if "last_reasoning" not in st.session_state:
        st.session_state.last_reasoning = ""


def log(message: str) -> None:
    timestamp = dt.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")


def push_trace(step: int, action: AgentAction, screenshot: Optional[bytes]) -> None:
    st.session_state.trace.append(
        {
            "step": step,
            "action": action,
            "screenshot": screenshot,
        }
    )
    st.session_state.trace = st.session_state.trace[-10:]


def _update_pointer(coords: Optional[Tuple[int, int]]) -> None:
    st.session_state.pointer = coords


def _current_pointer() -> Optional[Tuple[int, int]]:
    return st.session_state.pointer


def _extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _display_text(value: Optional[Union[str, int, float]]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


_init_state()

sandbox: SandboxManager = st.session_state.sandbox
vision: VisionAgent = st.session_state.vision

sandbox_label, sandbox_tone = _sandbox_state_label(sandbox)
screen_label = (
    f"{st.session_state.screen_size[0]}×{st.session_state.screen_size[1]}"
    if st.session_state.screen_size
    else "En attente"
)
pointer_value = (
    f"{int(st.session_state.pointer[0])}, {int(st.session_state.pointer[1])}"
    if st.session_state.pointer
    else "—"
)
reasoning_snippet = _truncate(st.session_state.last_reasoning)
action_tone = "success" if st.session_state.last_action_label != "IDLE" else "idle"

st.markdown(
    """
    <div class="visi-headline">
        <h1>visiOS · Bureau distant</h1>
        <p>Contrôlez votre poste isolé et suivez les décisions de l'agent sans surcouche distrayante.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

action_cols = st.columns([1, 1, 1])
if action_cols[0].button("Démarrer le sandbox", use_container_width=True):
    sandbox.start_container()
    log("Sandbox started")
if action_cols[1].button("Arrêter le sandbox", use_container_width=True):
    sandbox.stop_container()
    log("Sandbox stopped")
if action_cols[2].button("Réinitialiser l'interface", use_container_width=True):
    st.session_state.logs = []
    st.session_state.trace = []
    st.session_state.latest_screenshot = None
    st.session_state.pointer = None
    st.session_state.last_action_label = "IDLE"
    st.session_state.agent_steps = 0
    st.session_state.last_reasoning = ""
    st.session_state.last_capture_ts = None
    log("Interface visiOS réinitialisée")

stats_cols = st.columns(4)
stats_cols[0].markdown(
    _stat_card("Sandbox", sandbox_label, f"Port {sandbox.host_vnc_port}", tone=sandbox_tone),
    unsafe_allow_html=True,
)
stats_cols[1].markdown(
    _stat_card(
        "Dernière capture",
        _format_timestamp(st.session_state.last_capture_ts),
        screen_label,
        tone="idle",
    ),
    unsafe_allow_html=True,
)
stats_cols[2].markdown(
    _stat_card("Action agent", st.session_state.last_action_label, reasoning_snippet, tone=action_tone),
    unsafe_allow_html=True,
)
pointer_tone = "success" if st.session_state.pointer else "idle"
stats_cols[3].markdown(
    _stat_card("Curseur", pointer_value, "point de focus" if st.session_state.pointer else "non défini", tone=pointer_tone),
    unsafe_allow_html=True,
)

main_container = st.container()
col_main, col_side = main_container.columns([2.5, 1.2])

with col_main:
    viewer_tab, screenshot_tab = st.tabs(["Bureau distant", "Instantané"])
    with viewer_tab:
        st.markdown(
            f"""
            <div class='visi-frame'>
                <iframe src='{VNC_IFRAME_URL}' width='100%' height='720px' frameborder='0'></iframe>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with screenshot_tab:
        if st.session_state.latest_screenshot:
            st.image(st.session_state.latest_screenshot, caption="Capture la plus récente", use_column_width=True)
        else:
            st.caption("Aucune capture disponible pour le moment.")

with col_side:
    console_tab, history_tab = st.tabs(["Console", "Timeline"])
    with console_tab:
        log_lines = st.session_state.logs[-80:]
        if log_lines:
            formatted = "".join(
                f"<div class='visi-log-line'>{html.escape(line)}</div>" for line in log_lines
            )
        else:
            formatted = "<div class='visi-log-line'>Aucun log pour l'instant.</div>"
        st.markdown(f"<div class='visi-log'>{formatted}</div>", unsafe_allow_html=True)
    with history_tab:
        _render_timeline(st.session_state.trace)

user_prompt = st.chat_input("Décrivez l'action désirée")

if user_prompt:
    log(f"User: {user_prompt}")
    sandbox.start_container()
    last_signature = None
    repeat_count = 0

    for step in range(1, MAX_AGENT_STEPS + 1):
        log(f"Step {step}/{MAX_AGENT_STEPS}: capturing screenshot")
        screenshot, screen_size = sandbox.get_screenshot()
        st.session_state.latest_screenshot = screenshot
        st.session_state.screen_size = screen_size
        st.session_state.last_capture_ts = dt.datetime.now()
        st.session_state.agent_steps = step

        action: AgentAction = vision.decide_action(screenshot, user_prompt, screen_size=screen_size)
        if action.reasoning:
            log(f"Model reasoning: {action.reasoning}")
        log(f"Model (step {step}): {action.raw_response}")
        push_trace(step, action, screenshot)
        st.session_state.last_action_label = (action.action_type or "?").upper()
        st.session_state.last_reasoning = action.reasoning or ""

        signature = (
            action.action_type.lower(),
            round(action.x, 2) if action.x is not None else None,
            round(action.y, 2) if action.y is not None else None,
            _display_text(action.text).strip(),
        )
        if signature == last_signature:
            repeat_count += 1
            if repeat_count >= REPEAT_TOLERANCE:
                log(
                    f"Model repeated previous instruction {repeat_count + 1} times; stopping to avoid loops."
                )
                break
        else:
            repeat_count = 0

        action_kind = action._normalized_type()
        executed = False
        if action_kind == "click" and action.x is not None and action.y is not None:
            actual_x, actual_y = sandbox.send_click(action.x, action.y)
            _update_pointer((actual_x, actual_y))
            log(f"Click @ ({actual_x}, {actual_y})")
            executed = True
        elif action_kind == "type" and action.text:
            if action.x is not None and action.y is not None:
                focus_x, focus_y = sandbox.send_click(action.x, action.y)
                _update_pointer((focus_x, focus_y))
                log(f"(focus) Click @ ({focus_x}, {focus_y}) before typing")
                time.sleep(0.3)
            sandbox.send_text(action.text)
            log(f"Typed: {action.text}")
            executed = True
        elif action_kind == "scroll":
            hint_chunks = [chunk for chunk in [action.action_type, _display_text(action.text)] if chunk]
            hint = " ".join(hint_chunks) or None
            anchor: Optional[Tuple[int, int]] = None
            pointer_coords = action.pointer()
            current_pointer = _current_pointer()
            if pointer_coords:
                pointer_x, pointer_y = pointer_coords
                anchor = sandbox.move_pointer(pointer_x, pointer_y)
            elif current_pointer:
                pointer_x, pointer_y = current_pointer
                anchor = sandbox.move_pointer(pointer_x, pointer_y, absolute=True)
            elif st.session_state.screen_size:
                width, height = st.session_state.screen_size
                anchor = sandbox.move_pointer(width / 2, height / 2, absolute=True)
            if anchor:
                _update_pointer(anchor)

            scroll_value = action.scroll_amount
            if scroll_value is None and action.x is None and action.y is not None:
                scroll_value = action.y
            if scroll_value is None:
                scroll_value = _extract_number(action.text)

            if sandbox.send_scroll(scroll_value, hint):
                anchor_label = anchor or current_pointer
                log(
                    f"Scroll @ {anchor_label if anchor_label else '(unspecified)'} value={scroll_value} hint={hint}"
                )
                executed = True
            else:
                log("Scroll request ignored (missing magnitude)")
        else:
            log("No actionable command returned; stopping loop.")
            break

        last_signature = signature
        if step < MAX_AGENT_STEPS and ACTION_DELAY_SECONDS > 0:
            time.sleep(ACTION_DELAY_SECONDS)
    else:
        log("Reached maximum agent steps. Provide more guidance or increase VISIOS_AGENT_MAX_STEPS.")

    st.rerun()
