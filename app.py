"""Streamlit MVP for visiOS."""
from __future__ import annotations

import datetime as dt
import html
import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import List, Optional, Tuple, Union

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from backend.agent_runner import AgentRunConfig, AgentRunResult, AgentRunner, AgentStepRecord
from backend.container_manager import SandboxManager
from backend.vision_agent import AgentAction, VisionAgent


def _coerce_length(value: Optional[str], default: str = "75vh") -> str:
    if value is None:
        return default
    trimmed = value.strip()
    if not trimmed:
        return default
    if trimmed.replace(".", "", 1).isdigit():
        return f"{trimmed}px"
    lowered = trimmed.lower()
    for suffix in ("vh", "vw", "%", "px", "rem", "em"):
        if lowered.endswith(suffix):
            return trimmed
    return f"{trimmed}px"


st.set_page_config(page_title="VisiOS", layout="wide", initial_sidebar_state="collapsed")

DEFAULT_VNC_URL = "http://localhost:6901/vnc_lite.html"
VNC_IFRAME_URL = os.getenv("VISIOS_IFRAME_URL", DEFAULT_VNC_URL)
VNC_IFRAME_HEIGHT = _coerce_length(os.getenv("VISIOS_IFRAME_HEIGHT"), "78vh")
VNC_PASSWORD = os.getenv("VISIOS_VNC_PASSWORD", os.getenv("VNC_PW", "password"))
MAX_AGENT_STEPS = int(os.getenv("VISIOS_AGENT_MAX_STEPS", "3"))
ACTION_DELAY_SECONDS = float(os.getenv("VISIOS_ACTION_DELAY", "1.0"))
HEAVY_ACTION_DELAY_SECONDS = float(os.getenv("VISIOS_HEAVY_ACTION_DELAY", "3.0"))
REPEAT_TOLERANCE = int(os.getenv("VISIOS_REPEAT_TOLERANCE", "1"))
SAFETY_MODE = os.getenv("VISIOS_SAFETY_MODE", "balanced")
TRAJECTORY_DIR = os.getenv("VISIOS_TRAJECTORY_DIR", "trajectories")
SAVE_SCREENSHOTS = os.getenv("VISIOS_SAVE_SCREENSHOTS", "true").lower() in {"1", "true", "yes", "on", "oui"}

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


def _build_vnc_url(raw_url: str) -> str:
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


def _inject_branding() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {{
            --app-bg: #080a0f;
            --surface: #10141c;
            --surface-2: #161b24;
            --surface-3: #1d2430;
            --line: #2b3443;
            --line-soft: rgba(255,255,255,0.08);
            --text: #eef3fb;
            --text-soft: #a6b0c2;
            --text-faint: #697386;
            --accent: #ff5a52;
            --accent-2: #4aa3ff;
            --ok: #41c783;
            --warn: #e2b84f;
            --bad: #f06d6d;
        }}

        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background: var(--app-bg);
            color: var(--text);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        [data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {{
            display: none !important;
        }}

        .main .block-container {{
            max-width: 100%;
            padding: 0.85rem 1rem 1rem;
        }}

        div[data-testid="stVerticalBlock"] {{
            gap: 0.75rem;
        }}

        .visi-shell {{
            min-height: calc(100vh - 2rem);
        }}

        .visi-topbar {{
            display: grid;
            grid-template-columns: minmax(220px, 1fr) auto;
            align-items: center;
            gap: 1rem;
            padding: 0.72rem 0.9rem;
            background: linear-gradient(180deg, rgba(29,36,48,0.92), rgba(16,20,28,0.92));
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.25);
        }}

        .visi-brand {{
            display: flex;
            align-items: center;
            gap: 0.7rem;
            min-width: 0;
        }}

        .visi-mark {{
            width: 30px;
            height: 30px;
            display: grid;
            place-items: center;
            border-radius: 7px;
            background: #f3f6fb;
            color: #080a0f;
            font-weight: 700;
            letter-spacing: 0;
        }}

        .visi-title {{
            margin: 0;
            font-size: 1.04rem;
            font-weight: 600;
            letter-spacing: 0;
            line-height: 1.1;
        }}

        .visi-subtitle {{
            margin: 0;
            color: var(--text-soft);
            font-size: 0.78rem;
        }}

        .visi-topmeta {{
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        .visi-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            min-height: 30px;
            padding: 0 0.7rem;
            border: 1px solid var(--line);
            border-radius: 6px;
            background: rgba(255,255,255,0.035);
            color: var(--text-soft);
            font-size: 0.78rem;
            white-space: nowrap;
        }}

        .status-dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-faint);
            flex: 0 0 auto;
        }}

        .status-success {{ background: var(--ok); box-shadow: 0 0 12px rgba(65,199,131,0.4); }}
        .status-warning {{ background: var(--warn); }}
        .status-danger {{ background: var(--bad); }}
        .status-idle {{ background: var(--text-faint); }}

        .visi-main-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(330px, 380px);
            gap: 0.85rem;
            align-items: stretch;
        }}

        .visi-viewer-container {{
            background: #020306;
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            min-height: 520px;
            box-shadow: 0 22px 70px rgba(0,0,0,0.38);
        }}

        .visi-viewer-frame {{
            display: block;
            width: 100%;
            height: 78vh;
            min-height: 520px;
            border: 0;
            background: #020306;
        }}

        .visi-command-card {{
            border: 1px solid var(--line);
            border-radius: 6px;
            background: var(--surface);
            padding: 0.72rem;
        }}

        .visi-side-card {{
            height: 100%;
            min-height: 520px;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.8rem;
            overflow: hidden;
        }}

        .visi-side-title {{
            margin: 0 0 0.7rem;
            font-size: 0.86rem;
            font-weight: 600;
            color: var(--text);
        }}

        .visi-status-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.55rem;
            margin-bottom: 0.75rem;
        }}

        .visi-metric {{
            min-width: 0;
            padding: 0.62rem;
            border-radius: 6px;
            background: var(--surface-2);
            border: 1px solid var(--line-soft);
        }}

        .visi-metric-label {{
            color: var(--text-faint);
            font-size: 0.66rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }}

        .visi-metric-value {{
            color: var(--text);
            font-size: 0.84rem;
            font-weight: 600;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .visi-metric-sub {{
            color: var(--text-soft);
            font-size: 0.7rem;
            margin-top: 0.18rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .visi-timeline {{
            margin-top: 0.65rem;
            max-height: 48vh;
            overflow-y: auto;
        }}

        .visi-timeline-entry {{
            display: grid;
            grid-template-columns: 48px minmax(0, 1fr);
            gap: 0.65rem;
            padding: 0.72rem 0;
            border-bottom: 1px solid var(--line-soft);
        }}

        .visi-timeline-step {{
            color: var(--accent-2);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
        }}

        .visi-timeline-content h4 {{
            margin: 0;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text);
        }}

        .visi-timeline-content p {{
            margin: 0.2rem 0 0;
            font-size: 0.75rem;
            color: var(--text-soft);
            line-height: 1.35;
            word-break: break-word;
        }}

        .visi-empty {{
            margin-top: 0.75rem;
            padding: 0.8rem;
            border: 1px solid var(--line-soft);
            border-radius: 6px;
            color: var(--text-soft);
            background: var(--surface-2);
            font-size: 0.8rem;
        }}

        .visi-log-box {{
            background: #080b10;
            border: 1px solid var(--line-soft);
            border-radius: 6px;
            padding: 0.7rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-soft);
            height: 48vh;
            overflow-y: auto;
            line-height: 1.45;
        }}

        .visi-log-line {{
            padding: 0.24rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.045);
        }}

        .stButton button {{
            width: 100%;
            border-radius: 6px;
            border: 1px solid var(--line) !important;
            background: var(--surface-2) !important;
            color: var(--text) !important;
            min-height: 2.35rem;
            font-weight: 500;
        }}

        .stButton button:hover {{
            border-color: var(--accent) !important;
            color: white !important;
        }}

        div[data-baseweb="tab-list"] {{
            gap: 0.35rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 0.65rem;
        }}

        button[data-baseweb="tab"] {{
            padding: 0.55rem 0.45rem;
            color: var(--text-soft);
            font-size: 0.78rem;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--accent);
        }}

        div[data-baseweb="input"], div[data-baseweb="textarea"], textarea, input {{
            background: var(--surface-2) !important;
            border-color: var(--line) !important;
            color: var(--text) !important;
        }}

        .stTextArea textarea {{
            min-height: 74px !important;
            border-radius: 6px;
        }}

        .stAlert {{
            background: var(--surface-2);
            border-color: var(--line);
        }}

        [data-testid="stImage"] img {{
            border-radius: 6px;
            border: 1px solid var(--line-soft);
        }}

        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: #080a0f;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #3b4656;
            border-radius: 8px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #59667a;
        }}

        @media (max-width: 980px) {{
            .visi-main-grid {{
                grid-template-columns: 1fr;
            }}
            .visi-side-card {{
                min-height: auto;
            }}
            .visi-viewer-frame {{
                height: 62vh;
            }}
            .visi-topbar {{
                grid-template-columns: 1fr;
            }}
            .visi-topmeta {{
                justify-content: flex-start;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _status_dot_class(status: str) -> str:
    return {
        "success": "status-success",
        "warn": "status-warning",
        "danger": "status-danger",
        "idle": "status-idle",
    }.get(status, "status-idle")


def _status_chip(label: str, value: str, status: str = "idle") -> str:
    dot_class = _status_dot_class(status)
    return (
        "<span class='visi-chip'>"
        f"<span class='status-dot {dot_class}'></span>"
        f"{html.escape(label)}: <strong>{html.escape(value)}</strong>"
        "</span>"
    )


def _metric_card(label: str, value: str, sub: str = "", status: str = "idle") -> str:
    dot_class = _status_dot_class(status)
    return (
        "<div class='visi-metric'>"
        f"<div class='visi-metric-label'>{html.escape(label)}</div>"
        f"<div class='visi-metric-value'><span class='status-dot {dot_class}'></span> {html.escape(value)}</div>"
        f"<div class='visi-metric-sub'>{html.escape(sub)}</div>"
        "</div>"
    )


def _stat_card(label: str, value: str, sub: str = "", status: str = "idle") -> str:
    dot_class = {
        "success": "status-success",
        "warn": "status-warning",
        "danger": "status-danger",
        "idle": "status-idle",
    }.get(status, "status-idle")
    
    return (
        f"<div class='visi-stat-item'>"
        f"<div class='visi-stat-label'>{label}</div>"
        f"<div class='visi-stat-value'>"
        f"<span class='status-dot {dot_class}'></span>{value}"
        f"</div>"
        f"<div class='visi-stat-sub'>{sub}</div>"
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
        st.markdown("<div class='visi-empty'>Aucune action récente.</div>", unsafe_allow_html=True)
        return
    
    html_content = "<div class='visi-timeline'>"
    for entry in reversed(entries):
        action = entry["action"]
        text = _display_text(action.text).strip()
        status = entry.get("status", "pending")
        detail = entry.get("detail", "")
        coords = (
            f"({round(action.x, 1)}, {round(action.y, 1)})"
            if action.x is not None and action.y is not None
            else ""
        )
        target = (
            f" → ({round(action.end_x, 1)}, {round(action.end_y, 1)})"
            if action.end_x is not None and action.end_y is not None
            else ""
        )
        
        html_content += (
            f"<div class='visi-timeline-entry'>"
            f"<div class='visi-timeline-step'>STEP {entry['step']}</div>"
            f"<div class='visi-timeline-content'>"
            f"<h4>{html.escape(status.upper())} · {action.action_type.upper()} {coords}{target}</h4>"
            f"<p>{html.escape(text) or '—'}</p>"
            f"{f'<p>{html.escape(detail)}</p>' if detail else ''}"
            f"{f'<p><em>{html.escape(action.reasoning)}</em></p>' if action.reasoning else ''}"
            f"</div>"
            f"</div>"
        )
    html_content += "</div>"
    st.markdown(html_content, unsafe_allow_html=True)


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
    if "last_run_result" not in st.session_state:
        st.session_state.last_run_result = None
    if "ui_max_steps" not in st.session_state:
        st.session_state.ui_max_steps = MAX_AGENT_STEPS
    if "ui_safety_mode" not in st.session_state:
        st.session_state.ui_safety_mode = SAFETY_MODE
    if "ui_save_screenshots" not in st.session_state:
        st.session_state.ui_save_screenshots = SAVE_SCREENSHOTS


def log(message: str) -> None:
    timestamp = dt.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")


def push_trace(
    step: int,
    action: AgentAction,
    screenshot: Optional[bytes],
    status: str = "pending",
    detail: str = "",
    screenshot_path: Optional[str] = None,
) -> None:
    st.session_state.trace.append(
        {
            "step": step,
            "action": action,
            "screenshot": screenshot,
            "status": status,
            "detail": detail,
            "screenshot_path": screenshot_path,
        }
    )
    st.session_state.trace = st.session_state.trace[-10:]


def _display_text(value: Optional[Union[str, int, float]]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _format_safety_mode(mode: str) -> str:
    labels = {
        "balanced": "Équilibré",
        "strict": "Supervisé",
        "fast": "Rapide",
    }
    return labels.get(mode, mode)


def _build_run_config() -> AgentRunConfig:
    return AgentRunConfig(
        max_steps=int(st.session_state.ui_max_steps),
        action_delay_seconds=ACTION_DELAY_SECONDS,
        heavy_action_delay_seconds=HEAVY_ACTION_DELAY_SECONDS,
        repeat_tolerance=REPEAT_TOLERANCE,
        safety_mode=st.session_state.ui_safety_mode,
        trajectory_dir=TRAJECTORY_DIR,
        save_screenshots=bool(st.session_state.ui_save_screenshots),
    )


def _handle_runner_step(record: AgentStepRecord) -> None:
    st.session_state.latest_screenshot = record.screenshot
    st.session_state.screen_size = record.screen_size
    st.session_state.last_capture_ts = dt.datetime.now()
    st.session_state.agent_steps = record.step
    if record.pointer:
        st.session_state.pointer = record.pointer
    st.session_state.last_action_label = f"{record.action.action_type.upper()} · {record.status.upper()}"
    st.session_state.last_reasoning = record.action.reasoning or record.detail
    push_trace(
        record.step,
        record.action,
        record.screenshot,
        status=record.status,
        detail=record.detail,
        screenshot_path=record.screenshot_path,
    )


_init_state()

sandbox: SandboxManager = st.session_state.sandbox
vision: VisionAgent = st.session_state.vision

sandbox_label, sandbox_tone = _sandbox_state_label(sandbox)
screen_label = (
    f"{st.session_state.screen_size[0]}×{st.session_state.screen_size[1]}"
    if st.session_state.screen_size
    else "—"
)
pointer_value = (
    f"{int(st.session_state.pointer[0])}, {int(st.session_state.pointer[1])}"
    if st.session_state.pointer
    else "—"
)
reasoning_snippet = _truncate(st.session_state.last_reasoning)
action_tone = "success" if st.session_state.last_action_label != "IDLE" else "idle"
vnc_url = _build_vnc_url(VNC_IFRAME_URL)
user_prompt = None

st.markdown(
    f"""
    <div class="visi-topbar">
        <div class="visi-brand">
            <div class="visi-mark">V</div>
            <div>
                <h1 class="visi-title">VisiOS</h1>
                <p class="visi-subtitle">Computer-use sandbox · prompt-first operating system</p>
            </div>
        </div>
        <div class="visi-topmeta">
            {_status_chip("Sandbox", sandbox_label, sandbox_tone)}
            {_status_chip("Action", st.session_state.last_action_label, action_tone)}
            {_status_chip("Écran", screen_label, "idle")}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Main Content ---
main_col, side_col = st.columns([4.8, 1.55], gap="medium")

with main_col:
    st.markdown(
        f"""
        <div class='visi-viewer-container'>
            <iframe
                src='{html.escape(vnc_url)}'
                title='VisiOS Remote Desktop'
                class='visi-viewer-frame'
                style='height:{VNC_IFRAME_HEIGHT};'
                frameborder='0'
                allowfullscreen
            ></iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='visi-command-card'>", unsafe_allow_html=True)
    with st.form("agent-command-form", clear_on_submit=True):
        prompt_value = st.text_area(
            "Instruction",
            placeholder="Demande à VisiOS d'agir dans l'OS...",
            label_visibility="collapsed",
        )
        form_cols = st.columns([1, 1, 5])
        with form_cols[0]:
            submitted = st.form_submit_button("Exécuter", use_container_width=True)
        with form_cols[1]:
            capture_requested = st.form_submit_button("Observer", use_container_width=True)
        if submitted and prompt_value.strip():
            user_prompt = prompt_value.strip()
        elif capture_requested:
            try:
                sandbox.start_container()
                screenshot, screen_size = sandbox.get_screenshot()
                st.session_state.latest_screenshot = screenshot
                st.session_state.screen_size = screen_size
                st.session_state.last_capture_ts = dt.datetime.now()
                log("Observation manuelle capturée")
                st.rerun()
            except Exception as exc:
                log(f"Observation impossible: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.markdown("<div class='visi-side-card'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="visi-status-grid">
            {_metric_card("Sandbox", sandbox_label, f"Port {sandbox.host_vnc_port}", sandbox_tone)}
            {_metric_card("Dernière action", st.session_state.last_action_label, reasoning_snippet, action_tone)}
            {_metric_card("Curseur", pointer_value, "Position", "success" if st.session_state.pointer else "idle")}
            {_metric_card("Affichage", screen_label, _format_timestamp(st.session_state.last_capture_ts), "idle")}
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab1, tab2, tab3 = st.tabs(["Timeline", "Logs", "Pilotage"])
    
    with tab1:
        _render_timeline(st.session_state.trace)
        if st.session_state.latest_screenshot:
            st.image(st.session_state.latest_screenshot, caption="Dernier instantané", width="stretch")
            
    with tab2:
        log_lines = st.session_state.logs[-100:]
        if log_lines:
            formatted = "".join(f"<div class='visi-log-line'>{html.escape(line)}</div>" for line in log_lines)
        else:
            formatted = "<div class='visi-log-line'>Prêt.</div>"
        st.markdown(f"<div class='visi-log-box'>{formatted}</div>", unsafe_allow_html=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Démarrer", use_container_width=True):
                sandbox.start_container()
                log("Sandbox started")
                st.rerun()
        with c2:
            if st.button("Arrêter", use_container_width=True):
                sandbox.stop_container()
                log("Sandbox stopped")
                st.rerun()
        if st.button("Réinitialiser l'interface", use_container_width=True):
            st.session_state.logs = []
            st.session_state.trace = []
            st.session_state.latest_screenshot = None
            st.session_state.pointer = None
            st.session_state.last_action_label = "IDLE"
            st.session_state.agent_steps = 0
            st.session_state.last_reasoning = ""
            st.session_state.last_capture_ts = None
            st.session_state.last_run_result = None
            log("Interface réinitialisée")
            st.rerun()
        st.radio(
            "Mode",
            options=["balanced", "strict", "fast"],
            format_func=_format_safety_mode,
            key="ui_safety_mode",
            horizontal=True,
        )
        st.slider("Étapes", min_value=1, max_value=30, key="ui_max_steps")
        st.checkbox("Captures", key="ui_save_screenshots")
        last_result: Optional[AgentRunResult] = st.session_state.last_run_result
        if last_result:
            st.caption(f"Run {last_result.run_id}")
            st.write(f"{last_result.status}: {last_result.reason}")
            if last_result.trajectory_path:
                st.code(last_result.trajectory_path, language="text")
    st.markdown("</div>", unsafe_allow_html=True)


# --- Agent Logic ---
if user_prompt:
    log(f"User: {user_prompt}")
    runner = AgentRunner(
        sandbox=sandbox,
        vision=vision,
        config=_build_run_config(),
        log=log,
        on_step=_handle_runner_step,
    )
    try:
        st.session_state.last_run_result = runner.run(user_prompt)
    except Exception as exc:
        message = f"Run interrompu: {exc}"
        log(message)
        st.session_state.last_action_label = "ERREUR"
        st.session_state.last_reasoning = message
        st.session_state.last_run_result = AgentRunResult(
            run_id=dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
            status="failed",
            steps=st.session_state.agent_steps,
            reason=message,
        )
    st.rerun()
