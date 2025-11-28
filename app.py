"""Streamlit MVP for visiOS."""
from __future__ import annotations

import datetime as dt
import os
import time
from typing import List, Optional

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


_init_state()

sandbox: SandboxManager = st.session_state.sandbox
vision: VisionAgent = st.session_state.vision

col_start, col_stop = st.columns(2)
if col_start.button("Start Sandbox", use_container_width=True):
    sandbox.start_container()
    log("Sandbox started")
if col_stop.button("Stop Sandbox", use_container_width=True):
    sandbox.stop_container()
    log("Sandbox stopped")

col1, col2 = st.columns([7, 3])
with col1:
    st.markdown(
        f"<iframe src='{VNC_IFRAME_URL}' width='100%' height='700px' frameborder='0'></iframe>",
        unsafe_allow_html=True,
    )
    if st.session_state.latest_screenshot:
        st.image(st.session_state.latest_screenshot, caption="Latest screenshot", use_column_width=True)

with col2:
    st.subheader("Agent Console")
    st.write("Chat with visiOS to drive the remote desktop.")
    logs_text = "\n".join(st.session_state.logs[-50:])
    st.text_area("Execution Log", value=logs_text, height=400)

    st.subheader("Last actions")
    for entry in reversed(st.session_state.trace):
        action = entry["action"]
        st.markdown(
            f"**Step {entry['step']}** — `{action.action_type}` "
            f"{(action.x, action.y) if action.x is not None and action.y is not None else ''} "
            f"{action.text or ''}"
        )
        if entry["screenshot"]:
            st.image(entry["screenshot"], use_column_width=True)

user_prompt = st.chat_input("Décrivez l'action désirée")

if user_prompt:
    log(f"User: {user_prompt}")
    sandbox.start_container()
    last_signature = None
    repeat_count = 0

    for step in range(1, MAX_AGENT_STEPS + 1):
        log(f"Step {step}/{MAX_AGENT_STEPS}: capturing screenshot")
        screenshot, _size = sandbox.get_screenshot()
        st.session_state.latest_screenshot = screenshot

        action: AgentAction = vision.decide_action(screenshot, user_prompt)
        log(f"Model (step {step}): {action.raw_response}")
        push_trace(step, action, screenshot)

        signature = (
            action.action_type.lower(),
            round(action.x, 2) if action.x is not None else None,
            round(action.y, 2) if action.y is not None else None,
            action.text.strip() if action.text else "",
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

        action_kind = (action.action_type or "").strip().lower()
        executed = False
        if action_kind == "click" and action.x is not None and action.y is not None:
            sandbox.send_click(action.x, action.y)
            log(f"Click @ ({action.x}, {action.y})")
            executed = True
        elif action_kind == "type" and action.text:
            if action.x is not None and action.y is not None:
                sandbox.send_click(action.x, action.y)
                log(f"(focus) Click @ ({action.x}, {action.y}) before typing")
                time.sleep(0.3)
            sandbox.send_text(action.text)
            log(f"Typed: {action.text}")
            executed = True
        elif action_kind == "scroll":
            if sandbox.send_scroll(action.y, action.text):
                log(f"Scroll request value={action.y} hint={action.text}")
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
