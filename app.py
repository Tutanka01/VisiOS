"""Streamlit MVP for visiOS."""
from __future__ import annotations

import datetime as dt
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
        st.session_state.pointer: Optional[Tuple[int, int]] = None
    if "screen_size" not in st.session_state:
        st.session_state.screen_size: Optional[Tuple[int, int]] = None


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
            f"{_display_text(action.text)}"
        )
        if action.reasoning:
            st.caption(action.reasoning)
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
        screenshot, screen_size = sandbox.get_screenshot()
        st.session_state.latest_screenshot = screenshot
        st.session_state.screen_size = screen_size

        action: AgentAction = vision.decide_action(screenshot, user_prompt, screen_size=screen_size)
        if action.reasoning:
            log(f"Model reasoning: {action.reasoning}")
        log(f"Model (step {step}): {action.raw_response}")
        push_trace(step, action, screenshot)

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
