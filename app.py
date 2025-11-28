"""Streamlit MVP for visiOS."""
from __future__ import annotations

import datetime as dt
import os
from typing import List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from backend.container_manager import SandboxManager
from backend.vision_agent import AgentAction, VisionAgent

st.set_page_config(page_title="visiOS - Local Agent", layout="wide")

VNC_IFRAME_URL = os.getenv("VISIOS_IFRAME_URL", "http://localhost:6901/vnc.html")


def _init_state() -> None:
    if "sandbox" not in st.session_state:
        st.session_state.sandbox = SandboxManager()
    if "vision" not in st.session_state:
        st.session_state.vision = VisionAgent()
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "latest_screenshot" not in st.session_state:
        st.session_state.latest_screenshot = None


def log(message: str) -> None:
    timestamp = dt.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")


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

user_prompt = st.chat_input("Décrivez l'action désirée")

if user_prompt:
    log(f"User: {user_prompt}")
    sandbox.start_container()
    screenshot, _size = sandbox.get_screenshot()
    st.session_state.latest_screenshot = screenshot
    action: AgentAction = vision.decide_action(screenshot, user_prompt)
    log(f"Model: {action.raw_response}")

    if action.is_click() and action.x is not None and action.y is not None:
        sandbox.send_click(action.x, action.y)
        log(f"Click @ ({action.x}, {action.y})")
    elif action.is_type() and action.text:
        sandbox.send_text(action.text)
        log(f"Typed: {action.text}")
    else:
        log("No actionable command returned.")

    st.rerun()
