"""Vision agent that calls Ollama to interpret UI screenshots."""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import os

import requests

LOGGER = logging.getLogger(__name__)


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


DEFAULT_BASE_URL = os.getenv("VISIOS_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("VISIOS_OLLAMA_MODEL", "qwen2.5-vl")
DEFAULT_TEMPERATURE = _env_float("VISIOS_OLLAMA_TEMPERATURE", 0.1)
DEFAULT_TIMEOUT = _env_int("VISIOS_OLLAMA_TIMEOUT", 120)


@dataclass
class AgentAction:
    """Structured decision returned by the vision model."""

    action_type: str
    x: Optional[float] = None
    y: Optional[float] = None
    text: Optional[str] = None
    raw_response: str = ""

    def is_click(self) -> bool:
        return self.action_type.lower() == "click"

    def is_type(self) -> bool:
        return self.action_type.lower() == "type"


class VisionAgent:
    """Thin wrapper around the Ollama HTTP API for multimodal reasoning."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def decide_action(self, screenshot: bytes, user_prompt: str) -> AgentAction:
        """Send the screenshot + prompt to Ollama and parse the textual response."""

        LOGGER.info("Querying Ollama with %s bytes of image data", len(screenshot))
        encoded_image = base64.b64encode(screenshot).decode("ascii")
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": self._build_prompt(user_prompt),
            "images": [encoded_image],
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        url = f"{self.base_url}/api/generate"
        response = requests.post(url, json=payload, stream=True, timeout=self.timeout)
        response.raise_for_status()

        collected = []
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            collected.append(chunk.get("response", ""))
            if chunk.get("done"):
                break
        full_text = "".join(collected).strip()
        LOGGER.debug("Ollama response: %s", full_text)
        return self._parse_action(full_text)

    # ------------------------------------------------------------------
    def _build_prompt(self, user_prompt: str) -> str:
        template = (
            "You are visiOS, an accessibility agent.\n"
            "Given the attached desktop screenshot, decide the single next UI action "
            "needed to satisfy the user's request.\n"
            "Respond using JSON with the following schema: {" \
            "\"action\": \"click|type|noop\", " \
            "\"coordinates\": {\"x\": <number>, \"y\": <number>}, " \
            "\"text\": \"optional text to type\" }.\n"
            "Coordinates must be either pixels or within 0-1000.\n"
            "User request: {prompt}\n"
        )
        return template.format(prompt=user_prompt)

    def _parse_action(self, model_text: str) -> AgentAction:
        """Best-effort extraction of coordinates/text from model output."""

        action = AgentAction(action_type="noop", raw_response=model_text)

        json_blob = self._extract_json(model_text)
        if json_blob is not None:
            action = self._from_json(json_blob, model_text)
        else:
            match = re.search(r"click.*?(\d+)\D+(\d+)", model_text, re.IGNORECASE)
            if match:
                action = AgentAction(action_type="click", x=float(match.group(1)), y=float(match.group(2)), raw_response=model_text)
            elif "type" in model_text.lower():
                text_match = re.search(r"type[:\s]+(.+)$", model_text, re.IGNORECASE)
                action = AgentAction(action_type="type", text=text_match.group(1).strip() if text_match else "", raw_response=model_text)
        return action

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        candidate = None
        for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
            candidate = match.group(0)
        if candidate is None:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _from_json(payload: Dict[str, Any], raw_text: str) -> AgentAction:
        action = payload.get("action") or payload.get("type") or "noop"
        coords = payload.get("coordinates") or {}
        text = payload.get("text") or payload.get("value")
        x = coords.get("x") if isinstance(coords, dict) else payload.get("x")
        y = coords.get("y") if isinstance(coords, dict) else payload.get("y")
        return AgentAction(action_type=str(action), x=x, y=y, text=text, raw_response=raw_text)
