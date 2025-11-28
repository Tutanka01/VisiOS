"""Vision agent that calls Ollama to interpret UI screenshots."""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

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
DEFAULT_API_MODE = os.getenv("VISIOS_API_MODE", "ollama").lower()
DEFAULT_OPENAI_ENDPOINT = os.getenv("VISIOS_OPENAI_ENDPOINT", "responses").lower()

SYSTEM_PROMPT = (
    "You are visiOS, an accessibility agent.\n"
    "Given the attached desktop screenshot, decide the single next UI action "
    "needed to satisfy the user's request.\n"
    "Respond using JSON with the following schema: {{"
    "\"action\": \"click|type|noop\", "
    "\"coordinates\": {{\"x\": <number>, \"y\": <number>}}, "
    "\"text\": \"optional text to type\" }}.\n"
    "Coordinates must be either pixels or within 0-1000.\n"
)


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
        api_mode: str = DEFAULT_API_MODE,
        openai_endpoint: str = DEFAULT_OPENAI_ENDPOINT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.api_mode = api_mode.lower()
        self.openai_endpoint = openai_endpoint.lower()

    def decide_action(self, screenshot: bytes, user_prompt: str) -> AgentAction:
        """Send the screenshot + prompt to Ollama and parse the textual response."""

        LOGGER.info("Querying %s backend with %s bytes of image data", self.api_mode, len(screenshot))
        encoded_image = base64.b64encode(screenshot).decode("ascii")

        if self.api_mode == "openai":
            full_text = self._call_openai_api(encoded_image, user_prompt)
        else:
            full_text = self._call_ollama_api(encoded_image, user_prompt)

        LOGGER.debug("Vision backend response: %s", full_text)
        return self._parse_action(full_text)

    # ------------------------------------------------------------------
    def _build_prompt(self, user_prompt: str) -> str:
        return f"{SYSTEM_PROMPT}User request: {user_prompt}\n"

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

    # ------------------------------------------------------------------
    # Backend adapters
    # ------------------------------------------------------------------
    def _endpoint(self, path: str) -> str:
        cleaned_base = self.base_url.rstrip("/")
        cleaned_path = path.strip("/")
        return f"{cleaned_base}/{cleaned_path}" if cleaned_path else cleaned_base

    def _call_ollama_api(self, encoded_image: str, user_prompt: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": self._build_prompt(user_prompt),
            "images": [encoded_image],
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        url = self._endpoint("api/generate")
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
        return "".join(collected).strip()

    def _call_openai_api(self, encoded_image: str, user_prompt: str) -> str:
        if self.openai_endpoint in {"chat", "chat_completions", "v1/chat/completions"}:
            return self._call_openai_chat(encoded_image, user_prompt)

        if self.openai_endpoint in {"completions", "completion", "v1/completions"}:
            return self._call_openai_completion(encoded_image, user_prompt)

        return self._call_openai_responses(encoded_image, user_prompt)

    def _call_openai_responses(self, encoded_image: str, user_prompt: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT.strip()}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"User request: {user_prompt}"},
                        {"type": "input_image", "image_base64": encoded_image},
                    ],
                },
            ],
            "temperature": self.temperature,
        }
        url = self._endpoint("v1/responses")
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        text = self._extract_openai_text(data)
        if text:
            return text.strip()
        LOGGER.warning("Unable to parse OpenAI responses payload: %s", json.dumps(data)[:500])
        return ""

    def _call_openai_chat(self, encoded_image: str, user_prompt: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT.strip()}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"User request: {user_prompt}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                        },
                    ],
                },
            ],
            "temperature": self.temperature,
        }
        url = self._endpoint("v1/chat/completions")
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        text = self._extract_openai_text(data)
        if text:
            return text.strip()
        LOGGER.warning("Unable to parse OpenAI chat payload: %s", json.dumps(data)[:500])
        return ""

    def _call_openai_completion(self, encoded_image: str, user_prompt: str) -> str:
        prompt = (
            f"{SYSTEM_PROMPT}User request: {user_prompt}\n"
            "Image (base64 PNG) follows between <image></image> tags.\n"
            f"<image>{encoded_image}</image>"
        )
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
        }
        url = self._endpoint("v1/completions")
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        text = self._extract_openai_text(data)
        if text:
            return text.strip()
        LOGGER.warning("Unable to parse OpenAI completions payload: %s", json.dumps(data)[:500])
        return ""

    @staticmethod
    def _extract_openai_text(payload: Dict[str, Any]) -> str:
        def _iter_text(blobs: Iterable[Dict[str, Any]]) -> Iterable[str]:
            for blob in blobs:
                if "content" in blob and isinstance(blob["content"], list):
                    for content in blob["content"]:
                        if isinstance(content, dict):
                            text_value = content.get("text") or content.get("value")
                            if text_value:
                                yield str(text_value)
                else:
                    text_value = blob.get("text") or blob.get("value")
                    if text_value:
                        yield str(text_value)

        if "output" in payload and isinstance(payload["output"], list):
            return "".join(_iter_text(payload["output"]))

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for segment in content:
                    if isinstance(segment, dict):
                        if "text" in segment:
                            parts.append(str(segment["text"]))
                        elif segment.get("type") in {"output_text", "text"} and "value" in segment:
                            parts.append(str(segment["value"]))
                if parts:
                    return "".join(parts)
            if isinstance(content, str):
                return content
            if "text" in choice:
                return str(choice["text"])

        if "output_text" in payload:
            return str(payload["output_text"])

        return ""
