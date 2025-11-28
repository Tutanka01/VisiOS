"""Vision agent that calls Ollama to interpret UI screenshots."""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "un": 1,
    "une": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
    "petit": 1,  # fallback to avoid returning 0 when parsing phrases
}

SYSTEM_PROMPT = (
    "You are visiOS, an accessibility agent.\n"
    "Given the attached desktop screenshot, decide the single next UI action "
    "needed to satisfy the user's request.\n"
    "Think step-by-step: if text needs to be typed, first make sure the relevant field is focused (click or press <CTRL+L> for address bar) before typing.\n"
    "Respond using JSON with the following schema: {{"
    "\"action\": \"click|type|scroll|noop\", "
    "\"coordinates\": {{\"x\": <number>, \"y\": <number>}}, "
    "\"text\": \"optional text to type or special key tokens like <ENTER>, <CTRL+L>\" }}.\n"
    "When you need to type text (URLs, queries, commands), set action to \"type\" and fill the text field.\n"
    "Combine text with key tokens by embedding them inside angle brackets, e.g. 'example.com<ENTER>' or '<CTRL+L>https://site.com<ENTER>'.\n"
    "When a click is required, provide coordinates relative to the screenshot.\n"
    "Use action=\"scroll\" with text or y values to describe how far to scroll (positive for down, negative for up).\n"
    "Focus on high-impact actions: open apps if they are not visible, bring windows to the front, then interact with the UI element that enables the user's request.\n"
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
    reasoning: Optional[str] = None
    scroll_amount: Optional[float] = None

    def is_click(self) -> bool:
        return self._normalized_type() == "click"

    def is_type(self) -> bool:
        return self._normalized_type() == "type"

    def _normalized_type(self) -> str:
        raw = (self.action_type or "").strip().lower()
        if not raw:
            return ""
        normalized = raw.replace("-", "_")
        if normalized in {"tap", "press"}:
            return "click"
        if normalized.startswith("scroll") or normalized in {"wheel", "mousewheel"}:
            return "scroll"
        return normalized

    def pointer(self) -> Optional[Tuple[float, float]]:
        if self.x is None or self.y is None:
            return None
        return self.x, self.y


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

    def decide_action(
        self,
        screenshot: bytes,
        user_prompt: str,
        screen_size: Optional[Tuple[int, int]] = None,
    ) -> AgentAction:
        """Send the screenshot + prompt to the backend and parse the textual response."""

        LOGGER.info("Querying %s backend with %s bytes of image data", self.api_mode, len(screenshot))
        encoded_image = base64.b64encode(screenshot).decode("ascii")
        prompt = self._build_prompt(user_prompt, screen_size)

        if self.api_mode == "openai":
            full_text, backend_reasoning = self._call_openai_api(encoded_image, prompt, user_prompt)
        else:
            full_text, backend_reasoning = self._call_ollama_api(encoded_image, prompt)

        cleaned_text, inline_reasoning = self._strip_reasoning_sections(full_text)

        LOGGER.debug("Vision backend response: %s", cleaned_text)
        action = self._parse_action(cleaned_text)
        action.reasoning = self._merge_reasoning_segments(action.reasoning, backend_reasoning, inline_reasoning)
        return action

    # ------------------------------------------------------------------
    def _build_prompt(self, user_prompt: str, screen_size: Optional[Tuple[int, int]]) -> str:
        extra = []
        if screen_size and all(isinstance(value, (int, float)) and value > 0 for value in screen_size):
            width, height = screen_size
            extra.append(
                f"Current display resolution is {int(width)}x{int(height)} pixels. "
                "Return click coordinates in that pixel space (or normalized 0-1000 scale)."
            )
        else:
            extra.append(
                "When providing coordinates, assume the screenshot resolution and send either pixels or a 0-1000 normalized scale."
            )
        extra.append(
            "Prefer returning absolute pixels for precision. Only emit one action per response."
        )
        extra.append(
            "For scroll actions, always include the cursor coordinates where the wheel must occur and a numeric magnitude (positive = down, negative = up)."
        )
        extra.append(
            "Map qualitative hints to numbers: small/petit/doucement -> 1 to 3, medium -> 4 to 7, large/gros/rapidement -> 8 to 15, and respect explicit counts like 'trois scrolls' by multiplying the magnitude accordingly."
        )
        supplemental = " ".join(extra)
        return f"{SYSTEM_PROMPT}{supplemental}\nUser request: {user_prompt}\n"

    def _parse_action(self, model_text: str) -> AgentAction:
        """Best-effort extraction of coordinates/text from model output."""

        action = AgentAction(action_type="noop", raw_response=model_text)
        reasoning_parts: List[str] = []

        json_blob = self._extract_json(model_text)
        if json_blob is not None:
            reasoning_parts.extend(self._extract_reasoning_fields(json_blob))
            action = self._from_json(json_blob, model_text)
        else:
            match = re.search(r"click.*?(\d+)\D+(\d+)", model_text, re.IGNORECASE | re.DOTALL)
            if match:
                action = AgentAction(action_type="click", x=float(match.group(1)), y=float(match.group(2)), raw_response=model_text)
            elif "type" in model_text.lower():
                text_match = re.search(r"type[:\s]+(.+)$", model_text, re.IGNORECASE)
                action = AgentAction(action_type="type", text=text_match.group(1).strip() if text_match else "", raw_response=model_text)
            elif "scroll" in model_text.lower():
                amount = self._extract_numeric_from_text(model_text)
                action = AgentAction(action_type="scroll", text=model_text, raw_response=model_text, scroll_amount=amount)
        if reasoning_parts:
            action.reasoning = self._merge_reasoning_segments(action.reasoning, *reasoning_parts)
        return action

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        stripped = VisionAgent._sanitize_json_text(text)
        if stripped:
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                loosened = VisionAgent._loosen_braces(stripped)
                if loosened != stripped:
                    try:
                        return json.loads(loosened)
                    except json.JSONDecodeError:
                        pass

        candidate = None
        for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
            candidate = match.group(0)
        if candidate is None:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            loosened = VisionAgent._loosen_braces(candidate)
            if loosened != candidate:
                try:
                    return json.loads(loosened)
                except json.JSONDecodeError:
                    return None
            return None

    @staticmethod
    def _sanitize_json_text(text: str) -> str:
        stripped = text.strip()

        if stripped.startswith("```"):
            parts = stripped.split("```")
            if len(parts) >= 3:
                stripped = parts[1].strip()
            else:
                stripped = stripped.strip("`")

        while stripped.startswith("{{") and stripped.endswith("}}"):  # handle double braces
            stripped = stripped[1:-1].strip()

        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        return ""

    @staticmethod
    def _loosen_braces(text: str) -> str:
        if "{{" not in text and "}}" not in text:
            return text
        return text.replace("{{", "{").replace("}}", "}")

    @staticmethod
    def _from_json(payload: Dict[str, Any], raw_text: str) -> AgentAction:
        action = payload.get("action") or payload.get("type") or "noop"
        coords = payload.get("coordinates") or payload.get("coords") or {}
        text = payload.get("text") or payload.get("value")
        x = y = None
        if isinstance(coords, dict):
            x = VisionAgent._to_number(coords.get("x") or coords.get("X"))
            y = VisionAgent._to_number(coords.get("y") or coords.get("Y"))
        elif isinstance(coords, (list, tuple)) and len(coords) >= 2:
            x = VisionAgent._to_number(coords[0])
            y = VisionAgent._to_number(coords[1])
        else:
            x = VisionAgent._to_number(payload.get("x"))
            y = VisionAgent._to_number(payload.get("y"))
        scroll_amount = VisionAgent._extract_scroll_amount(payload, text)
        return AgentAction(
            action_type=str(action),
            x=x,
            y=y,
            text=text,
            raw_response=raw_text,
            scroll_amount=scroll_amount,
        )

    @staticmethod
    def _extract_scroll_amount(payload: Dict[str, Any], text: Optional[str]) -> Optional[float]:
        candidates = []
        for key in ("scroll", "delta", "amount", "value", "distance", "lines", "magnitude"):
            if key in payload:
                candidates.append(payload[key])
        coords = payload.get("coordinates") or payload.get("coords")
        if isinstance(coords, dict):
            for key in ("delta", "amount"):
                if key in coords:
                    candidates.append(coords[key])
        meta = payload.get("meta") or payload.get("metadata")
        if isinstance(meta, dict):
            for key in ("scroll", "delta", "amount"):
                if key in meta:
                    candidates.append(meta[key])
        for candidate in candidates:
            number = VisionAgent._to_number(candidate)
            if number is not None:
                return number
        return VisionAgent._extract_numeric_from_text(text)

    @staticmethod
    def _extract_numeric_from_text(text: Optional[str]) -> Optional[float]:
        if not text:
            return None
        if isinstance(text, (int, float)):
            return float(text)
        if not isinstance(text, str):
            text = str(text)
        match = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if not match:
            return VisionAgent._word_to_number(text)
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _word_to_number(text: str) -> Optional[float]:
        lowered = text.lower()
        tokens = re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ]+", lowered)
        for token in tokens:
            normalized = VisionAgent._strip_accents(token)
            if normalized in NUMBER_WORDS:
                return float(NUMBER_WORDS[normalized])
        return None

    @staticmethod
    def _strip_accents(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    @staticmethod
    def _to_number(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Backend adapters
    # ------------------------------------------------------------------
    def _endpoint(self, path: str) -> str:
        cleaned_base = self.base_url.rstrip("/")
        cleaned_path = path.strip("/")
        return f"{cleaned_base}/{cleaned_path}" if cleaned_path else cleaned_base

    def _call_ollama_api(self, encoded_image: str, prompt: str) -> Tuple[str, Optional[str]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
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
        return "".join(collected).strip(), None

    def _call_openai_api(
        self,
        encoded_image: str,
        prompt: str,
        user_prompt: str,
    ) -> Tuple[str, Optional[str]]:
        if self.openai_endpoint in {"chat", "chat_completions", "v1/chat/completions"}:
            return self._call_openai_chat(encoded_image, prompt, user_prompt)

        if self.openai_endpoint in {"completions", "completion", "v1/completions"}:
            return self._call_openai_completion(encoded_image, prompt, user_prompt)

        return self._call_openai_responses(encoded_image, prompt, user_prompt)

    def _call_openai_responses(
        self,
        encoded_image: str,
        prompt: str,
        user_prompt: str,
    ) -> Tuple[str, Optional[str]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": prompt.strip()}],
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
        text, reasoning = self._extract_openai_text(data)
        if text:
            return text.strip(), reasoning
        LOGGER.warning("Unable to parse OpenAI responses payload: %s", json.dumps(data)[:500])
        return "", reasoning

    def _call_openai_chat(
        self,
        encoded_image: str,
        prompt: str,
        user_prompt: str,
    ) -> Tuple[str, Optional[str]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": prompt.strip()}],
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
        text, reasoning = self._extract_openai_text(data)
        if text:
            return text.strip(), reasoning
        LOGGER.warning("Unable to parse OpenAI chat payload: %s", json.dumps(data)[:500])
        return "", reasoning

    def _call_openai_completion(
        self,
        encoded_image: str,
        prompt: str,
        user_prompt: str,
    ) -> Tuple[str, Optional[str]]:
        composed_prompt = (
            f"{prompt}"  # already includes System + user prompt introduction
            "Image (base64 PNG) follows between <image></image> tags.\n"
            f"<image>{encoded_image}</image>"
        )
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": composed_prompt,
            "temperature": self.temperature,
        }
        url = self._endpoint("v1/completions")
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        text, reasoning = self._extract_openai_text(data)
        if text:
            return text.strip(), reasoning
        LOGGER.warning("Unable to parse OpenAI completions payload: %s", json.dumps(data)[:500])
        return "", reasoning

    @staticmethod
    def _extract_openai_text(payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        reasoning_segments: List[str] = []

        def _iter_text(blobs: Iterable[Dict[str, Any]]) -> Iterable[str]:
            for blob in blobs:
                if not isinstance(blob, dict):
                    continue
                if "content" in blob and isinstance(blob["content"], list):
                    for content in blob["content"]:
                        if not isinstance(content, dict):
                            continue
                        segment_type = content.get("type")
                        if segment_type == "reasoning":
                            normalized = VisionAgent._normalize_reasoning_value(content.get("reasoning"))
                            if normalized:
                                reasoning_segments.append(normalized)
                            continue
                        text_value = content.get("text") or content.get("value")
                        if text_value:
                            yield str(text_value)
                else:
                    text_value = blob.get("text") or blob.get("value")
                    if text_value:
                        yield str(text_value)

        if "output" in payload and isinstance(payload["output"], list):
            text_value = "".join(_iter_text(payload["output"]))
            return text_value, VisionAgent._merge_reasoning_segments(*reasoning_segments)

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for segment in content:
                    if not isinstance(segment, dict):
                        continue
                    segment_type = segment.get("type")
                    if segment_type == "reasoning":
                        normalized = VisionAgent._normalize_reasoning_value(segment.get("reasoning"))
                        if normalized:
                            reasoning_segments.append(normalized)
                        continue
                    if "text" in segment:
                        parts.append(str(segment["text"]))
                    elif segment_type in {"output_text", "text"} and "value" in segment:
                        parts.append(str(segment["value"]))
                if parts:
                    return "".join(parts), VisionAgent._merge_reasoning_segments(*reasoning_segments)
            if isinstance(content, str):
                return content, VisionAgent._merge_reasoning_segments(*reasoning_segments)
            if "text" in choice:
                return str(choice["text"]), VisionAgent._merge_reasoning_segments(*reasoning_segments)

        if "output_text" in payload:
            return str(payload["output_text"]), VisionAgent._merge_reasoning_segments(*reasoning_segments)

        return "", VisionAgent._merge_reasoning_segments(*reasoning_segments)

    @staticmethod
    def _strip_reasoning_sections(text: str) -> Tuple[str, Optional[str]]:
        if not text:
            return text, None

        reasoning_parts: List[str] = []
        cleaned = text

        def _store_snippet(snippet: str) -> None:
            snippet = snippet.strip()
            if snippet:
                reasoning_parts.append(snippet)

        for tag in ("think", "thinking", "reasoning", "thought", "thoughts"):
            pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)

            def _replace(match: re.Match[str]) -> str:
                _store_snippet(match.group(1))
                return ""

            cleaned = pattern.sub(_replace, cleaned)

        block_pattern = re.compile(
            r"(?is)(?:^|\n)\s*(?:reasoning|analysis|thoughts?)\s*:\s*(.+?)(?=\n\s*(?:action|result|answer)\s*:|\Z)"
        )

        cleaned = block_pattern.sub(lambda match: (_store_snippet(match.group(1)) or "\n"), cleaned)
        cleaned = cleaned.strip()

        return cleaned, VisionAgent._merge_reasoning_segments(*reasoning_parts)

    @staticmethod
    def _merge_reasoning_segments(*segments: Optional[str]) -> Optional[str]:
        parts: List[str] = []
        for segment in segments:
            if not segment:
                continue
            normalized = segment.strip()
            if normalized and normalized not in parts:
                parts.append(normalized)
        if parts:
            return "\n\n".join(parts)
        return None

    @staticmethod
    def _normalize_reasoning_value(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        if isinstance(value, (list, tuple)):
            parts = [VisionAgent._normalize_reasoning_value(item) for item in value]
            parts = [part for part in parts if part]
            if parts:
                return "\n".join(parts)
            return None
        if isinstance(value, dict):
            for candidate in ("text", "value", "content", "summary"):
                if candidate in value:
                    normalized = VisionAgent._normalize_reasoning_value(value[candidate])
                    if normalized:
                        return normalized
            nested = [VisionAgent._normalize_reasoning_value(v) for v in value.values()]
            nested = [part for part in nested if part]
            if nested:
                return "\n".join(nested)
            return None
        try:
            text = str(value).strip()
        except Exception:  # pragma: no cover - extremely defensive
            return None
        return text or None

    @staticmethod
    def _extract_reasoning_fields(payload: Dict[str, Any]) -> List[str]:
        reasoning_keys = (
            "reasoning",
            "thinking",
            "thoughts",
            "thought",
            "analysis",
            "plan",
            "inner_monologue",
            "chain_of_thought",
        )
        collected: List[str] = []
        for key in reasoning_keys:
            if key in payload:
                normalized = VisionAgent._normalize_reasoning_value(payload[key])
                if normalized:
                    collected.append(normalized)
        for extra_key in ("meta", "metadata"):
            extra = payload.get(extra_key)
            if isinstance(extra, dict):
                collected.extend(VisionAgent._extract_reasoning_fields(extra))
        return collected
