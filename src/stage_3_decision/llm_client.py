"""
Thin wrapper around the Ollama Python client for local LLM inference.

Responsibilities:
  - Send system + user prompt pairs to a locally running Ollama server
  - Extract JSON from the raw response text (handles markdown code fences)
  - Retry on transient failures; fall back gracefully when Ollama is unavailable
  - Single point of configuration for model name and host

Setup (one-time):
    1. Install Ollama from https://ollama.com
    2. Pull the model:  ollama pull llama3.2
    3. Ollama server runs automatically in the background on port 11434
"""
from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

_DEFAULT_MODEL = "llama3.2"
_DEFAULT_HOST  = "http://localhost:11434"
_MAX_RETRIES   = 2

# JSON extraction: strip optional ```json ... ``` or ``` ... ``` fences
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)```",
    re.IGNORECASE,
)


class OllamaClient:
    """
    Chat client for a locally running Ollama model.

    Args:
        model: Ollama model tag (default: llama3.2).
        host:  Ollama server URL (default: http://localhost:11434).
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host:  str = _DEFAULT_HOST,
    ) -> None:
        self.model = model
        self.host  = host
        self._client = self._build_client()

    # ── public API ────────────────────────────────────────────────────────────

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        format: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """
        Send a chat request and return the raw response string.

        Retries up to _MAX_RETRIES times on transient errors.
        Returns an empty string if all retries fail.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        kwargs: dict[str, Any] = {}
        if format:
            kwargs["format"] = format
        if options:
            kwargs["options"] = options

        for attempt in range(1, _MAX_RETRIES + 2):
            try:
                response = self._client.chat(model=self.model, messages=messages, **kwargs)
                content  = response["message"]["content"]
                logger.debug(
                    "OllamaClient.chat | model={} | attempt={} | chars={}",
                    self.model, attempt, len(content),
                )
                return content
            except Exception as exc:
                logger.warning(
                    "OllamaClient.chat | attempt={} failed | error={}",
                    attempt, exc,
                )
                if attempt > _MAX_RETRIES:
                    logger.error(
                        "OllamaClient.chat | all retries exhausted | model={}", self.model
                    )
                    return ""
        return ""

    def chat_json(
        self,
        system_prompt: str,
        user_prompt:   str,
        fallback:      dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat request, extract and parse the JSON response.
        Uses Ollama's native JSON mode and extended generation limit.
        If extraction fails, returns ``fallback`` (defaults to empty dict).
        """
        raw = self.chat(
            system_prompt,
            user_prompt,
            format="json",
            options={"num_ctx": 8192, "num_predict": 2048},
        )
        if not raw:
            logger.warning("OllamaClient.chat_json | empty response; using fallback")
            return fallback or {}

        result = _extract_json(raw)
        if result is None:
            logger.warning(
                "OllamaClient.chat_json | JSON parse failed | raw_snippet={}",
                raw[:120],
            )
            return fallback or {}

        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_client(self):
        try:
            import ollama
            client = ollama.Client(host=self.host)
            logger.info(
                "OllamaClient initialised | model={} | host={}", self.model, self.host
            )
            return client
        except ImportError as exc:
            raise ImportError(
                "ollama package not installed. Run: pip install ollama>=0.3.0"
            ) from exc


# ── module helpers ────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict[str, Any] | None:
    """
    Try several strategies to extract a JSON dict from an LLM response.

    1. Direct parse (model returned clean JSON).
    2. Extract content inside ```json ... ``` or ``` ... ``` fences.
    3. Find the first {...} block in the text.
    """
    text = text.strip()

    # Strategy 1 — direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2 — fenced code block
    match = _JSON_FENCE_RE.search(text)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 3 — first {...} block
    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            result = json.loads(text[brace_start : brace_end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None
