"""LLM providers over plain HTTP (``urllib.request``): Anthropic Messages API and any
OpenAI-compatible chat completions endpoint. No third-party dependencies."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
ANTHROPIC_VERSION = "2026-10-21"
_RETRY_STATUS = frozenset([408, 409, 429, 500, 502, 503, 504, 529])


class ProviderError(RuntimeError):
    """An LLM call failed (HTTP error, timeout, malformed response)."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


Transport = Callable[[str, Dict[str, str], bytes, float], Dict[str, Any]]


def urllib_transport(url: str, headers: Dict[str, str], body: bytes, timeout: float) -> Dict[str, Any]:
    """POST ``body`` and return the decoded JSON response. Raises ``ProviderError``."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise ProviderError(f"HTTP {exc.code} from {url}: {detail}", status=exc.code) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ProviderError(f"request to {url} failed: {exc}") from exc
    try:
        data = json.loads(payload)
    except ValueError as exc:
        raise ProviderError(f"response from {url} is not JSON") from exc
    if not isinstance(data, dict):
        raise ProviderError(f"response from {url} is not a JSON object")
    return data


class Provider:
    """Base class. ``complete`` sends one system + user message and returns the reply text."""

    name = "base"

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        base_url: str,
        timeout: float = 60.0,
        max_retries: int = 2,
        transport: Optional[Transport] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.transport: Transport = transport or urllib_transport

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion:
        raise NotImplementedError

    def _post(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        attempt = 0
        while True:
            try:
                return self.transport(url, headers, body, self.timeout)
            except ProviderError as exc:
                retryable = exc.status is None or exc.status in _RETRY_STATUS
                if not retryable or attempt >= self.max_retries:
                    raise
                time.sleep(min(2.0**attempt, 8.0))
                attempt += 1


class AnthropicProvider(Provider):
    """Claude through the Messages API (``POST /v1/messages``)."""

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com",
        api_version: str = ANTHROPIC_VERSION,
        effort: Optional[str] = "low",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key or os.environ.get("ANTHROPIC_API_KEY"), base_url, **kwargs)
        self.api_version = api_version
        # field extraction needs little reasoning: current models think by default, and "low" keeps
        # those tokens (billed as output) down; None leaves the model's own default
        self.effort = effort

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion:
        if not self.api_key:
            raise ProviderError("no API key: set ANTHROPIC_API_KEY or pass api_key")
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.effort:
            payload["output_config"] = {"effort": self.effort}
        data = self._post(self.base_url + "/v1/messages", headers, payload)
        if data.get("type") == "error":
            raise ProviderError("API error: {}".format(json.dumps(data.get("error"))[:300]))
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        if data.get("stop_reason") == "refusal":
            raise ProviderError("the model declined the request (stop_reason=refusal)")
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            input_tokens=int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            model=str(data.get("model") or self.model),
        )


class OpenAICompatibleProvider(Provider):
    """Any endpoint that implements ``POST {base_url}/chat/completions`` (OpenAI and compatibles)."""

    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        json_mode: bool = True,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        # OPENAI_API_KEY is only sent to OpenAI itself: a key for another endpoint (a local server,
        # a gateway) has to be passed explicitly, so the OpenAI key never leaks to a third party.
        if api_key is None and urllib.parse.urlsplit(base_url).hostname == "api.openai.com":
            api_key = os.environ.get("OPENAI_API_KEY")
        super().__init__(model, api_key, base_url, **kwargs)
        self.json_mode = json_mode
        self.temperature = temperature

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = "Bearer " + self.api_key
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        data = self._post(self.base_url + "/chat/completions", headers, payload)
        if "error" in data and not data.get("choices"):
            raise ProviderError("API error: {}".format(json.dumps(data.get("error"))[:300]))
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("response has no choices")
        message = choices[0].get("message") or {}
        usage = data.get("usage") or {}
        return Completion(
            text=str(message.get("content") or ""),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            model=str(data.get("model") or self.model),
        )


def make_provider(
    name: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
) -> Provider:
    """Build a provider by name: ``anthropic`` or ``openai`` (any OpenAI-compatible endpoint)."""
    if name == "anthropic":
        return AnthropicProvider(
            model=model or DEFAULT_ANTHROPIC_MODEL,
            api_key=api_key,
            base_url=base_url or "https://api.anthropic.com",
            timeout=timeout,
        )
    if name in ("openai", "openai-compatible"):
        return OpenAICompatibleProvider(
            model=model or DEFAULT_OPENAI_MODEL,
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            timeout=timeout,
        )
    raise ValueError(f"unknown provider {name!r} (use anthropic or openai)")
