from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from jobspec.providers import Completion, Provider
from jobspec.text import Document, prepare

DATA = Path(__file__).resolve().parent.parent / "data"


class FakeProvider(Provider):
    """A provider that returns canned replies and records the prompts it was sent."""

    name = "fake"

    def __init__(
        self,
        replies: Optional[List[str]] = None,
        reply: Optional[Callable[[str, str], str]] = None,
        model: str = "claude-haiku-4-5",
        input_tokens: int = 1200,
        output_tokens: int = 200,
    ) -> None:
        super().__init__(model=model, api_key="test", base_url="http://localhost")
        self.replies = list(replies or [])
        self.reply = reply
        self.calls: List[Dict[str, Any]] = []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        text = self.reply(system, user) if self.reply is not None else (self.replies.pop(0) if self.replies else "{}")
        return Completion(text, self.input_tokens, self.output_tokens, self.model)


@pytest.fixture
def fake_provider() -> Callable[..., FakeProvider]:
    return FakeProvider


def doc(text: str) -> Document:
    return prepare(text)


def reply(**fields: Any) -> str:
    return json.dumps(fields)
