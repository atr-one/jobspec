"""The common extractor interface and its result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .schema import JobSpec


@dataclass
class Usage:
    """Token use and cost of the LLM calls behind one extraction (all zero for the rules)."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = 0.0

    def add(self, other: Usage) -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        if self.cost_usd is None or other.cost_usd is None:
            self.cost_usd = None
        else:
            self.cost_usd += other.cost_usd


@dataclass
class ExtractionResult:
    spec: JobSpec
    latency_s: float = 0.0
    usage: Usage = field(default_factory=Usage)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


class Extractor:
    """Base class: ``extract`` turns posting text (plain or HTML) into an ``ExtractionResult``."""

    name = "base"

    def extract(self, text: str, html: Optional[bool] = None) -> ExtractionResult:
        raise NotImplementedError
