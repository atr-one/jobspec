"""Hybrid extractor: rules first, the LLM only for what the rules left empty or unsure."""

from __future__ import annotations

import copy
import time
from typing import List, Optional, Sequence

from .base import ExtractionResult, Extractor, Usage
from .llm import LLMExtractor
from .rules import RuleExtractor
from .schema import FIELDS


class HybridExtractor(Extractor):
    """Run the rules; ask the LLM, in one call, only for the fields that are empty or whose rule
    confidence is below ``threshold``. Every field keeps its ``source`` and ``confidence``.

    A rule value is replaced only when the LLM returns a value for that field; an empty LLM answer
    leaves the rule value in place. When no field needs help no LLM call is made.
    """

    name = "hybrid"

    def __init__(
        self,
        llm: LLMExtractor,
        rules: Optional[RuleExtractor] = None,
        threshold: float = 0.6,
        fields: Sequence[str] = FIELDS,
    ) -> None:
        self.llm = llm
        self.rules = rules or RuleExtractor()
        self.threshold = threshold
        self.fields = list(fields)

    def fields_to_ask(self, result: ExtractionResult) -> List[str]:
        spec = result.spec
        return [f for f in self.fields if spec.is_empty(f) or spec.confidence(f) < self.threshold]

    def extract(self, text: str, html: Optional[bool] = None) -> ExtractionResult:
        started = time.perf_counter()
        base = self.rules.extract(text, html)
        need = self.fields_to_ask(base)
        if not need:
            base.latency_s = time.perf_counter() - started
            return base
        answer = self.llm.extract(text, html, fields=need)
        merged = copy.deepcopy(base.spec)
        for name in need:
            if answer.spec.is_empty(name):
                continue
            meta = answer.spec.meta.get(name)
            merged.set(name, answer.spec.get(name), "llm", meta.confidence if meta else 0.7)
        usage = Usage()
        usage.add(answer.usage)
        return ExtractionResult(
            spec=merged,
            latency_s=time.perf_counter() - started,
            usage=usage,
            warnings=base.warnings + answer.warnings,
            error=answer.error,
        )
