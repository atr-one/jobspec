"""jobspec: structured, validated records from raw job postings.

>>> from jobspec import RuleExtractor
>>> spec = RuleExtractor().extract("Senior Go Engineer\\nRemote (US)\\n$150k-$180k").spec
>>> spec.seniority, spec.work_mode, spec.salary.currency
('senior', 'remote', 'USD')
"""

from __future__ import annotations

from .base import ExtractionResult, Extractor, Usage
from .evaluate import EvalReport, Example, evaluate, load_dataset
from .hybrid import HybridExtractor
from .llm import LLMExtractor
from .providers import AnthropicProvider, Completion, OpenAICompatibleProvider, Provider, ProviderError
from .rules import RuleExtractor
from .schema import FieldMeta, JobSpec, Salary, SchemaError, json_schema

__version__ = "0.1.0"

__all__ = [
    "AnthropicProvider",
    "Completion",
    "EvalReport",
    "Example",
    "ExtractionResult",
    "Extractor",
    "FieldMeta",
    "HybridExtractor",
    "JobSpec",
    "LLMExtractor",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderError",
    "RuleExtractor",
    "Salary",
    "SchemaError",
    "Usage",
    "__version__",
    "evaluate",
    "json_schema",
    "load_dataset",
]
