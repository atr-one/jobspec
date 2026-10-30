from __future__ import annotations

from jobspec.hybrid import HybridExtractor
from jobspec.llm import LLMExtractor

from .conftest import FakeProvider, reply

POSTING = """Senior Backend Engineer
Location: Remote (US)
Salary: $150,000 - $180,000 per year
Our stack: Go, PostgreSQL and Kubernetes. 5+ years of experience.
Apply at https://acme.example/jobs/1
"""


def test_llm_fills_only_what_rules_left_empty() -> None:
    provider = FakeProvider(
        [
            reply(
                title="SHOULD NOT REPLACE",
                employment_type="full_time",
                visa_sponsorship=False,
                apply_email="jobs@acme.example",
            )
        ]
    )
    result = HybridExtractor(LLMExtractor(provider)).extract(POSTING)
    spec = result.spec
    assert spec.title == "Senior Backend Engineer"
    assert spec.meta["title"].source == "rule"
    assert spec.employment_type == "full_time" and spec.meta["employment_type"].source == "llm"
    assert spec.visa_sponsorship is False and spec.meta["visa_sponsorship"].source == "llm"
    assert spec.salary is not None and spec.meta["salary"].source == "rule"
    assert result.usage.calls == 1
    asked = provider.calls[0]["user"]
    assert "Only fill these keys:" in asked
    assert "employment_type" in asked and "title," not in asked.split("Only fill these keys:")[1].split(".")[0]


def test_low_confidence_rule_value_is_replaced_but_empty_answer_keeps_it() -> None:
    text = "Data person\nSalary: $150,000 - $180,000"  # no period written: inferred, lower confidence
    provider = FakeProvider([reply(salary=None, title="Data Analyst")])
    hybrid = HybridExtractor(LLMExtractor(provider), threshold=0.71)
    spec = hybrid.extract(text).spec
    assert spec.salary is not None and spec.meta["salary"].source == "rule"
    assert spec.title == "Data Analyst" and spec.meta["title"].source == "llm"


def test_no_llm_call_when_every_field_is_confident() -> None:
    provider = FakeProvider([])
    hybrid = HybridExtractor(LLMExtractor(provider), fields=["title", "salary", "work_mode"])
    result = hybrid.extract(POSTING)
    assert provider.calls == [] and result.usage.calls == 0
    assert result.spec.work_mode == "remote"


def test_llm_error_keeps_rule_values() -> None:
    result = HybridExtractor(LLMExtractor(FakeProvider(["not json"]))).extract(POSTING)
    assert result.error
    assert result.spec.title == "Senior Backend Engineer"
    assert result.spec.salary is not None
