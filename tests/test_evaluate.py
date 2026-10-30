from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pytest

from jobspec.base import ExtractionResult, Extractor, Usage
from jobspec.evaluate import EVAL_FIELDS, Example, FieldScore, evaluate, load_dataset, score_item
from jobspec.rules import RuleExtractor
from jobspec.schema import JobSpec, Salary

from .conftest import DATA


class Canned(Extractor):
    name = "canned"

    def __init__(self, outputs: List[JobSpec], cost: Optional[float] = 0.001) -> None:
        self.outputs = list(outputs)
        self.cost = cost

    def extract(self, text: str, html: Optional[bool] = None) -> ExtractionResult:
        return ExtractionResult(
            spec=self.outputs.pop(0),
            latency_s=0.5,
            usage=Usage(calls=1, input_tokens=100, output_tokens=10, cost_usd=self.cost),
        )


def scores() -> dict:  # type: ignore[type-arg]
    return {
        n: FieldScore(n, "set" if n in ("locations", "remote_regions", "tech_stack") else "scalar") for n in EVAL_FIELDS
    }


def test_field_score_math() -> None:
    s = FieldScore("x", "scalar", n=10, correct=7, tp=4, pred_count=5, gold_count=8)
    assert s.accuracy == 0.7
    assert s.precision == 0.8
    assert s.recall == 0.5
    assert s.f1 == pytest.approx(2 * 0.8 * 0.5 / 1.3)
    assert FieldScore("y", "scalar").f1 is None
    assert FieldScore("z", "scalar", n=1, gold_count=1).f1 == 0.0


def test_scalar_and_null_agreement() -> None:
    sc = scores()
    gold = JobSpec(seniority="senior", work_mode=None, visa_sponsorship=False)
    pred = JobSpec(seniority="mid", work_mode=None, visa_sponsorship=False)
    mism = score_item(gold, pred, sc)
    assert "seniority" in mism and "work_mode" not in mism
    assert (sc["seniority"].correct, sc["seniority"].tp, sc["seniority"].pred_count) == (0, 0, 1)
    assert (sc["work_mode"].correct, sc["work_mode"].gold_count) == (1, 0)
    assert sc["visa_sponsorship"].tp == 1  # False is a stated value


def test_salary_tolerance_and_subfields() -> None:
    sc = scores()
    gold = JobSpec(salary=Salary(100000, 120000, "USD", "year", None))
    pred = JobSpec(salary=Salary(101000, 120000, "USD", "month", None))
    mism = score_item(gold, pred, sc)
    assert sc["salary.min"].correct == 1  # within 2 %
    assert sc["salary.period"].correct == 0
    assert sc["salary"].correct == 0 and "salary" in mism
    sc2 = scores()
    score_item(gold, JobSpec(salary=Salary(105000, 120000, "USD", "year")), sc2)
    assert sc2["salary.min"].correct == 0


def test_set_fields_precision_recall() -> None:
    sc = scores()
    gold = JobSpec(tech_stack=["Python", "Go", "SQL"], locations=["Berlin, Germany"])
    pred = JobSpec(tech_stack=["python", "golang", "Rust"], locations=["berlin"])
    score_item(gold, pred, sc)
    ts = sc["tech_stack"]
    assert (ts.tp, ts.pred_count, ts.gold_count, ts.correct) == (2, 3, 3, 0)
    assert sc["locations"].correct == 1  # city-level, case and country insensitive


def test_title_url_email_normalization() -> None:
    sc = scores()
    gold = JobSpec(title="Data Analyst", apply_url="https://Acme.example/jobs/1/", apply_email="Jobs@Acme.example")
    pred = JobSpec(title="data  analyst.", apply_url="https://acme.example/jobs/1", apply_email="jobs@acme.example")
    assert score_item(gold, pred, sc) == {}


def test_evaluate_totals_and_reports(tmp_path: Path) -> None:
    examples = [
        Example("a", "x", JobSpec(seniority="senior", tech_stack=["Go"])),
        Example("b", "y", JobSpec(seniority="junior")),
    ]
    report = evaluate(Canned([JobSpec(seniority="senior", tech_stack=["Go"]), JobSpec(seniority="senior")]), examples)
    assert report.n == 2
    assert report.fields["seniority"].accuracy == 0.5
    assert report.fields["tech_stack"].f1 == 1.0
    assert report.avg_latency_s == 0.5
    assert report.usage.calls == 2 and report.usage.cost_usd == pytest.approx(0.002)
    assert report.cost_per_1000 == pytest.approx(1.0)
    data = report.to_dict()
    json.dumps(data)
    assert data["items"][1]["mismatches"]["seniority"] == {"gold": "junior", "pred": "senior"}
    md = report.to_markdown()
    assert "| seniority | 0.50 |" in md and "cost per 1,000 postings" in md
    assert "seniority" in report.to_text()


def test_unknown_cost_propagates() -> None:
    report = evaluate(Canned([JobSpec()], cost=None), [Example("a", "x", JobSpec())])
    assert report.usage.cost_usd is None and report.cost_per_1000 is None
    assert "n/a" in report.to_text()


def test_load_dataset_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "1", "text": "x", "label": {"seniority": "guru"}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        load_dataset(str(bad))


@pytest.mark.parametrize("name,size", [("eval.jsonl", 30), ("dev.jsonl", 12), ("holdout.jsonl", 12)])
def test_bundled_datasets_are_valid(name: str, size: int) -> None:
    examples = load_dataset(str(DATA / name))
    assert len(examples) == size
    assert len({e.id for e in examples}) == size
    assert all(e.label.title and e.label.is_valid() for e in examples)


# Measured 0.997 macro accuracy with 0.2.0 (0.994 with 0.1.0); the threshold leaves a small margin so that
# a refactor that breaks a rule fails here, while the numbers in the README stay the reference.
RULES_MIN_MACRO_ACCURACY = 0.97
RULES_MIN_FIELD_ACCURACY = 0.9


def test_rule_extractor_reaches_minimum_on_bundled_data() -> None:
    report = evaluate(RuleExtractor(), load_dataset(str(DATA / "eval.jsonl")))
    macro = report.macro("accuracy")
    assert macro is not None and macro >= RULES_MIN_MACRO_ACCURACY
    for name in EVAL_FIELDS:
        accuracy = report.fields[name].accuracy
        assert accuracy is not None and accuracy >= RULES_MIN_FIELD_ACCURACY, name
