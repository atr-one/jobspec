from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from jobspec import cli
from jobspec.base import ExtractionResult, Extractor
from jobspec.batch import BatchInput, done_in_output, extract_many, read_jsonl
from jobspec.llm import LLMExtractor
from jobspec.rules import RuleExtractor
from jobspec.schema import JobSpec

from .conftest import FakeProvider, reply

# claude-haiku-4-5 at the example prices, 1,200 input and 200 output tokens per call
COST_PER_CALL = (1200 * 1.00 + 200 * 5.00) / 1_000_000


def _number(user: str) -> int:
    m = re.search(r"Posting number (\d+)", user)
    assert m
    return int(m.group(1))


def slow_first(system: str, user: str) -> str:
    """Earlier postings answer later, so a pool that yields in completion order would reorder them."""
    n = _number(user)
    time.sleep(max(0.0, 0.06 - n * 0.01))
    return reply(title=f"Engineer {n}")


def postings(n: int) -> List[Dict[str, Any]]:
    return [{"id": f"p{i}", "text": f"Posting number {i}\nRemote"} for i in range(n)]


def write_jsonl(path: Path, rows: List[Any]) -> None:
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_out(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- library ----------------------------------------------------------------------------------


def test_results_come_back_in_input_order_with_a_pool() -> None:
    provider = FakeProvider(reply=slow_first)
    items = [(row["id"], row["text"]) for row in postings(8)]
    run = extract_many(items, LLMExtractor(provider), workers=4)
    results = list(run)
    assert [r.id for r in results] == [f"p{i}" for i in range(8)]
    assert [r.result.spec.title for r in results if r.result] == [f"Engineer {i}" for i in range(8)]
    assert [r.index for r in results] == list(range(8))
    assert run.count == 8 and run.failures == 0
    assert run.usage.calls == 8 and run.usage.input_tokens == 8 * 1200
    assert run.usage.cost_usd == pytest.approx(8 * COST_PER_CALL)


def test_plain_strings_get_their_position_as_id() -> None:
    run = extract_many(["Senior Go Engineer\nRemote (US)", "Junior QA Engineer\nOn-site"], RuleExtractor(), workers=1)
    results = list(run)
    assert [r.id for r in results] == ["0", "1"]
    assert [r.result.spec.seniority for r in results if r.result] == ["senior", "junior"]


class Exploding(Extractor):
    name = "exploding"

    def extract(self, text: str, html: Optional[bool] = None) -> ExtractionResult:
        if "boom" in text:
            raise RuntimeError("parser crashed")
        return ExtractionResult(spec=JobSpec(title=text))


def test_an_exception_in_the_extractor_is_one_failed_record() -> None:
    run = extract_many(["a", "boom", "c"], Exploding(), workers=2)
    results = list(run)
    assert [r.ok for r in results] == [True, False, True]
    assert results[1].errors == ["RuntimeError: parser crashed"]
    assert run.failures == 1


def test_max_cost_stops_before_the_limit() -> None:
    provider = FakeProvider(reply=lambda s, u: reply(title="x"))
    limit = 4.5 * COST_PER_CALL
    run = extract_many([(str(i), f"Posting number {i}") for i in range(20)], LLMExtractor(provider), 4, limit)
    results = list(run)
    assert len(results) == 4 and len(provider.calls) == 4
    assert run.stopped and run.stop_reason and "past" in run.stop_reason
    assert run.usage.cost_usd is not None and run.usage.cost_usd <= limit


def test_max_cost_without_a_price_stops_after_the_first_call() -> None:
    provider = FakeProvider(reply=lambda s, u: reply(title="x"), model="unpriced-model")
    run = extract_many(["a", "b", "c"], LLMExtractor(provider), workers=4, max_cost=1.0)
    assert len(list(run)) == 1
    assert run.stopped and run.stop_reason and "unknown" in run.stop_reason


def test_bad_arguments() -> None:
    with pytest.raises(ValueError):
        extract_many([], RuleExtractor(), workers=0)
    with pytest.raises(ValueError):
        extract_many([], RuleExtractor(), max_cost=0)


def test_read_jsonl_reports_bad_lines_and_skips_done_ones() -> None:
    lines = [
        json.dumps({"id": 1, "text": "Go Engineer"}),
        "{not json",
        json.dumps(["a list"]),
        json.dumps({"id": "x", "body": "no text key"}),
        "",
        json.dumps({"text": "no id", "format": "html"}),
        json.dumps({"id": "done", "text": "already written"}),
    ]
    items = list(read_jsonl(lines, skip_ids={"done"}))
    assert [(i.id, i.line) for i in items] == [("1", 1), (None, 2), (None, 3), ("x", 4), (None, 6)]
    assert items[0].error is None and items[0].text == "Go Engineer"
    assert items[1].error and "invalid JSON" in items[1].error
    assert items[2].error and "not a JSON object" in items[2].error
    assert items[3].error and "'text'" in items[3].error
    assert items[4].html is True
    assert [i.line for i in read_jsonl(lines, skip_lines={2, 6})] == [1, 3, 4, 7]


def test_unreadable_inputs_keep_their_place() -> None:
    items = [BatchInput("a", "Senior Go Engineer"), BatchInput(None, None, line=2, error="bad"), BatchInput("c", "QA")]
    results = list(extract_many(items, RuleExtractor(), workers=2))
    assert [r.id for r in results] == ["a", None, "c"]
    assert results[1].errors == ["bad"] and results[1].to_dict()["record"] is None


def test_done_in_output_drops_a_cut_off_last_line(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    out.write_bytes(b'{"id": "a", "line": 1}\n{"id": null, "line": 2}\n{"id": "c", "li')
    assert done_in_output(str(out)) == ({"a"}, {2})
    assert out.read_bytes().endswith(b'"line": 2}\n')
    assert done_in_output(str(tmp_path / "missing.jsonl")) == (set(), set())


# --- command line -----------------------------------------------------------------------------


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeProvider:
    provider = FakeProvider(reply=slow_first)
    monkeypatch.setattr(cli, "make_provider", lambda *args, **kwargs: provider)
    return provider


def test_cli_batch_llm_writes_records_in_input_order(
    tmp_path: Path, fake_llm: FakeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(src, postings(6))
    code = cli.main(["batch", "--in", str(src), "--out", str(out), "--extractor", "llm", "--workers", "3"])
    assert code == 0
    rows = read_out(out)
    assert [r["id"] for r in rows] == [f"p{i}" for i in range(6)]
    assert rows[2]["record"]["title"] == "Engineer 2"
    assert rows[2]["meta"]["title"] == {"source": "llm", "confidence": 0.7}
    assert rows[2]["errors"] == [] and rows[2]["usage"]["input_tokens"] == 1200
    err = capsys.readouterr().err
    assert "records written: 6 (failed: 0)" in err
    assert "LLM calls: 6" in err and "tokens in / out: 7200 / 1200" in err and "elapsed:" in err


def test_cli_batch_resume_skips_ids_already_written(tmp_path: Path, fake_llm: FakeProvider) -> None:
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(src, postings(5))
    args = ["batch", "--in", str(src), "--out", str(out), "--extractor", "llm"]
    assert cli.main([*args, "--max-cost", str(2.5 * COST_PER_CALL)]) == 3
    assert [r["id"] for r in read_out(out)] == ["p0", "p1"]
    assert cli.main([*args, "--resume"]) == 0
    assert [r["id"] for r in read_out(out)] == ["p0", "p1", "p2", "p3", "p4"]
    assert len(fake_llm.calls) == 5
    assert cli.main([*args, "--resume"]) == 0  # nothing left to do
    assert len(read_out(out)) == 5 and len(fake_llm.calls) == 5


def test_cli_batch_cost_stop_message(
    tmp_path: Path, fake_llm: FakeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(src, postings(10))
    limit = str(3.2 * COST_PER_CALL)
    args = ["batch", "--in", str(src), "--out", str(out), "--extractor", "llm", "--max-cost", limit]
    assert cli.main(args) == 3
    assert len(read_out(out)) == 3
    err = capsys.readouterr().err
    assert "stopped before the end of the input" in err and "--resume" in err


def test_cli_batch_malformed_lines_are_reported_not_fatal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    rows: List[Any] = [
        {"key": "a", "body": "Senior Go Engineer\nRemote (US)\n$150k-$180k per year"},
        "{broken",
        {"key": "c", "text": "wrong field"},
        {"key": "d", "body": "Junior QA Engineer\nOn-site in Berlin"},
    ]
    write_jsonl(src, rows)
    args = ["batch", "--in", str(src), "--out", str(out), "--text-field", "body", "--id-field", "key"]
    assert cli.main(args) == 1
    got = read_out(out)
    assert [r["id"] for r in got] == ["a", None, "c", "d"]
    assert got[0]["record"]["salary"]["max"] == 180000 and got[0]["meta"]["salary"]["source"] == "rule"
    assert got[1]["line"] == 2 and "invalid JSON" in got[1]["errors"][0]
    assert got[2]["record"] is None and "'body'" in got[2]["errors"][0]
    assert got[3]["record"]["seniority"] == "junior"
    assert "records written: 4 (failed: 2)" in capsys.readouterr().err
    # resume: the broken line is recognised by its line number, the rest by id
    assert cli.main([*args, "--resume"]) == 0
    assert len(read_out(out)) == 4


def test_cli_batch_rejects_bad_options(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    write_jsonl(src, postings(1))
    assert cli.main(["batch", "--in", str(src), "--out", str(tmp_path / "o"), "--workers", "0"]) == 2
    assert cli.main(["batch", "--in", str(src), "--out", str(tmp_path / "o"), "--max-cost", "0"]) == 2
