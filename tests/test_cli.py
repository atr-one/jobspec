from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from jobspec.cli import main
from jobspec.evaluate import EVAL_FIELDS

from .conftest import DATA


def test_schema(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == "JobSpec"


def test_extract_rules(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    posting = tmp_path / "posting.txt"
    posting.write_text("Senior Go Engineer\nRemote (US)\nSalary: $150k-$180k per year\n", encoding="utf-8")
    assert main(["extract", str(posting), "--extractor", "rules", "--meta"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["seniority"] == "senior"
    assert out["salary"]["max"] == 180000
    assert out["meta"]["salary"]["source"] == "rule"
    assert out["_run"]["extractor"] == "rules"


def test_eval_writes_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    data = str(DATA / "eval.jsonl")
    assert main(["eval", "--data", data, "--extractor", "rules", "--limit", "5", "--report", str(md)]) == 0
    assert main(["eval", "--data", data, "--report", str(js), "--errors"]) == 0
    assert "| field |" in md.read_text(encoding="utf-8")
    assert json.loads(js.read_text(encoding="utf-8"))["postings"] == 30
    assert "macro accuracy" in capsys.readouterr().out


def test_eval_fail_under() -> None:
    assert main(["eval", "--data", str(DATA / "eval.jsonl"), "--fail-under", "1.01"]) == 1


def test_eval_llm_without_key_reports_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["eval", "--data", str(DATA / "eval.jsonl"), "--extractor", "llm", "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "errors                       2" in out


def test_estimate(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["estimate", "--data", str(DATA / "eval.jsonl"), "--model", "claude-haiku-4-5"]) == 0
    assert "estimated cost per 1,000 postings" in capsys.readouterr().out
    assert main(["estimate", "--data", str(DATA / "eval.jsonl"), "--model", "unknown-model"]) == 1


def test_missing_file_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["extract", "does-not-exist.txt"]) == 2
    assert "jobspec:" in capsys.readouterr().err


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2


def test_eval_csv_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = str(DATA / "eval.jsonl")
    assert main(["eval", "--data", data, "--format", "csv", "--limit", "3", "--errors"]) == 0
    captured = capsys.readouterr()
    rows = list(csv.reader(io.StringIO(captured.out)))
    assert rows[0] == ["field", "accuracy", "precision", "recall", "f1", "n", "gold_count", "pred_count", "tp"]
    assert [r[0] for r in rows[1:]] == list(EVAL_FIELDS)
    assert all(r[5] == "3" for r in rows[1:])
    assert "mismatches:" in captured.err and "macro accuracy" not in captured.out
    report = tmp_path / "r.csv"
    assert main(["eval", "--data", data, "--limit", "3", "--report", str(report)]) == 0
    assert report.read_text(encoding="utf-8") == captured.out
