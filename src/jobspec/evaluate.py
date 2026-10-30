"""Evaluation harness: compare an extractor's output with labeled postings, field by field.

Scalar fields (enums, booleans, numbers, strings) are compared after light normalization;
salary amounts within a relative tolerance; list fields (tech stack, locations, regions) as sets
with precision / recall / F1. Accuracy counts an agreed "not stated" (both empty) as correct;
precision and recall look only at stated values.
"""

from __future__ import annotations

import json
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from .base import Extractor, Usage
from .rules.stack import canonical_tech
from .schema import FIELDS, LIST_FIELDS, JobSpec, SchemaError

SALARY_SUBFIELDS: Tuple[str, ...] = ("min", "max", "currency", "period", "gross_net")
EVAL_FIELDS: Tuple[str, ...] = (
    "title",
    "seniority",
    "employment_type",
    "work_mode",
    "locations",
    "remote_regions",
    "salary",
    "salary.min",
    "salary.max",
    "salary.currency",
    "salary.period",
    "salary.gross_net",
    "experience_years_min",
    "tech_stack",
    "visa_sponsorship",
    "apply_url",
    "apply_email",
)
SALARY_TOLERANCE = 0.02


@dataclass
class Example:
    id: str
    text: str
    label: JobSpec
    html: Optional[bool] = None
    notes: str = ""


def load_dataset(path: str) -> List[Example]:
    """Read a JSONL file of ``{"id", "text", "label", "format"?, "notes"?}`` objects."""
    examples: List[Example] = []
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                row = json.loads(line)
                label = JobSpec.from_dict(row["label"])
            except (ValueError, KeyError, SchemaError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            fmt = row.get("format")
            html = True if fmt == "html" else False if fmt == "text" else None
            examples.append(
                Example(
                    id=str(row.get("id", number)),
                    text=row["text"],
                    label=label,
                    html=html,
                    notes=row.get("notes", ""),
                )
            )
    return examples


# --- normalization ----------------------------------------------------------------------------


def _fold(s: str) -> str:
    decomposed = unicodedata.normalize("NFKD", s)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(plain.casefold().split())


def _norm_title(s: str) -> str:
    return _fold(s).strip(" .,:;-|")


def _norm_location(s: str) -> str:
    return _fold(s.split(",")[0]).strip(" .")


def _norm_url(s: str) -> str:
    s = s.strip().rstrip("/")
    if "://" in s:
        scheme, rest = s.split("://", 1)
        host, _, path = rest.partition("/")
        s = scheme.lower() + "://" + host.lower() + ("/" + path if path else "")
    return s


def field_value(spec: JobSpec, name: str) -> Any:
    if name.startswith("salary."):
        sub = name.split(".", 1)[1]
        return getattr(spec.salary, sub) if spec.salary is not None else None
    return spec.get(name)


def normalize_set(name: str, values: Iterable[str]) -> Set[str]:
    if name == "locations":
        return {_norm_location(v) for v in values if v.strip()}
    if name == "remote_regions":
        return {v.strip().upper() for v in values if v.strip()}
    if name == "tech_stack":
        return {canonical_tech(v).casefold() for v in values if v.strip()}
    return {_fold(v) for v in values}


def _amount_equal(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= SALARY_TOLERANCE * max(abs(a), abs(b), 1.0)


def scalar_equal(name: str, gold: Any, pred: Any) -> bool:
    if name == "salary":
        g = gold if gold is not None and not gold.is_empty() else None
        p = pred if pred is not None and not pred.is_empty() else None
        if g is None or p is None:
            return g is None and p is None
        return all(scalar_equal("salary." + sub, getattr(g, sub), getattr(p, sub)) for sub in SALARY_SUBFIELDS)
    if gold is None or pred is None:
        return gold is None and pred is None
    if name in ("salary.min", "salary.max"):
        return _amount_equal(gold, pred)
    if name == "title":
        return bool(_norm_title(gold) == _norm_title(pred))
    if name == "apply_url":
        return bool(_norm_url(gold) == _norm_url(pred))
    if name == "apply_email":
        return bool(gold.strip().casefold() == pred.strip().casefold())
    return bool(gold == pred)


# --- scores -----------------------------------------------------------------------------------


@dataclass
class FieldScore:
    name: str
    kind: str  # "scalar" or "set"
    n: int = 0
    correct: int = 0
    tp: int = 0
    pred_count: int = 0
    gold_count: int = 0

    @property
    def accuracy(self) -> Optional[float]:
        return self.correct / self.n if self.n else None

    @property
    def precision(self) -> Optional[float]:
        return self.tp / self.pred_count if self.pred_count else None

    @property
    def recall(self) -> Optional[float]:
        return self.tp / self.gold_count if self.gold_count else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None and r is None:
            return None
        p, r = p or 0.0, r or 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "n": self.n,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "gold_count": self.gold_count,
            "pred_count": self.pred_count,
            "tp": self.tp,
        }


@dataclass
class ItemResult:
    id: str
    latency_s: float
    usage: Usage
    error: Optional[str]
    warnings: List[str]
    mismatches: Dict[str, Dict[str, Any]]
    prediction: Dict[str, Any]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


@dataclass
class EvalReport:
    extractor: str
    data_path: str
    fields: Dict[str, FieldScore]
    items: List[ItemResult] = field(default_factory=list)
    wall_s: float = 0.0
    model: Optional[str] = None

    # totals ---------------------------------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def errors(self) -> int:
        return sum(1 for i in self.items if i.error)

    @property
    def usage(self) -> Usage:
        total = Usage()
        for item in self.items:
            total.add(item.usage)
        return total

    @property
    def avg_latency_s(self) -> float:
        return sum(i.latency_s for i in self.items) / self.n if self.n else 0.0

    @property
    def max_latency_s(self) -> float:
        return max((i.latency_s for i in self.items), default=0.0)

    def macro(self, metric: str) -> Optional[float]:
        values = [getattr(self.fields[f], metric) for f in FIELDS]
        defined = [v for v in values if v is not None]
        return sum(defined) / len(defined) if defined else None

    @property
    def cost_per_1000(self) -> Optional[float]:
        cost = self.usage.cost_usd
        if cost is None or not self.n:
            return None
        return cost / self.n * 1000

    # output ---------------------------------------------------------------------------------

    def rows(self) -> List[Tuple[str, str, str, str, str, str]]:
        def fmt(v: Optional[float]) -> str:
            return "-" if v is None else f"{v:.2f}"

        out = []
        for name in EVAL_FIELDS:
            s = self.fields[name]
            out.append((name, fmt(s.accuracy), fmt(s.precision), fmt(s.recall), fmt(s.f1), str(s.gold_count)))
        return out

    def totals(self) -> List[Tuple[str, str]]:
        usage = self.usage

        def money(v: Optional[float]) -> str:
            return "n/a (no price for this model)" if v is None else f"${v:.4f}"

        macro_acc, macro_f1 = self.macro("accuracy"), self.macro("f1")
        out = [
            ("postings", str(self.n)),
            ("errors", str(self.errors)),
            ("macro accuracy (12 fields)", "-" if macro_acc is None else f"{macro_acc:.3f}"),
            ("macro F1 (12 fields)", "-" if macro_f1 is None else f"{macro_f1:.3f}"),
            ("avg latency", "%.1f ms" % (self.avg_latency_s * 1000)),
            ("max latency", "%.1f ms" % (self.max_latency_s * 1000)),
        ]
        if usage.calls:
            out += [
                ("LLM calls", str(usage.calls)),
                ("input / output tokens", f"{usage.input_tokens} / {usage.output_tokens}"),
                ("total cost", money(usage.cost_usd)),
                ("cost per 1,000 postings", money(self.cost_per_1000)),
            ]
        else:
            out.append(("total cost", "$0 (no LLM calls)"))
        return out

    def to_text(self) -> str:
        head = ("field", "acc", "prec", "rec", "f1", "gold")
        widths = [24, 6, 6, 6, 6, 5]
        lines = [
            "extractor: {}{}   data: {}".format(
                self.extractor, f" ({self.model})" if self.model else "", self.data_path
            )
        ]
        lines.append("".join(h.ljust(w) if i == 0 else h.rjust(w) for i, (h, w) in enumerate(zip(head, widths))))
        lines.append("-" * sum(widths))
        for row in self.rows():
            lines.append("".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths))))
        lines.append("-" * sum(widths))
        for key, value in self.totals():
            lines.append(f"{key:<28} {value}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        title = "Extractor `{}`{} on `{}`".format(
            self.extractor,
            f" (`{self.model}`)" if self.model else "",
            self.data_path,
        )
        lines = [
            "### " + title,
            "",
            "| field | accuracy | precision | recall | F1 | gold values |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in self.rows():
            lines.append("| {} |".format(" | ".join(row)))
        lines += ["", "| total | value |", "|---|---:|"]
        for key, value in self.totals():
            lines.append(f"| {key} | {value} |")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        usage = self.usage
        return {
            "extractor": self.extractor,
            "model": self.model,
            "data": self.data_path,
            "postings": self.n,
            "errors": self.errors,
            "macro_accuracy": self.macro("accuracy"),
            "macro_f1": self.macro("f1"),
            "avg_latency_ms": self.avg_latency_s * 1000,
            "max_latency_ms": self.max_latency_s * 1000,
            "llm_calls": usage.calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": usage.cost_usd,
            "cost_per_1000_postings_usd": self.cost_per_1000,
            "fields": {name: self.fields[name].to_dict() for name in EVAL_FIELDS},
            "items": [
                {
                    "id": i.id,
                    "latency_ms": i.latency_s * 1000,
                    "input_tokens": i.usage.input_tokens,
                    "output_tokens": i.usage.output_tokens,
                    "cost_usd": i.usage.cost_usd,
                    "error": i.error,
                    "warnings": i.warnings,
                    "mismatches": i.mismatches,
                    "prediction": i.prediction,
                }
                for i in self.items
            ],
        }


def score_item(gold: JobSpec, pred: JobSpec, scores: Dict[str, FieldScore]) -> Dict[str, Dict[str, Any]]:
    """Add one example to ``scores``; return the fields that differ."""
    mismatches: Dict[str, Dict[str, Any]] = {}
    for name in EVAL_FIELDS:
        s = scores[name]
        s.n += 1
        g, p = field_value(gold, name), field_value(pred, name)
        if name in LIST_FIELDS:
            gs, ps = normalize_set(name, g), normalize_set(name, p)
            s.tp += len(gs & ps)
            s.gold_count += len(gs)
            s.pred_count += len(ps)
            if gs == ps:
                s.correct += 1
            else:
                mismatches[name] = {"gold": sorted(gs), "pred": sorted(ps)}
            continue
        if name == "salary":
            g = g if g is not None and not g.is_empty() else None
            p = p if p is not None and not p.is_empty() else None
        equal = scalar_equal(name, g, p)
        if g is not None:
            s.gold_count += 1
        if p is not None:
            s.pred_count += 1
        if equal:
            s.correct += 1
            if g is not None:
                s.tp += 1
        elif not name.startswith("salary."):
            mismatches[name] = {"gold": _jsonable(g), "pred": _jsonable(p)}
    return mismatches


def evaluate(
    extractor: Extractor,
    examples: List[Example],
    data_path: str = "",
    progress: Optional[Callable[[int, int, ItemResult], None]] = None,
) -> EvalReport:
    scores = {name: FieldScore(name, "set" if name in LIST_FIELDS else "scalar") for name in EVAL_FIELDS}
    report = EvalReport(extractor=extractor.name, data_path=data_path, fields=scores)
    provider = getattr(extractor, "provider", None) or getattr(getattr(extractor, "llm", None), "provider", None)
    report.model = getattr(provider, "model", None)
    started = time.perf_counter()
    for index, example in enumerate(examples, 1):
        result = extractor.extract(example.text, example.html)
        mismatches = score_item(example.label, result.spec, scores)
        item = ItemResult(
            id=example.id,
            latency_s=result.latency_s,
            usage=result.usage,
            error=result.error,
            warnings=result.warnings,
            mismatches=mismatches,
            prediction=result.spec.to_dict(include_meta=True),
        )
        report.items.append(item)
        if progress:
            progress(index, len(examples), item)
    report.wall_s = time.perf_counter() - started
    return report
