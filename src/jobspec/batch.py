"""Batch extraction: many postings through one extractor, streamed, in input order.

``extract_many`` takes any iterable of postings and yields one ``BatchResult`` per posting in
the order of the input, while up to ``workers`` postings are extracted at the same time in a
thread pool. Nothing is read ahead beyond a small window, so a file of any size can be streamed.

With ``max_cost`` the run stops before a posting whose cost could take the total past the limit.
The cost of a call is not known before it is made, so the most expensive posting seen so far is
used as the estimate for each posting still to be sent (postings already in flight count too).
Until the first LLM call has finished, only one posting is in flight.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import IO, Any, Deque, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

from .base import ExtractionResult, Extractor, Usage
from .schema import JobSpec


@dataclass
class BatchInput:
    """One posting to extract. ``error`` marks an input line that could not be read."""

    id: Optional[str]
    text: Optional[str]
    html: Optional[bool] = None
    line: Optional[int] = None
    error: Optional[str] = None


@dataclass
class BatchResult:
    """The outcome for one input posting."""

    index: int
    id: Optional[str]
    line: Optional[int]
    result: Optional[ExtractionResult]
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        """The JSON object written per line by ``jobspec batch``."""
        out: Dict[str, Any] = {"id": self.id, "line": self.line}
        if self.result is None:
            out.update({"record": None, "meta": {}, "errors": list(self.errors), "warnings": []})
            return out
        full = self.result.spec.to_dict(include_meta=True)
        meta = full.pop("meta")
        usage = self.result.usage
        out.update(
            {
                "record": full,
                "meta": meta,
                "errors": list(self.errors),
                "warnings": list(self.result.warnings),
                "usage": {
                    "calls": usage.calls,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cost_usd": usage.cost_usd,
                },
                "latency_ms": round(self.result.latency_s * 1000, 1),
            }
        )
        return out


PostingLike = Union[str, Tuple[Any, str], BatchInput]


def _as_input(raw: PostingLike, index: int) -> BatchInput:
    if isinstance(raw, BatchInput):
        return raw
    if isinstance(raw, str):
        return BatchInput(id=str(index), text=raw)
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], str):
        return BatchInput(id=None if raw[0] is None else str(raw[0]), text=raw[1])
    return BatchInput(id=str(index), text=None, error=f"unsupported input item of type {type(raw).__name__}")


class BatchRun:
    """Iterate over it to run the batch. Totals are filled in as results are yielded.

    ``stopped`` is set when the run ended before the end of the input because of ``max_cost``;
    ``stop_reason`` says why.
    """

    def __init__(
        self,
        items: Iterable[PostingLike],
        extractor: Extractor,
        workers: int = 4,
        max_cost: Optional[float] = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if max_cost is not None and max_cost <= 0:
            raise ValueError("max_cost must be positive")
        self._items = items
        self.extractor = extractor
        self.workers = workers
        self.max_cost = max_cost
        self.count = 0
        self.failures = 0
        self.usage = Usage()
        self.elapsed_s = 0.0
        self.stopped = False
        self.stop_reason: Optional[str] = None
        self._lock = threading.Lock()
        self._running = 0
        self._spent = 0.0
        self._estimate: Optional[float] = None
        self._unknown_cost = False

    # budget ---------------------------------------------------------------------------------

    def _settle(self, result: ExtractionResult) -> None:
        with self._lock:
            self._running -= 1
            if not result.usage.calls:
                return
            cost = result.usage.cost_usd
            if cost is None:
                self._unknown_cost = True
                return
            self._spent += cost
            if self._estimate is None or cost > self._estimate:
                self._estimate = cost

    def _can_start(self) -> bool:
        if self.max_cost is None:
            return True
        with self._lock:
            if self._unknown_cost:
                return False
            if self._estimate is None:
                return self._running == 0
            return self._spent + (self._running + 1) * self._estimate <= self.max_cost

    def _why_stopped(self) -> str:
        if self._unknown_cost:
            return "the cost of a call is unknown (no price for this model), so max_cost cannot be enforced"
        limit = self.max_cost or 0.0
        return f"the next posting could take the total cost past {limit:.4f} USD (spent {self._spent:.4f} USD)"

    # work -----------------------------------------------------------------------------------

    def _work(self, item: BatchInput) -> ExtractionResult:
        try:
            result = self.extractor.extract(item.text or "", item.html)
        except Exception as exc:  # one bad posting must not end the batch
            result = ExtractionResult(spec=JobSpec(), error=f"{type(exc).__name__}: {exc}")
        self._settle(result)
        return result

    def _submit(self, pool: Optional[ThreadPoolExecutor], item: BatchInput) -> Future[ExtractionResult]:
        with self._lock:
            self._running += 1
        if pool is not None:
            return pool.submit(self._work, item)
        future: Future[ExtractionResult] = Future()
        future.set_result(self._work(item))
        return future

    def _finish(self, index: int, item: BatchInput, future: Optional[Future[ExtractionResult]]) -> BatchResult:
        if future is None:
            out = BatchResult(index, item.id, item.line, None, [item.error or "unreadable input"])
        else:
            result = future.result()
            errors = [result.error] if result.error else []
            out = BatchResult(index, item.id, item.line, result, errors)
            self.usage.add(result.usage)
        self.count += 1
        if out.errors:
            self.failures += 1
        return out

    def __iter__(self) -> Iterator[BatchResult]:
        started = time.perf_counter()
        pool = ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None
        window = self.workers * 2
        pending: Deque[Tuple[int, BatchInput, Optional[Future[ExtractionResult]]]] = deque()

        def head_done() -> bool:
            future = pending[0][2]
            return future is None or future.done()

        try:
            for index, raw in enumerate(self._items):
                item = _as_input(raw, index)
                if item.error is not None or item.text is None:
                    if item.error is None:
                        item.error = "no posting text"
                    pending.append((index, item, None))
                    while pending and head_done():
                        yield self._finish(*pending.popleft())
                    continue
                while True:
                    while pending and head_done():
                        yield self._finish(*pending.popleft())
                    if len(pending) < window and self._can_start():
                        break
                    running = [f for _, _, f in pending if f is not None and not f.done()]
                    if not running:
                        if len(pending) < window:
                            self.stopped = True
                            self.stop_reason = self._why_stopped()
                        break
                    wait(running, return_when=FIRST_COMPLETED)
                if self.stopped:
                    break
                pending.append((index, item, self._submit(pool, item)))
            while pending:
                yield self._finish(*pending.popleft())
        finally:
            for _, _, future in pending:
                if future is not None:
                    future.cancel()
            if pool is not None:
                pool.shutdown(wait=True)
            self.elapsed_s = time.perf_counter() - started


def extract_many(
    items: Iterable[PostingLike],
    extractor: Extractor,
    workers: int = 4,
    max_cost: Optional[float] = None,
) -> BatchRun:
    """Extract every posting in ``items`` and yield ``BatchResult`` objects in input order.

    An item is a posting text (its position becomes the id), an ``(id, text)`` pair or a
    ``BatchInput``. ``workers`` postings are extracted at the same time (use 1 for the rule
    extractor: it is CPU-bound and gains nothing from threads). With ``max_cost`` (USD) the run
    stops before a posting that could take the total past the limit; check ``run.stopped``.

    >>> from jobspec import RuleExtractor, extract_many
    >>> run = extract_many(["Senior Go Engineer\\nRemote (US)"], RuleExtractor(), workers=1)
    >>> [r.result.spec.seniority for r in run], run.count
    (['senior'], 1)
    """
    return BatchRun(items, extractor, workers=workers, max_cost=max_cost)


# --- JSONL input and output -------------------------------------------------------------------


def read_jsonl(
    lines: Iterable[str],
    text_field: str = "text",
    id_field: str = "id",
    skip_ids: Optional[Set[str]] = None,
    skip_lines: Optional[Set[int]] = None,
) -> Iterator[BatchInput]:
    """Turn JSONL lines into ``BatchInput`` objects one at a time.

    A line that is not a JSON object, or has no string in ``text_field``, becomes an input with
    ``error`` set instead of raising. Ids are strings; a row without an id gets ``id`` None and is
    identified by its line number. Rows whose id is in ``skip_ids`` (or, without an id, whose line
    is in ``skip_lines``) are skipped.
    """
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            if skip_lines and number in skip_lines:
                continue
            yield BatchInput(id=None, text=None, line=number, error=f"line {number}: invalid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            if skip_lines and number in skip_lines:
                continue
            yield BatchInput(id=None, text=None, line=number, error=f"line {number}: not a JSON object")
            continue
        raw_id = row.get(id_field)
        item_id = None if raw_id is None else str(raw_id)
        if item_id is not None and skip_ids and item_id in skip_ids:
            continue
        if item_id is None and skip_lines and number in skip_lines:
            continue
        text = row.get(text_field)
        if not isinstance(text, str):
            yield BatchInput(id=item_id, text=None, line=number, error=f"line {number}: no string field {text_field!r}")
            continue
        fmt = row.get("format")
        html = True if fmt == "html" else False if fmt == "text" else None
        yield BatchInput(id=item_id, text=text, html=html, line=number)


def done_in_output(path: str) -> Tuple[Set[str], Set[int]]:
    """Ids (and line numbers of id-less rows) already written to a batch output file.

    A last line cut off by a crash is removed from the file, so that the run can append after it.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except FileNotFoundError:
        return set(), set()
    if data and not data.endswith(b"\n"):
        keep = data.rfind(b"\n") + 1
        with open(path, "r+b") as fh:
            fh.truncate(keep)
        data = data[:keep]
    ids: Set[str] = set()
    lines: Set[int] = set()
    for raw in data.decode("utf-8").splitlines():
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("id") is not None:
            ids.add(str(row["id"]))
        elif isinstance(row.get("line"), int):
            lines.add(row["line"])
    return ids, lines


def write_jsonl(results: Iterable[BatchResult], out: IO[str]) -> None:
    """Write one JSON object per result and flush after each, so a crash loses little."""
    for item in results:
        out.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        out.flush()
