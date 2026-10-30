"""Command line: ``jobspec extract``, ``batch``, ``eval``, ``schema`` and ``estimate``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import IO, Iterator, List, Optional

from . import __version__
from .base import Extractor
from .batch import BatchResult, done_in_output, extract_many, read_jsonl, write_jsonl
from .evaluate import evaluate, load_dataset
from .hybrid import HybridExtractor
from .llm import LLMExtractor, build_prompt
from .pricing import estimate_tokens, load_prices, price_for
from .providers import make_provider
from .rules import RuleExtractor
from .schema import json_schema


def _add_llm_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("LLM options (llm and hybrid extractors)")
    g.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    g.add_argument("--model", help="model name (default: claude-haiku-5-5 / gpt-4.1-mini)")
    g.add_argument("--base-url", help="API base URL, e.g. a local OpenAI-compatible server")
    g.add_argument(
        "--api-key-env",
        help="name of the environment variable holding the key (default: ANTHROPIC_API_KEY or OPENAI_API_KEY)",
    )
    g.add_argument("--max-tokens", type=int, default=1024, help="output token limit per call")
    g.add_argument("--timeout", type=float, default=60.0, help="seconds per HTTP request")
    g.add_argument("--prices", help="JSON file of model -> [input, output] USD per 1M tokens")
    g.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="hybrid: ask the LLM for fields whose rule confidence is below this (default 0.6)",
    )


def _make_extractor(args: argparse.Namespace) -> Extractor:
    if getattr(args, "prices", None):
        load_prices(args.prices)
    if args.extractor == "rules":
        return RuleExtractor()
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    provider = make_provider(args.provider, args.model, args.base_url, api_key, args.timeout)
    llm = LLMExtractor(provider, max_tokens=args.max_tokens)
    if args.extractor == "llm":
        return llm
    return HybridExtractor(llm, threshold=args.threshold)


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def cmd_extract(args: argparse.Namespace) -> int:
    extractor = _make_extractor(args)
    html = True if args.html else None
    result = extractor.extract(_read(args.file), html)
    out = result.spec.to_dict(include_meta=args.meta)
    if args.meta:
        out["_run"] = {
            "extractor": extractor.name,
            "latency_ms": round(result.latency_s * 1000, 1),
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "cost_usd": result.usage.cost_usd,
            "warnings": result.warnings,
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
        return 1
    return 0


def _money(value: Optional[float]) -> str:
    return "n/a (no price for this model)" if value is None else f"${value:.4f}"


def cmd_batch(args: argparse.Namespace) -> int:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.max_cost is not None and args.max_cost <= 0:
        raise ValueError("--max-cost must be positive")
    skip_ids, skip_lines = done_in_output(args.out) if args.resume else (set(), set())
    extractor = _make_extractor(args)
    workers = 1 if extractor.name == "rules" else args.workers
    source: IO[str] = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")  # noqa: SIM115
    interrupted = False
    try:
        inputs = read_jsonl(source, args.text_field, args.id_field, skip_ids, skip_lines)
        run = extract_many(inputs, extractor, workers=workers, max_cost=args.max_cost)
        results: Iterator[BatchResult] = iter(run)
        with open(args.out, "a" if args.resume else "w", encoding="utf-8", newline="\n") as out:
            try:
                write_jsonl(results, out)
            except KeyboardInterrupt:
                interrupted = True
                close = getattr(results, "close", None)
                if close is not None:
                    close()
    finally:
        if source is not sys.stdin:
            source.close()
    usage = run.usage
    print(
        "records written: {} (failed: {}{})".format(
            run.count,
            run.failures,
            f", already done: {len(skip_ids) + len(skip_lines)}" if args.resume else "",
        ),
        file=sys.stderr,
    )
    if usage.calls:
        print(
            f"LLM calls: {usage.calls}   tokens in / out: {usage.input_tokens} / {usage.output_tokens}"
            f"   cost: {_money(usage.cost_usd)}",
            file=sys.stderr,
        )
    else:
        print("LLM calls: 0   cost: $0", file=sys.stderr)
    print(f"elapsed: {run.elapsed_s:.1f} s", file=sys.stderr)
    if interrupted:
        print("interrupted: run again with --resume to continue", file=sys.stderr)
        return 130
    if run.stopped:
        print(f"stopped before the end of the input: {run.stop_reason}", file=sys.stderr)
        print("run again with --resume (and a higher --max-cost) to continue", file=sys.stderr)
        return 3
    return 1 if run.failures else 0


def cmd_eval(args: argparse.Namespace) -> int:
    examples = load_dataset(args.data)
    if args.limit:
        examples = examples[: args.limit]
    extractor = _make_extractor(args)

    def progress(i: int, total: int, item: object) -> None:
        if extractor.name != "rules":
            print(f"\r{i}/{total}", end="", file=sys.stderr, flush=True)

    report = evaluate(extractor, examples, args.data, progress)
    if extractor.name != "rules":
        print("", file=sys.stderr)
    print(report.to_text())
    if args.errors:
        print("\nmismatches:")
        for item in report.items:
            for name, diff in item.mismatches.items():
                gold = json.dumps(diff["gold"], ensure_ascii=False)
                pred = json.dumps(diff["pred"], ensure_ascii=False)
                print(f"  {item.id:<8} {name:<22} gold={gold} pred={pred}")
            if item.error:
                print(f"  {item.id:<8} error: {item.error}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            if args.report.lower().endswith(".json"):
                json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)
            else:
                fh.write(report.to_markdown())
        print(f"\nreport written to {args.report}")
    if args.fail_under is not None:
        macro = report.macro("accuracy") or 0.0
        if macro < args.fail_under:
            print(f"macro accuracy {macro:.3f} is below {args.fail_under:.3f}", file=sys.stderr)
            return 1
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    print(json.dumps(json_schema(), indent=2))
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    """Estimate LLM cost per 1,000 postings without calling any API."""
    if args.prices:
        load_prices(args.prices)
    examples = load_dataset(args.data)
    system, _ = build_prompt("", "000000000000")
    overhead = estimate_tokens(system) + 40
    posting_tokens = [estimate_tokens(build_prompt(e.text, "000000000000")[1]) for e in examples]
    avg_in = overhead + sum(posting_tokens) / max(len(posting_tokens), 1)
    price = price_for(args.model)
    print(f"postings: {len(examples)}   model: {args.model}")
    print(f"estimated input tokens per posting:  {avg_in:.0f} (system prompt ~{overhead})")
    print(f"assumed output tokens per posting:   {args.output_tokens}")
    if price is None:
        print(f"no price for {args.model}: pass --prices FILE")
        return 1
    per_1000 = (avg_in * price[0] + args.output_tokens * price[1]) / 1_000_000 * 1000
    print(
        "prices used (USD per 1M tokens):     input {:.2f}, output {:.2f} (examples - check your provider)".format(
            *price
        )
    )
    print(f"estimated cost per 1,000 postings:   ${per_1000:.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jobspec",
        description="Turn job postings into structured records and evaluate extractors.",
    )
    parser.add_argument("--version", action="version", version="jobspec " + __version__)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("extract", help="extract one posting (file or - for stdin) to JSON")
    p.add_argument("file")
    p.add_argument("--extractor", choices=["rules", "llm", "hybrid"], default="rules")
    p.add_argument("--html", action="store_true", help="treat input as HTML (detected by default)")
    p.add_argument("--meta", action="store_true", help="include per-field source/confidence and run info")
    _add_llm_options(p)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("batch", help="extract a JSONL file of postings to a JSONL file of records")
    p.add_argument("--in", dest="input", required=True, help="JSONL with one posting per line (- for stdin)")
    p.add_argument("--out", required=True, help="output JSONL: one record per input line, in input order")
    p.add_argument("--extractor", choices=["rules", "llm", "hybrid"], default="rules")
    p.add_argument("--text-field", default="text", help="key of the posting text (default: text)")
    p.add_argument("--id-field", default="id", help="key of the posting id (default: id)")
    p.add_argument("--workers", type=int, default=4, help="parallel LLM calls (default 4; rules run in one thread)")
    p.add_argument("--max-cost", type=float, help="stop before the total LLM cost could exceed this (USD)")
    p.add_argument("--resume", action="store_true", help="append to --out, skipping ids already in it")
    _add_llm_options(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("eval", help="score an extractor against labeled JSONL")
    p.add_argument("--data", required=True, help="JSONL with id, text, label")
    p.add_argument("--extractor", choices=["rules", "llm", "hybrid"], default="rules")
    p.add_argument("--limit", type=int, help="only the first N postings")
    p.add_argument("--report", help="write a report: .json or .md")
    p.add_argument("--errors", action="store_true", help="list every mismatching field")
    p.add_argument("--fail-under", type=float, help="exit 1 if macro accuracy is below this")
    _add_llm_options(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("schema", help="print the JSON Schema of the output")
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("estimate", help="estimate LLM cost per 1,000 postings (no API calls)")
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--output-tokens", type=int, default=250, help="assumed reply size (default 250)")
    p.add_argument("--prices", help="JSON file of model -> [input, output] USD per 1M tokens")
    p.set_defaults(func=cmd_estimate)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    try:
        code: int = args.func(args)
    except (OSError, ValueError) as exc:
        print(f"jobspec: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
