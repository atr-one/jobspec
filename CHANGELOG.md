# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-30

### Added

- `jobspec batch`: extract a JSONL file of postings into a JSONL file of records (id, record,
  per-field meta, errors, warnings, usage), streamed line by line and written in input order.
  `--text-field`, `--id-field`, `--workers` (thread pool for `llm` and `hybrid`, default 4),
  `--max-cost` (stops before the total LLM cost could pass the limit), `--resume` (skips ids
  already in the output file), and a final summary of records, failures, tokens, cost and elapsed
  time. Malformed input lines are written as error records and do not stop the run.
- `extract_many(items, extractor, workers=..., max_cost=...)` and `BatchInput`, `BatchResult`,
  `BatchRun` for batch extraction from code.
- `jobspec eval --format csv` prints one CSV row per field; `--report file.csv` writes the same.
- A new held-out set, `data/holdout.jsonl` (12 postings: English from six markets including an
  HTML posting, French, German, Spanish, Portuguese and Russian), written without running the
  rules on it and evaluated once with the final rules: macro accuracy 0.917.

### Changed

- The 0.1.0 `data/holdout.jsonl` is renamed to `data/dev.jsonl`: the rule fixes below were made by
  looking at its misses, so it is no longer a held-out set. The README now reports three tables
  (eval in-sample, dev, new holdout) and says which sets the rules were tuned on.
- Rules: hybrid work given only as office weekdays ("in the office Tuesday to Thursday"), "N or
  more years" / "N years or more", PJ as a contract (Brazil), work-permit sponsorship worded without
  the word "visa", "B2B" as a market ("a B2B SaaS company") no longer read as a B2B contract,
  "innerhalb Deutschlands" as a remote region, `Ciudad:` / `City:` / `Contratação:` labels.
- Locations without a "Location:" label are read from short header lines that start with a known
  city ("Dublin, Ireland - on-site", "Hyderabad" in a pipe-separated title) and from prose such as
  "from our Boston office" or "híbrido em Belo Horizonte". The city list grows from 122 to 306
  names.
- `jobspec eval --report` files are written with LF line endings on every platform.

## [0.1.0] - 2026-10-30

### Added

- `JobSpec` schema with `to_dict()` / `from_dict()`, validation, per-field provenance
  (`source`, `confidence`) and a JSON Schema export (`jobspec schema`).
- Rule extractor (offline, deterministic) for title, seniority, employment type, work mode,
  locations, remote regions, salary (range, currency, period, gross/net; k, lakh/LPA, Indian
  digit grouping, dot and space thousands), years of experience, tech stack (about 160
  technologies with aliases and guards for "Go", "R", "C", "Swift", "Spring" and similar),
  visa sponsorship and application URL/email. English, German, Spanish, Portuguese, Russian and
  Hindi-English postings.
- LLM extractor over the Anthropic Messages API and any OpenAI-compatible chat completions
  endpoint, using only `urllib`. Nonce-delimited untrusted input, lenient JSON parsing with
  repair, field-by-field validation, token usage and cost from an editable price table.
- Hybrid extractor: rules first, one LLM call for the fields the rules left empty or unsure.
- Evaluation harness (`jobspec eval`) with per-field accuracy, precision, recall and F1, salary
  tolerance, set metrics for lists, latency, tokens and cost, and JSON / Markdown reports.
- `jobspec estimate` for an offline cost estimate per 1,000 postings.
- 30 labeled synthetic postings (`data/eval.jsonl`) and a 12-posting held-out set
  (`data/holdout.jsonl`).

[0.2.0]: https://github.com/atr-one/jobspec/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/atr-one/jobspec/releases/tag/v0.1.0
