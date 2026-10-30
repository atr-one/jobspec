# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
