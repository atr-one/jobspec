# jobspec

jobspec turns a raw job posting (plain text or HTML) into a structured, validated record: title,
seniority, employment type, work mode, locations, remote regions, salary (range, currency, period,
gross/net), years of experience, tech stack, visa sponsorship and where to apply.

It ships three extractors behind one interface, plus an evaluation harness that compares them on
labeled postings:

- **rules**: regular expressions and a curated vocabulary. Offline, deterministic, no API key,
  a few milliseconds per posting.
- **llm**: the same schema through a language model with JSON output. It works with the Anthropic
  Messages API and with any OpenAI-compatible chat completions endpoint.
- **hybrid**: rules first, then one LLM call for the fields the rules left empty or were unsure about.

The evaluation harness is the main reason the package exists. Before you put a model into a
parsing pipeline that will see millions of postings, you want to know, field by field, how
accurate it is, how slow it is and what it costs, compared with a free baseline.

jobspec has no runtime dependencies. HTTP calls use `urllib.request` from the standard library.

## Install

```bash
pip install "git+https://github.com/atr-one/jobspec"   # the jobspec command and package
pip install -e ".[dev]"                                # from a clone, with pytest, ruff and mypy
```

The distribution is named `jobspec-ai` (the name `jobspec` on PyPI belongs to another project); it is not on PyPI yet. The import name and the command are `jobspec`.

Python 3.9 or newer.

## Command line

Extract one posting:

```bash
jobspec extract posting.txt --extractor rules
jobspec extract posting.html --extractor rules --meta     # add per-field source/confidence
cat posting.txt | jobspec extract - --extractor llm        # uses ANTHROPIC_API_KEY
```

```json
{
  "title": "Senior Java Developer",
  "seniority": "senior",
  "employment_type": "contract",
  "work_mode": "remote",
  "locations": [],
  "remote_regions": ["PL"],
  "salary": {"min": 20000, "max": 26000, "currency": "PLN", "period": "month", "gross_net": "net"},
  "experience_years_min": 5,
  "tech_stack": ["Java", "Spring Boot", "Kafka", "PostgreSQL"],
  "visa_sponsorship": null,
  "apply_url": null,
  "apply_email": "jobs@kestrel-payments.example"
}
```

Evaluate an extractor against labeled data:

```bash
jobspec eval --data data/eval.jsonl --extractor rules
jobspec eval --data data/eval.jsonl --extractor rules --errors          # list every mismatch
jobspec eval --data data/eval.jsonl --extractor llm --report llm.md     # or .json
jobspec eval --data data/eval.jsonl --extractor llm --provider openai --model gpt-4.1-mini
jobspec eval --data data/eval.jsonl --extractor llm --provider openai \
    --base-url http://localhost:11434/v1 --model llama3.1
```

`OPENAI_API_KEY` is sent only to `api.openai.com`. For any other `--base-url`, name the variable
that holds that endpoint's key with `--api-key-env MY_GATEWAY_KEY`; without it no key is sent.

Other commands:

```bash
jobspec schema                                          # JSON Schema of the output
jobspec estimate --data data/eval.jsonl --model claude-haiku-5-5   # offline cost estimate
```

LLM options: `--provider anthropic|openai`, `--model`, `--base-url`, `--api-key-env` (name of
the environment variable with the key; defaults are `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`),
`--max-tokens`, `--timeout`, `--prices prices.json`, and `--threshold` for the hybrid extractor.
`--fail-under 0.9` makes `eval` exit with status 1 below a macro accuracy, for CI.

## Batch mode

`jobspec batch` extracts a JSONL file of postings into a JSONL file of records:

```bash
jobspec batch --in postings.jsonl --out records.jsonl --extractor rules
jobspec batch --in postings.jsonl --out records.jsonl --extractor hybrid --workers 8 --max-cost 5
jobspec batch --in postings.jsonl --out records.jsonl --extractor hybrid --workers 8 --max-cost 10 --resume
```

- **Input** is read one line at a time, so the file can be larger than memory (`--in -` reads
  stdin). Each line is a JSON object; the posting text is in `--text-field` (default `text`) and
  its id in `--id-field` (default `id`). An optional `"format": "html"` or `"text"` is honored.
- **Output** has one line per input line, in input order:

  ```json
  {"id": "p17", "line": 17, "record": {...JobSpec...}, "meta": {"salary": {"source": "rule", "confidence": 0.95}},
   "errors": [], "warnings": [], "usage": {"calls": 1, "input_tokens": 812, "output_tokens": 140, "cost_usd": 0.00015},
   "latency_ms": 930.4}
  ```

  Ids are written as strings. A line that is not valid JSON, is not an object, or has no string
  in the text field is written with `"record": null` and the reason in `errors`; it does not stop
  the run. A failed LLM call is reported the same way (with the rule values, for `hybrid`).
- **Workers.** `llm` and `hybrid` send up to `--workers` requests at a time (default 4). The rule
  extractor is CPU-bound and always runs in one thread.
- **Cost limit.** With `--max-cost USD` the run stops before a posting that could take the total
  cost past the limit. The cost of a call is known only after it returns, so the most expensive
  posting so far serves as the estimate for each posting still to be sent, counting those in
  flight; until the first call returns only one posting is in flight. The limit can therefore be
  exceeded only by a posting that costs more than every posting before it. A model without a price
  has an unknown cost, and the run stops after its first call.
- **Resume.** `--resume` appends to `--out` and skips every input id already in it (and, for lines
  without an id, the same line number). A last output line cut off by a crash is removed first.
  A record with errors counts as done; delete its line to retry it.
- **Summary.** At the end the command prints, on stderr, the number of records written, failures,
  LLM calls, input and output tokens, cost and elapsed time. Exit status: 0 when every record was
  extracted without error, 1 when some records have errors, 3 when the cost limit stopped the run,
  130 after Ctrl-C (the records written so far are kept; run again with `--resume`).

From code, `extract_many` does the same for any iterable of texts, `(id, text)` pairs or
`BatchInput` objects, and yields results in input order:

```python
from jobspec import LLMExtractor, AnthropicProvider, extract_many

rows = ((row["id"], row["text"]) for row in my_postings)
run = extract_many(rows, LLMExtractor(AnthropicProvider()), workers=4, max_cost=2.0)
for item in run:
    print(item.id, item.errors or item.result.spec.title)
print(run.count, run.failures, run.usage.cost_usd, run.elapsed_s, run.stopped)
```

## Library

```python
from jobspec import RuleExtractor, LLMExtractor, HybridExtractor, AnthropicProvider, OpenAICompatibleProvider

text = open("posting.txt", encoding="utf-8").read()

result = RuleExtractor().extract(text)
spec = result.spec  # a JobSpec
print(spec.salary, spec.tech_stack)
print(spec.to_dict(include_meta=True))  # adds {"meta": {"salary": {"source": "rule", "confidence": 0.95}, ...}}

llm = LLMExtractor(AnthropicProvider(model="claude-haiku-5-5"))  # reads ANTHROPIC_API_KEY
result = llm.extract(text)
print(result.usage.input_tokens, result.usage.output_tokens, result.usage.cost_usd)

local = LLMExtractor(OpenAICompatibleProvider(model="qwen2.5", base_url="http://localhost:8000/v1"))
hybrid = HybridExtractor(local, threshold=0.6)
print(hybrid.extract(text).spec.meta)  # which fields came from the rules and which from the model
```

Evaluation from code:

```python
from jobspec import RuleExtractor, evaluate, load_dataset

report = evaluate(RuleExtractor(), load_dataset("data/eval.jsonl"), "data/eval.jsonl")
print(report.to_text())
print(report.macro("accuracy"), report.fields["salary"].accuracy, report.cost_per_1000)
```

Any object with an `extract(text, html=None) -> ExtractionResult` method can be evaluated, so you
can benchmark your own parser the same way. A provider is any subclass of `jobspec.Provider` that
implements `complete(system, user, max_tokens)`; the tests use a fake one.

## Schema

`JobSpec` is a dataclass. Unknown values are `null` (scalars) or `[]` (lists).

| field | type | values |
|---|---|---|
| `title` | string | job title without salary, location, work-mode or gender-marker suffixes |
| `seniority` | enum | `intern`, `junior`, `mid`, `senior`, `staff`, `principal`, `lead`, `head` |
| `employment_type` | enum | `full_time`, `part_time`, `contract`, `internship` |
| `work_mode` | enum | `remote`, `hybrid`, `onsite` |
| `locations` | list of strings | work locations as written |
| `remote_regions` | list of strings | ISO 3166-1 alpha-2 codes or `WORLDWIDE`, `EU`, `EUROPE`, `EMEA`, `LATAM`, `APAC`, `NORTH_AMERICA`, `AMERICAS` |
| `salary` | object | `min`, `max` (full units), `currency` (ISO 4217), `period` (`hour`, `day`, `week`, `month`, `year`), `gross_net` (`gross`, `net`, or null when not stated) |
| `experience_years_min` | integer | minimum years required |
| `tech_stack` | list of strings | canonical names from `vocab.py` (for example "k8s" becomes "Kubernetes") |
| `visa_sponsorship` | boolean | `true` offered, `false` ruled out, `null` not mentioned |
| `apply_url`, `apply_email` | string | where to apply |

`JobSpec.to_dict()` / `JobSpec.from_dict()` round-trip, `from_dict` raises `SchemaError` on any
invalid value, and `spec.problems()` lists violations. With `include_meta=True` every set field
also carries `{"source": "rule" | "llm", "confidence": 0..1}`. Rule confidences are fixed per
rule (for example, a salary with an explicit period scores higher than one whose period was
inferred); LLM confidences are a constant (0.7 by default), not a calibrated probability.
`jobspec schema` prints the JSON Schema (draft 2020-12).

## Rule extractor results

On the bundled data (`jobspec eval --data data/eval.jsonl --extractor rules`, 30 postings):

| field | accuracy | precision | recall | F1 | gold values |
|---|---:|---:|---:|---:|---:|
| title | 1.00 | 1.00 | 1.00 | 1.00 | 30 |
| seniority | 1.00 | 1.00 | 1.00 | 1.00 | 22 |
| employment_type | 1.00 | 1.00 | 1.00 | 1.00 | 24 |
| work_mode | 1.00 | 1.00 | 1.00 | 1.00 | 30 |
| locations | 0.97 | 1.00 | 0.94 | 0.97 | 18 |
| remote_regions | 0.97 | 0.93 | 1.00 | 0.97 | 14 |
| salary (all five parts right) | 1.00 | 1.00 | 1.00 | 1.00 | 29 |
| experience_years_min | 1.00 | 1.00 | 1.00 | 1.00 | 27 |
| tech_stack | 1.00 | 1.00 | 1.00 | 1.00 | 131 |
| visa_sponsorship | 1.00 | 1.00 | 1.00 | 1.00 | 9 |
| apply_url | 1.00 | 1.00 | 1.00 | 1.00 | 18 |
| apply_email | 1.00 | 1.00 | 1.00 | 1.00 | 12 |

Macro accuracy over the 12 fields: 0.994. Macro F1: 0.995. Average latency: about 6 ms per
posting on a laptop. Cost: zero.

**Read this number with care.** The rules were written together with these 30 postings, so this is
an in-sample score. To get a fairer estimate, `data/holdout.jsonl` (12 postings) was written after
the rules were frozen and run once, without changes afterwards:

| field | accuracy | F1 |
|---|---:|---:|
| title | 1.00 | 1.00 |
| seniority | 1.00 | 1.00 |
| employment_type | 0.83 | 0.89 |
| work_mode | 0.92 | 0.96 |
| locations | 0.42 | 0.22 |
| remote_regions | 0.92 | 0.86 |
| salary | 1.00 | 1.00 |
| experience_years_min | 0.92 | 0.95 |
| tech_stack | 1.00 | 1.00 |
| visa_sponsorship | 0.92 | 0.67 |
| apply_url | 1.00 | 1.00 |
| apply_email | 1.00 | 1.00 |

Macro accuracy on the held-out set: 0.910, macro F1: 0.878. The misses are informative:
locations written without a "Location:" label, hybrid work described only by weekdays
("Tuesday through Thursday in our Boston office"), "4 or more years", work-permit sponsorship
without the word "visa", PJ (a Brazilian contractor term) not being recognized, and "B2B SaaS
company" being read as a B2B contract. Expect real postings, which are longer and messier, to
score lower still. Both sets are synthetic; see [data/README.md](data/README.md).

## Running the LLM evaluation with your own key

```bash
export ANTHROPIC_API_KEY=...            # or OPENAI_API_KEY with --provider openai
jobspec eval --data data/eval.jsonl --extractor llm --model claude-haiku-5-5 --report haiku.md
jobspec eval --data data/eval.jsonl --extractor hybrid --model claude-haiku-5-5 --report hybrid.md
```

The 30 bundled postings make 30 calls (about 6 US cents with Claude Haiku 4.5 at the example
prices below). The report adds token counts, total cost and cost per 1,000 postings, taken from
the `usage` block of each API response. No LLM was called while building this package, so no
LLM numbers are published here; run the command above to get them for your model.

**Cost per 1,000 postings, estimated.** The price table in `src/jobspec/pricing.py` holds example
list prices (USD per 1M input / output tokens) that were current when this was written. Prices
change: check your provider's price list, and pass `--prices prices.json` to override them.

The estimate assumes about 530 tokens of system prompt per call and 250 output tokens. The bundled
postings are short (about 180 tokens each); a typical real posting of about 4,000 characters adds
about 1,000 input tokens.

| model | example price in / out | bundled postings (~710 tokens in) | 4,000-character posting (~1,570 tokens in) |
|---|---|---:|---:|
| claude-haiku-4-5 | $1.00 / $5.00 | $1.96 | $2.82 |
| claude-haiku-5-5 | $0.10 / $0.50 | $0.20 | $0.28 |
| claude-sonnet-5-5 | $2.00 / $10.00 | $3.91 | $5.64 |
| gpt-4.1-mini | $0.40 / $1.60 | $0.68 | $1.03 |

`jobspec estimate --data your.jsonl --model NAME` recomputes the left column for your own postings
without calling any API. The hybrid extractor still makes one call for most postings (some field,
such as visa sponsorship, is usually empty after the rules), but it asks only for the missing
fields, so its output is shorter.

## How the LLM extractor treats input

- **Untrusted text.** The posting goes into the user message between `<<<POSTING nonce>>>` and
  `<<<END POSTING nonce>>>` lines, with a random nonce per call. Any `<<<` or `>>>` inside the
  posting is broken up, so the posting cannot close the block. The system prompt says that
  everything inside the block is third-party data and that instructions in it must be ignored.
  `data/eval.jsonl` includes a posting with an injection attempt (`en-30`).
- **Lenient parsing.** Code fences, text around the object, trailing commas, Python literals
  (`True`, `None`), single quotes and replies cut off at `max_tokens` are repaired where possible.
  Each repair is recorded in `result.warnings`.
- **Validation.** Every field is checked against the schema. Synonyms are mapped (`"full-time"`
  becomes `full_time`, `"120k"` becomes 120000, `"k8s"` becomes Kubernetes); a value that still
  does not fit (an unknown enum, a currency that is not ISO 4217, a non-http URL) is dropped and
  reported, never passed through.
- **Usage and cost.** Input and output tokens come from the API response. Cost is computed from
  the price table; a model without a price reports `n/a`.
- Requests use `max_tokens` 1024 and do not set sampling parameters (API version `2026-10-21`
  rejects `temperature`). Anthropic requests ask for `effort: "low"`: current Claude models
  reason by default, field extraction needs little of it, and those tokens are billed as output.
  Pass `AnthropicProvider(effort=None)` to keep the model's default. Transient errors (429,
  5xx, 529 and network errors) are retried twice with backoff.

## Limitations

- The rules cover English, German, Spanish, Portuguese, Russian and Hindi-English phrasing. Other
  languages will mostly get the salary, stack, URL and email right and miss the rest.
- Locations are read from labeled lines ("Location:", "Standort:", "Город:") and a few prose
  patterns with a list of about 100 cities. Unlabeled locations are often missed (see the held-out
  results).
- One salary per posting: the most salary-like amount wins. Separate ranges for different levels
  or locations are not represented. When no period is written, the period is inferred from the
  currency and the amount, and the confidence is lowered.
- Seniority is not inferred from years of experience.
- The tech vocabulary has about 160 entries. The rule extractor does not report anything outside
  it; the LLM extractor does.
- Confidence values are heuristics, not calibrated probabilities.
- The bundled data is synthetic and small (30 + 12 postings). Use it to compare extractors and to
  catch regressions. For a decision that matters, label a few hundred of your own postings in the
  same JSONL format.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
mypy
```

CI runs ruff, mypy and pytest on Python 3.9 to 3.13. The test suite needs no network and no API
key: the LLM paths are tested with a fake provider and a fake HTTP transport.
`tests/test_evaluate.py` fails if the rule extractor drops below 0.97 macro accuracy on
`data/eval.jsonl`.

## License

MIT. Copyright (c) 2026 Dmitrii Maslennikov.
