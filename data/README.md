# Bundled evaluation data

Every posting in this directory is **synthetic**. The companies, people, addresses, email
addresses and URLs are invented (domains end in `.example`), and no text was copied from a real
job posting. The postings imitate common real-world formats, so they are useful for comparing
extractors, but they are short and cleaner than most real postings.

| file | postings | purpose |
|---|---:|---|
| `eval.jsonl` | 30 | the main labeled set: 18 English, 2 German, 3 Spanish, 2 Russian, 2 Portuguese, 3 Hindi-English (India) |
| `holdout.jsonl` | 12 | written after the rules were frozen and never used to tune them |

The rule extractor was developed alongside `eval.jsonl`, so its score there is an in-sample
number. `holdout.jsonl` gives a less optimistic estimate. A model evaluated through the LLM
extractor has seen neither file, so both files are out-of-sample for it.

## Format

One JSON object per line:

```json
{"id": "en-05", "lang": "en", "text": "...posting...", "label": { ...JobSpec... }, "notes": "...", "format": "html"}
```

- `text` is the posting as plain text or HTML. `format` is `"html"`, `"text"` or omitted (detect).
- `label` follows `JobSpec.to_dict()` (see `jobspec schema`). Missing keys mean "not stated".
- `notes` names what the posting tests (for example "'Go' as a verb", "remote-first but must live
  in Poland").

## Labeling conventions

These rules were used for every label. Read them before adding postings.

- **title**: as written, minus gender markers (`(m/w/d)`), salary, location, work mode,
  experience and contract-length suffixes. Seniority words and parenthesized specializations
  stay ("Backend Engineer (Node.js)", "Аналитик данных (middle)").
- **seniority**: only when the title or an explicit statement gives it. It is not inferred from
  years of experience. Mapping: working student, Werkstudent, estágio -> `intern`; graduate ->
  `junior`; semi senior, pleno, II, middle -> `mid`; engineering manager, team/tech lead ->
  `lead`; head of, director, VP -> `head`.
- **employment_type**: when stated or unambiguous from a local term (CLT and "contrato a término
  indefinido" -> `full_time`; B2B, PJ, 1099, day-rate contracting -> `contract`). A Werkstudent
  job stated as Teilzeit is `part_time`. "Contract: Permanent" is not a contract job.
- **work_mode**: as stated. Office days per week mean `hybrid`.
- **locations**: work locations as written in the posting. For remote jobs, only a location the
  posting gives as the job's location (not where the hire may live).
- **remote_regions**: for remote jobs, where the hire may live: ISO 3166-1 alpha-2 codes, or
  `WORLDWIDE`, `EU`, `EUROPE`, `EMEA`, `LATAM`, `APAC`, `NORTH_AMERICA`, `AMERICAS`. The UK is `GB`.
- **salary**: amounts in full units for one period ("120k" -> 120000, "18 LPA" -> 1800000).
  "Up to X" is a max only, "from X" a min only, a single figure is both. When no period is written
  the label uses the market convention (annual for "$150k", monthly for roubles). `gross_net` only
  when the posting says gross/brutto/до вычета or net/netto/на руки. "Competitive salary" is no
  salary.
- **experience_years_min**: the overall minimum. When several figures appear ("8+ years, including
  3+ on infrastructure") the overall one is the largest. "No experience required" is `null`.
- **tech_stack**: every technology named, including nice-to-haves, using the canonical names of
  `src/jobspec/vocab.py`. Postings avoid technologies outside the vocabulary so that labels do not
  depend on naming choices.
- **visa_sponsorship**: `true` when offered (including work-permit sponsorship), `false` when
  ruled out, `null` when not mentioned.
- **apply_url / apply_email**: the place the posting says to apply.

## Adding data

Real postings are welcome only if you have the right to redistribute them. Run
`jobspec eval --data your.jsonl --errors` to see every mismatch; `load_dataset` rejects labels that
do not fit the schema.
