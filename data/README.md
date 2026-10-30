# Bundled evaluation data

Every posting in this directory is **synthetic**. The companies, people, addresses, email
addresses and URLs are invented (domains end in `.example`), and no text was copied from a real
job posting. The postings imitate common real-world formats, so they are useful for comparing
extractors, but they are short and cleaner than most real postings.

| file | postings | purpose |
|---|---:|---|
| `eval.jsonl` | 30 | the main labeled set: 18 English, 2 German, 3 Spanish, 2 Russian, 2 Portuguese, 3 Hindi-English (India) |
| `dev.jsonl` | 12 | the 0.1.0 held-out set (named `holdout.jsonl` in 0.1.0); the 0.2.0 rule fixes were made by looking at it |
| `holdout.jsonl` | 12 | new in 0.2.0: written without running the rules on it, evaluated once after the rules were final |

The rule extractor was developed alongside `eval.jsonl`, so its score there is an in-sample
number. `dev.jsonl` was written after the 0.1.0 rules were frozen and was a held-out set for
0.1.0; its misses were then used to fix the 0.2.0 rules, so it is now a development set and its
0.2.0 score is in-sample too.

`holdout.jsonl` (0.2.0) has 12 new postings: 7 English (United States, Poland, India, Singapore,
United Kingdom, EMEA remote, and an HTML posting from Australia), 1 French, 1 German (Austria),
1 Spanish (Mexico), 1 Portuguese (Portugal) and 1 Russian. They were written and labeled by hand
before the extractor was run on them, then evaluated exactly once with the final 0.2.0 rules;
no rule was changed after that run, so the misses it found are still in the rules. The author of
the postings knew the rule code, which is a source of bias this set cannot remove. When the rules
are changed using what this set shows, it stops being held out: rename it, write a new one, and
say so in the README.

A model evaluated through the LLM extractor has seen none of these files, so all three are
out-of-sample for it.

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

- **title**: as written, minus gender markers (`(m/w/d)`, `(H/F)`), salary, location, work
  mode, experience, employment-type and contract-length suffixes. Seniority words and
  parenthesized specializations stay ("Backend Engineer (Node.js)", "Аналитик данных (middle)").
- **seniority**: only when the title or an explicit statement gives it. It is not inferred from
  years of experience. Mapping: working student, Werkstudent, estágio -> `intern`; graduate ->
  `junior`; semi senior, pleno, II, middle -> `mid`; engineering manager, team/tech lead ->
  `lead`; head of, director, VP -> `head`.
- **employment_type**: when stated or unambiguous from a local term (CLT, CDI, "contrato a término
  indefinido" and "contrato sem termo" -> `full_time`; B2B (as a contract form, not "a B2B SaaS
  company"), PJ, 1099, day-rate contracting -> `contract`). A Werkstudent job stated as Teilzeit
  is `part_time`. "Contract: Permanent" is not a contract job.
- **work_mode**: as stated. Office days per week, named office weekdays, or some home-office
  days per week mean `hybrid`.
- **locations**: work locations as written in the posting. For remote jobs, only a location the
  posting gives as the job's location (not where the hire may live).
- **remote_regions**: for remote jobs, where the hire may live: ISO 3166-1 alpha-2 codes, or
  `WORLDWIDE`, `EU`, `EUROPE`, `EMEA`, `LATAM`, `APAC`, `NORTH_AMERICA`, `AMERICAS`. The UK is `GB`.
- **salary**: amounts in full units for one period ("120k" -> 120000, "18 LPA" -> 1800000).
  "Up to X" is a max only, "from X" a min only (so is a stated minimum that "may be exceeded"),
  a single figure is both. When no period is written
  the label uses the market convention (annual for "$150k", monthly for roubles). `gross_net` only
  when the posting says gross/brutto/до вычета or net/netto/на руки. "Competitive salary" is no
  salary.
- **experience_years_min**: the overall minimum. When several figures appear ("8+ years, including
  3+ on infrastructure") the overall one is the largest. "No experience required" is `null`.
- **tech_stack**: every technology named, including nice-to-haves, using the canonical names of
  `src/jobspec/vocab.py`. Postings avoid technologies outside the vocabulary so that labels do not
  depend on naming choices.
- **visa_sponsorship**: `true` when offered (including work-permit, residence-permit and
  Employment Pass sponsorship or support), `false` when ruled out, `null` when not mentioned.
- **apply_url / apply_email**: the place the posting says to apply.

## Adding data

Real postings are welcome only if you have the right to redistribute them. Run
`jobspec eval --data your.jsonl --errors` to see every mismatch; `load_dataset` rejects labels that
do not fit the schema.
