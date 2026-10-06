# clinical-notes-eval

A small, testable pipeline that turns **synthetic** clinical notes into structured data, evaluates the result field by field, and applies a simple prior-authorisation policy that **never denies on its own**: it approves, or pends the case for a clinician.

It compares two extractors on the same labelled data:

- a **rules baseline** (regexes and a small lexicon), runs offline;
- an **LLM extractor** (Anthropic API) with input redaction, strict JSON validation and one retry.

I built it to practise the loop that matters for LLM agents on documents: extract, measure, look at the errors, improve.

> **Scope and honesty.** All data is synthetic (invented names, MRNs, 555-01xx phone numbers, invented clinical content). No real patient data is used. The policy is made up and is not medical guidance. The redaction step is a demo safeguard, not HIPAA de-identification. Templated synthetic notes are far easier than real ones, so the numbers below say nothing about real-world performance.

## Pipeline

```
note text -> redact PHI -> extractor (rules | LLM) -> validate -> Extraction
                                                          |
                              metrics vs gold  <----------+----> policy: approve | pend
```

| Module | Role |
|---|---|
| `cne/synthetic.py` | Seeded generator of varied notes with gold labels (abbreviations, negations, months vs weeks) |
| `cne/redact.py` | Removes names, MRNs, phones, emails, dates before text leaves the process |
| `cne/extractors.py` | `BaselineExtractor` and `LLMExtractor` (injectable client) |
| `cne/schema.py` | `Extraction` dataclass and strict validation of model output |
| `cne/metrics.py` | Per-field accuracy, micro precision/recall/F1 for lists, note exact match, decision agreement |
| `cne/policy.py` | Synthetic policy; never auto-denies |

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -q                              # 121 tests, no network, no API key
python -m cne.cli generate --n 200 --seed 7      # regenerate data/notes.jsonl
python -m cne.cli eval --extractor baseline      # offline
export ANTHROPIC_API_KEY=...                     # your own key
python -m cne.cli eval --extractor llm --limit 50
```

Results land in `results/<extractor>.json`, and every field-level mistake is written to `results/<extractor>_errors.jsonl` for error analysis. The model defaults to `claude-haiku-4-5-20251001`; override it with `CNE_MODEL`.

## Results (200 synthetic notes, seed 7)

| Metric | Rules baseline | LLM extractor |
|---|---|---|
| age / sex / red flag accuracy | 1.00 / 1.00 / 1.00 | not run yet |
| diagnosis accuracy | 0.705 | not run yet |
| requested procedure accuracy | 0.920 | not run yet |
| conservative therapy weeks accuracy | 0.850 | not run yet |
| medications F1 / allergies F1 | 1.00 / 1.00 | not run yet |
| note exact match | 0.580 | not run yet |
| policy decision agreement | 0.865 | not run yet |

The LLM column stays empty until I run it with my own API key; I will not fill it with estimates.

## What the baseline gets wrong (from `results/baseline_errors.jsonl`)

All 105 field errors fall into three groups:

- **59 diagnosis errors:** abbreviations and alternative wording ("knee OA", "chronic LBP", "radiculopathy of the lumbar spine") that the lexicon does not contain.
- **30 therapy-duration errors:** "about 3 months" instead of weeks. The baseline returns `null` rather than guess.
- **16 procedure errors:** "MRI L-spine".

These are the cases a language model should handle. They are also where an LLM could be wrong with confidence, which is why the evaluation, not the model, decides what ships.

## Design decisions

- **Never deny automatically.** The policy only approves or pends for clinician review. Missing information pends instead of guessing.
- **Redact before sending.** Text is scrubbed before it reaches any client; a test asserts that identifiers never reach the API client.
- **Validate model output.** Malformed JSON, wrong types or unknown values raise, trigger one retry, then return an empty extraction and count as an invalid output.
- **Injectable client.** The LLM path is tested with a fake client, so the suite needs no network or key.
- **Closed-set labels.** Diagnoses and procedures are normalised to a fixed list, which makes accuracy measurable.

## Limitations

- Synthetic, templated data; real notes are messier (scans, tables, contradictions across pages).
- Regex redaction misses free-text names, addresses and many other identifiers.
- No HIPAA compliance claim: no BAA, audit logging, access control or encryption layer.
- Small label set, single note per case, no multi-document reasoning.
- Temperature is left at the API default for compatibility across models, so LLM runs are not perfectly reproducible.

## Next steps

- Run and record the LLM extractor; compare cost and latency per note.
- Add a held-out set written by hand, to avoid tuning to the generator.
- Add confidence scores and route low-confidence cases to review.

## License

MIT
