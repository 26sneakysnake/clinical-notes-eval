"""Command line entry point.

    python -m cne.cli generate --n 200 --seed 7
    python -m cne.cli eval --extractor baseline
    python -m cne.cli eval --extractor llm --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .extractors import BaselineExtractor, LLMExtractor
from .metrics import field_errors, score
from .schema import Extraction, Medication
from .synthetic import generate, read_jsonl, write_jsonl

DATA = Path("data/notes.jsonl")
RESULTS = Path("results")


def _gold(record: dict) -> Extraction:
    g = dict(record["gold"])
    g["medications"] = [Medication(**m) for m in g["medications"]]
    return Extraction(**g)


def cmd_generate(args: argparse.Namespace) -> int:
    write_jsonl(generate(args.n, args.seed), DATA)
    print(f"wrote {args.n} synthetic notes to {DATA}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    if args.extractor == "llm" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; the llm extractor needs it.", file=sys.stderr)
        return 2
    extractor = BaselineExtractor() if args.extractor == "baseline" else LLMExtractor()

    records = read_jsonl(DATA)
    if args.limit:
        records = records[: args.limit]
    golds = [_gold(r) for r in records]
    preds = [extractor.extract(r["text"]) for r in records]

    result = score(preds, golds)
    if isinstance(extractor, LLMExtractor):
        result["model"] = extractor.model
        result["invalid_outputs"] = extractor.failures

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{extractor.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    with (RESULTS / f"{extractor.name}_errors.jsonl").open("w") as f:
        for r, p, g in zip(records, preds, golds):
            for err in field_errors(p, g):
                f.write(json.dumps({"id": r["id"], **err}) + "\n")

    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cne")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate synthetic notes")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("eval", help="evaluate an extractor on data/notes.jsonl")
    e.add_argument("--extractor", choices=["baseline", "llm"], default="baseline")
    e.add_argument("--limit", type=int, default=0)
    e.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
