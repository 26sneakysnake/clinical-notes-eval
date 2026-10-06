"""Two interchangeable extractors: a rules baseline and an LLM-backed one."""
from __future__ import annotations

import json
import os
import re
from typing import Protocol

from .redact import redact
from .schema import DIAGNOSES, PROCEDURES, Extraction, Medication, normalize_dose


class Extractor(Protocol):
    name: str

    def extract(self, note: str) -> Extraction: ...


# --------------------------------------------------------------------------- #
# Rules baseline: what you would ship in a day without an LLM.
# --------------------------------------------------------------------------- #
_AGE_SEX_A = re.compile(r"(\d{1,3})\s*-?\s*(?:year\s*-?\s*old|y/o|yo)\s+(female|male|f|m)\b", re.I)
_AGE_SEX_B = re.compile(r"\b(female|male),?\s+age\s+(\d{1,3})\b", re.I)
_PROC = {
    "mri lumbar spine": re.compile(r"mri\s+(?:of\s+the\s+)?lumbar\s+spine|lumbar\s+spine\s+mri", re.I),
    "mri knee": re.compile(r"mri\s+(?:of\s+the\s+)?knee|knee\s+mri", re.I),
    "mri brain": re.compile(r"mri\s+(?:of\s+the\s+)?brain|brain\s+mri", re.I),
}
_WEEKS = [
    re.compile(r"(\d+)\s*(?:weeks?|wks?)\s+of\s+(?:physical therapy|physiotherapy|pt)\b", re.I),
    re.compile(r"\b(?:physical therapy|physiotherapy|pt)\s*(?:x|for)\s*(\d+)\s*(?:weeks?|wks?)", re.I),
]
_NO_THERAPY = re.compile(r"no conservative therapy|has not yet attempted", re.I)
_MED = re.compile(r"\b([A-Za-z]{4,})\s+(\d+(?:\.\d+)?)\s?(mg|mcg|g)\b")


class BaselineExtractor:
    name = "baseline"

    def extract(self, note: str) -> Extraction:
        low = note.lower()
        ex = Extraction()

        if m := _AGE_SEX_A.search(note):
            ex.age, ex.sex = int(m.group(1)), m.group(2)[0].upper()
        elif m := _AGE_SEX_B.search(note):
            ex.age, ex.sex = int(m.group(2)), m.group(1)[0].upper()

        ex.diagnosis = next((d for d in DIAGNOSES if d in low), None)
        ex.requested_procedure = next((p for p, rx in _PROC.items() if rx.search(note)), None)

        for rx in _WEEKS:
            if m := rx.search(note):
                ex.conservative_therapy_weeks = int(m.group(1))
                break
        else:
            if _NO_THERAPY.search(note):
                ex.conservative_therapy_weeks = 0

        if re.search(r"denies red flags|no red flag", low):
            ex.red_flag = False
        elif re.search(r"red flags?:|new focal weakness|progressive neurological", low):
            ex.red_flag = True

        if m := re.search(r"current medications:(.*?)(?:allergies|nkda|no known)", note, re.I | re.S):
            ex.medications = [
                Medication(name.lower(), normalize_dose(f"{dose}{unit}"))
                for name, dose, unit in _MED.findall(m.group(1))
            ]

        if m := re.search(r"allergies:\s*([^\n.]*)", note, re.I):
            ex.allergies = [a.strip().lower() for a in m.group(1).split(",") if a.strip()]
        return ex


# --------------------------------------------------------------------------- #
# LLM extractor: redact -> prompt -> parse -> validate -> retry once.
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = f"""You extract structured data from a clinical note.
Return ONE JSON object and nothing else (no prose, no code fences) with exactly these keys:
- age: integer or null
- sex: "F", "M" or null
- diagnosis: one of {list(DIAGNOSES)} or null if none fits
- requested_procedure: one of {list(PROCEDURES)} or null if none fits
- conservative_therapy_weeks: integer number of weeks of conservative therapy (convert months to weeks, 4 weeks per month), 0 if the note says none was tried, null if not stated
- red_flag: true if red flags are present, false if explicitly absent, null if not stated
- medications: list of {{"name": lowercase string, "dose": "<number> <unit>"}}
- allergies: list of lowercase strings, empty list if none

Rules: use only what the note states. If unsure, use null. Never guess."""


def parse_json(raw: str) -> dict:
    """Extract the first JSON object from a model reply, tolerating code fences."""
    raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found")
    return json.loads(raw[start : end + 1])


class LLMExtractor:
    name = "llm"

    def __init__(self, client=None, model: str | None = None, redact_input: bool = True, max_retries: int = 1):
        self.model = model or os.environ.get("CNE_MODEL", DEFAULT_MODEL)
        self.redact_input = redact_input
        self.max_retries = max_retries
        self.failures = 0
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic  # imported lazily so tests and the baseline need no SDK

            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def extract(self, note: str) -> Extraction:
        text = redact(note) if self.redact_input else note
        for _ in range(self.max_retries + 1):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
            raw = "".join(b.text for b in resp.content if getattr(b, "type", "text") == "text")
            try:
                return Extraction.from_dict(parse_json(raw))
            except (ValueError, json.JSONDecodeError):
                continue
        self.failures += 1
        return Extraction()
