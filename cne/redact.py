"""Regex-based PHI scrubbing applied BEFORE any text leaves the process.

This is a defence-in-depth step for the demo, NOT HIPAA de-identification.
Regexes miss free-text names, addresses and many other identifiers. A real
system needs a validated de-identification pipeline and a BAA with every
vendor that touches PHI.
"""
from __future__ import annotations

import re

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"MRN[-: ]*\d{5,}", re.IGNORECASE), "[MRN]"),
    (re.compile(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "[DATE]"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATE]"),
    (re.compile(r"(Patient|Name):\s*[A-Z][a-z]+(?: [A-Z][a-z]+)+"), r"\1: [NAME]"),
    (re.compile(r"\bDr\.\s+[A-Z][a-z]+(?: [A-Z][a-z]+)?"), "Dr. [NAME]"),
]


def redact(text: str) -> str:
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text
