"""Synthetic clinical note generator.

Everything here is invented: names, MRNs, phone numbers (555-01xx range) and
clinical content. No real patient data is used anywhere in this repository.
Notes are deliberately varied (abbreviations, negations, different phrasings)
so that a simple rules baseline makes realistic mistakes.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .schema import Extraction, Medication

FIRST = ["Maria", "James", "Aisha", "Chen", "Olivia", "Lucas", "Fatima", "Noah", "Sofia", "Daniel", "Priya", "Ethan"]
LAST = ["Alvarez", "Brooks", "Khan", "Wu", "Novak", "Moreau", "Haddad", "Fischer", "Rossi", "Okafor", "Patel", "Nguyen"]
DOCTORS = ["Reed", "Sato", "Bennett", "Duarte", "Lindqvist"]

# canonical diagnosis, canonical procedure, surface forms used in the note text
CASES = [
    ("lumbar radiculopathy", "mri lumbar spine", ["lumbar radiculopathy", "radiculopathy of the lumbar spine"]),
    ("chronic low back pain", "mri lumbar spine", ["chronic low back pain", "chronic LBP"]),
    ("knee osteoarthritis", "mri knee", ["knee osteoarthritis", "knee OA"]),
    ("meniscal tear", "mri knee", ["meniscal tear", "suspected meniscal tear"]),
    ("chronic migraine", "mri brain", ["chronic migraine", "chronic migraines"]),
]
PROCEDURE_FORMS = {
    "mri lumbar spine": ["MRI lumbar spine", "lumbar spine MRI", "MRI of the lumbar spine", "MRI L-spine"],
    "mri knee": ["MRI knee", "knee MRI", "MRI of the knee"],
    "mri brain": ["MRI brain", "brain MRI", "MRI of the brain"],
}
MEDS = [
    ("ibuprofen", "400 mg"), ("naproxen", "500 mg"), ("gabapentin", "300 mg"),
    ("cyclobenzaprine", "10 mg"), ("sumatriptan", "50 mg"), ("acetaminophen", "500 mg"),
    ("amlodipine", "5 mg"), ("metformin", "500 mg"),
]
ALLERGIES = ["penicillin", "sulfa", "latex", "codeine"]
WEEKS = [0, 2, 3, 4, 6, 8, 12, 16]


def _age_sex(rng: random.Random, age: int, sex: str) -> str:
    word = {"F": "female", "M": "male"}[sex]
    return rng.choice(
        [f"{age}-year-old {word}", f"{age} y/o {sex}", f"{age}yo {sex}", f"{word.capitalize()}, age {age}"]
    )


def _therapy(rng: random.Random, weeks: int) -> str:
    if weeks == 0:
        return rng.choice(
            ["No conservative therapy has been tried so far.", "Has not yet attempted physical therapy or other conservative management."]
        )
    options = [
        f"Completed {weeks} weeks of physical therapy without relief.",
        f"PT x {weeks} wks with minimal improvement.",
        f"Has been in physiotherapy for {weeks} weeks, symptoms persist.",
    ]
    if weeks % 4 == 0:
        options.append(f"Conservative management for about {weeks // 4} months.")
    return rng.choice(options)


def _red_flag(rng: random.Random, flag: bool) -> str:
    if flag:
        return rng.choice(
            ["Red flags: progressive neurological deficit noted.", "Exam shows new focal weakness; red flags present."]
        )
    return rng.choice(
        ["Denies red flags (no weakness, no bowel or bladder changes).", "No red flag symptoms reported."]
    )


def _meds(rng: random.Random, meds: list[Medication]) -> str:
    if not meds:
        return "Current medications: none."
    style = rng.choice(["inline", "bullets"])
    if style == "inline":
        joined = ", ".join(f"{m.name} {m.dose.replace(' ', '') if rng.random() < 0.4 else m.dose}" for m in meds)
        return f"Current medications: {joined}."
    lines = "\n".join(f"- {m.name} {m.dose} {rng.choice(['BID', 'TID', 'daily', 'PRN'])}" for m in meds)
    return f"Current medications:\n{lines}"


def _allergies(rng: random.Random, allergies: list[str]) -> str:
    if not allergies:
        return rng.choice(["No known drug allergies.", "NKDA."])
    return f"Allergies: {', '.join(allergies)}."


def make_note(rng: random.Random, idx: int) -> dict:
    diagnosis, procedure, dx_forms = rng.choice(CASES)
    age = rng.randint(24, 84)
    sex = rng.choice(["F", "M"])
    weeks = rng.choice(WEEKS)
    flag = rng.random() < 0.25
    meds = [Medication(*m) for m in rng.sample(MEDS, rng.randint(0, 3))]
    allergies = rng.sample(ALLERGIES, rng.randint(0, 2))

    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    text = "\n".join(
        [
            f"Patient: {name}   MRN-{rng.randint(100000, 999999)}   DOB: {rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{2026 - age}",
            f"Visit date: {rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/2026   Referring provider: Dr. {rng.choice(DOCTORS)}, phone (415) 555-01{rng.randint(10, 99)}",
            "",
            f"HPI: {_age_sex(rng, age, sex)} presenting with ongoing symptoms. {_therapy(rng, weeks)} {_red_flag(rng, flag)}",
            _meds(rng, meds),
            _allergies(rng, allergies),
            f"Assessment: {rng.choice(dx_forms)}.",
            f"Plan: requesting {rng.choice(PROCEDURE_FORMS[procedure])}.",
        ]
    )
    gold = Extraction(
        age=age,
        sex=sex,
        diagnosis=diagnosis,
        requested_procedure=procedure,
        conservative_therapy_weeks=weeks,
        red_flag=flag,
        medications=meds,
        allergies=sorted(allergies),
    )
    return {"id": f"note-{idx:04d}", "text": text, "gold": gold.to_dict()}


def generate(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    return [make_note(rng, i) for i in range(n)]


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
