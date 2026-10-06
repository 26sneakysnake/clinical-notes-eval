"""Data model shared by every extractor, the metrics and the policy engine."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

DIAGNOSES = (
    "lumbar radiculopathy",
    "chronic low back pain",
    "knee osteoarthritis",
    "meniscal tear",
    "chronic migraine",
)
PROCEDURES = ("mri lumbar spine", "mri knee", "mri brain")

SCALAR_FIELDS = (
    "age",
    "sex",
    "diagnosis",
    "requested_procedure",
    "conservative_therapy_weeks",
    "red_flag",
)
LIST_FIELDS = ("medications", "allergies")


def normalize_dose(dose: str) -> str:
    """'400mg' -> '400 mg', '0.5 MG' -> '0.5 mg'."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(mg|mcg|g)\s*", dose, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"unrecognised dose: {dose!r}")
    return f"{m.group(1)} {m.group(2).lower()}"


@dataclass(frozen=True)
class Medication:
    name: str
    dose: str

    def key(self) -> tuple[str, str]:
        return (self.name.strip().lower(), normalize_dose(self.dose))


@dataclass
class Extraction:
    age: int | None = None
    sex: str | None = None  # "F" or "M"
    diagnosis: str | None = None
    requested_procedure: str | None = None
    conservative_therapy_weeks: int | None = None
    red_flag: bool | None = None
    medications: list[Medication] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Extraction":
        """Validate and normalise a dict (for example parsed LLM output).

        Raises ValueError on anything malformed so callers can retry or flag it.
        """
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")

        def opt_int(name: str) -> int | None:
            v = data.get(name)
            if v is None:
                return None
            if isinstance(v, bool):
                raise ValueError(f"{name}: expected integer")
            if isinstance(v, (int, float)) and float(v).is_integer():
                return int(v)
            if isinstance(v, str) and v.strip().isdigit():
                return int(v.strip())
            raise ValueError(f"{name}: expected integer, got {v!r}")

        def opt_str(name: str) -> str | None:
            v = data.get(name)
            if v is None:
                return None
            if not isinstance(v, str):
                raise ValueError(f"{name}: expected string")
            v = v.strip().lower()
            return v or None

        sex = opt_str("sex")
        if sex is not None:
            sex = sex.upper()[:1]
            if sex not in ("F", "M"):
                raise ValueError("sex: expected F or M")

        red_flag = data.get("red_flag")
        if red_flag is not None and not isinstance(red_flag, bool):
            raise ValueError("red_flag: expected boolean or null")

        meds_raw = data.get("medications") or []
        if not isinstance(meds_raw, list):
            raise ValueError("medications: expected list")
        meds: list[Medication] = []
        for item in meds_raw:
            if not isinstance(item, dict) or "name" not in item or "dose" not in item:
                raise ValueError("medications: each item needs name and dose")
            meds.append(
                Medication(str(item["name"]).strip().lower(), normalize_dose(str(item["dose"])))
            )

        allergies_raw = data.get("allergies") or []
        if not isinstance(allergies_raw, list):
            raise ValueError("allergies: expected list")
        allergies = [
            str(a).strip().lower()
            for a in allergies_raw
            if str(a).strip().lower() not in ("", "none", "nkda")
        ]

        return cls(
            age=opt_int("age"),
            sex=sex,
            diagnosis=opt_str("diagnosis"),
            requested_procedure=opt_str("requested_procedure"),
            conservative_therapy_weeks=opt_int("conservative_therapy_weeks"),
            red_flag=red_flag,
            medications=meds,
            allergies=allergies,
        )
