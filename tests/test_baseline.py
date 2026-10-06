"""Hand-written notes (not produced by the generator) so the baseline is not
only tested against the templates it was developed on."""
from cne.extractors import BaselineExtractor

ex = BaselineExtractor().extract


def test_age_sex_variants():
    assert (ex("62 y/o F with pain").age, ex("62 y/o F with pain").sex) == (62, "F")
    assert (ex("Male, age 41").age, ex("Male, age 41").sex) == (41, "M")
    assert ex("35-year-old male").sex == "M"


def test_therapy_weeks_and_none():
    assert ex("Completed 8 weeks of physical therapy.").conservative_therapy_weeks == 8
    assert ex("PT x 6 wks, no relief.").conservative_therapy_weeks == 6
    assert ex("No conservative therapy has been tried so far.").conservative_therapy_weeks == 0
    assert ex("Nothing about therapy here.").conservative_therapy_weeks is None


def test_red_flag_negation_is_not_confused_with_presence():
    assert ex("Denies red flags (no weakness).").red_flag is False
    assert ex("Red flags: progressive neurological deficit noted.").red_flag is True
    assert ex("Nothing documented.").red_flag is None


def test_medications_and_allergies():
    note = "Current medications: ibuprofen 400mg, gabapentin 300 mg.\nAllergies: penicillin, sulfa."
    out = ex(note)
    assert {m.key() for m in out.medications} == {("ibuprofen", "400 mg"), ("gabapentin", "300 mg")}
    assert out.allergies == ["penicillin", "sulfa"]
    assert ex("Current medications: none.\nNKDA.").medications == []


def test_known_gap_abbreviation_returns_none_instead_of_guessing():
    assert ex("Assessment: knee OA.").diagnosis is None
