import pytest

from cne.schema import Extraction, normalize_dose


def test_normalize_dose():
    assert normalize_dose("400mg") == "400 mg"
    assert normalize_dose("0.5 MG") == "0.5 mg"
    with pytest.raises(ValueError):
        normalize_dose("a lot")


def test_from_dict_normalises_and_validates():
    ex = Extraction.from_dict(
        {
            "age": "58",
            "sex": "female",
            "diagnosis": " Knee Osteoarthritis ",
            "conservative_therapy_weeks": 8.0,
            "red_flag": False,
            "medications": [{"name": "Ibuprofen", "dose": "400mg"}],
            "allergies": ["Penicillin", "none"],
        }
    )
    assert ex.age == 58 and ex.sex == "F"
    assert ex.diagnosis == "knee osteoarthritis"
    assert ex.conservative_therapy_weeks == 8
    assert ex.medications[0].key() == ("ibuprofen", "400 mg")
    assert ex.allergies == ["penicillin"]


@pytest.mark.parametrize(
    "bad",
    [
        {"age": "old"},
        {"age": True},
        {"sex": "X"},
        {"red_flag": "yes"},
        {"medications": [{"name": "x"}]},
        {"allergies": "penicillin"},
    ],
)
def test_from_dict_rejects_malformed(bad):
    with pytest.raises(ValueError):
        Extraction.from_dict(bad)
