import pytest

from cne.metrics import field_errors, score
from cne.schema import Extraction, Medication


def gold():
    return Extraction(
        age=50, sex="F", diagnosis="knee osteoarthritis", requested_procedure="mri knee",
        conservative_therapy_weeks=8, red_flag=False,
        medications=[Medication("ibuprofen", "400 mg"), Medication("gabapentin", "300 mg")],
        allergies=["penicillin"],
    )


def test_perfect_prediction():
    r = score([gold()], [gold()])
    assert r["note_exact_match"] == 1.0
    assert r["decision_agreement"] == 1.0
    assert r["fields"]["medications"]["f1"] == 1.0


def test_list_metrics_count_missing_and_extra_items():
    pred = gold()
    pred.medications = [Medication("ibuprofen", "400 mg"), Medication("naproxen", "500 mg")]
    r = score([pred], [gold()])["fields"]["medications"]
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["f1"] == 0.5


def test_dose_formatting_is_not_an_error():
    pred = gold()
    pred.medications = [Medication("Ibuprofen", "400mg"), Medication("gabapentin", "300 mg")]
    assert field_errors(pred, gold()) == []


def test_decision_agreement_can_differ_from_field_accuracy():
    pred = gold()
    pred.conservative_therapy_weeks = 2  # wrong field, flips the decision
    r = score([pred], [gold()])
    assert r["fields"]["conservative_therapy_weeks"]["accuracy"] == 0.0
    assert r["decision_agreement"] == 0.0


def test_input_validation():
    with pytest.raises(ValueError):
        score([], [])
    with pytest.raises(ValueError):
        score([gold()], [])
