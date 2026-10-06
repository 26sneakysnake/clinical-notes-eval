import pytest

from cne.policy import adjudicate
from cne.schema import Extraction


def case(**kw):
    return Extraction(**kw)


def test_lumbar_mri_needs_six_weeks_or_red_flag():
    assert adjudicate(case(requested_procedure="mri lumbar spine", conservative_therapy_weeks=6, red_flag=False)).decision == "approve"
    assert adjudicate(case(requested_procedure="mri lumbar spine", conservative_therapy_weeks=2, red_flag=True)).decision == "approve"
    assert adjudicate(case(requested_procedure="mri lumbar spine", conservative_therapy_weeks=2, red_flag=False)).decision == "pend"


def test_missing_information_pends_instead_of_guessing():
    assert adjudicate(case(requested_procedure="mri knee")).decision == "pend"
    assert adjudicate(case()).decision == "pend"
    assert adjudicate(case(requested_procedure="mri brain")).decision == "pend"


@pytest.mark.parametrize("weeks", [None, 0, 3, 5, 6, 40])
@pytest.mark.parametrize("flag", [None, True, False])
@pytest.mark.parametrize("proc", [None, "mri knee", "mri brain", "mri lumbar spine", "ct chest"])
def test_never_denies_on_its_own(proc, flag, weeks):
    d = adjudicate(case(requested_procedure=proc, red_flag=flag, conservative_therapy_weeks=weeks))
    assert d.decision in ("approve", "pend")
    assert d.reasons
