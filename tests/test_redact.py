from cne.redact import redact

NOTE = (
    "Patient: Maria Alvarez   MRN-123456   DOB: 04/02/1960\n"
    "Referring provider: Dr. Reed, phone (415) 555-0142, maria@example.com\n"
    "Assessment: knee osteoarthritis."
)


def test_identifiers_are_removed():
    out = redact(NOTE)
    for leaked in ("Maria", "Alvarez", "123456", "04/02/1960", "Reed", "555-0142", "example.com"):
        assert leaked not in out


def test_clinical_content_is_kept():
    assert "knee osteoarthritis" in redact(NOTE)


def test_idempotent():
    once = redact(NOTE)
    assert redact(once) == once
