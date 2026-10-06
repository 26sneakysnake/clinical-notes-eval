from cne.synthetic import generate


def test_deterministic_for_a_given_seed():
    assert generate(10, seed=1) == generate(10, seed=1)
    assert generate(10, seed=1) != generate(10, seed=2)


def test_gold_labels_are_consistent_with_text():
    for rec in generate(100, seed=3):
        gold, text = rec["gold"], rec["text"]
        assert str(gold["age"]) in text
        for med in gold["medications"]:
            assert med["name"] in text
        for allergy in gold["allergies"]:
            assert allergy in text


def test_dob_matches_age():
    rec = generate(1, seed=5)[0]
    assert f"/{2026 - rec['gold']['age']}" in rec["text"]
