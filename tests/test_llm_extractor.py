"""The LLM path is tested with a fake client: no network, no API key."""
import json
from types import SimpleNamespace

from cne.extractors import LLMExtractor, parse_json

NOTE = "Patient: Maria Alvarez   MRN-123456\nHPI: 58 y/o F. Assessment: knee OA. Plan: MRI knee."

GOOD = {
    "age": 58, "sex": "F", "diagnosis": "knee osteoarthritis", "requested_procedure": "mri knee",
    "conservative_therapy_weeks": None, "red_flag": None, "medications": [], "allergies": [],
}


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.sent.append(kwargs["messages"][0]["content"])
        block = SimpleNamespace(type="text", text=self.replies.pop(0))
        return SimpleNamespace(content=[block])


def test_parse_json_tolerates_code_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_identifiers_never_reach_the_client():
    client = FakeClient([json.dumps(GOOD)])
    LLMExtractor(client=client).extract(NOTE)
    assert "Maria" not in client.sent[0] and "123456" not in client.sent[0]


def test_valid_reply_is_parsed():
    out = LLMExtractor(client=FakeClient([json.dumps(GOOD)])).extract(NOTE)
    assert out.diagnosis == "knee osteoarthritis" and out.age == 58


def test_invalid_json_is_retried_once():
    client = FakeClient(["not json", json.dumps(GOOD)])
    ex = LLMExtractor(client=client)
    assert ex.extract(NOTE).age == 58
    assert len(client.sent) == 2 and ex.failures == 0


def test_persistent_failure_returns_empty_extraction_and_is_counted():
    ex = LLMExtractor(client=FakeClient(["nope", '{"age": "old"}']))
    out = ex.extract(NOTE)
    assert out.age is None and ex.failures == 1
