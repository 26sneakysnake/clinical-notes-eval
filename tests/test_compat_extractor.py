"""OpenAI-compatible path tested with an injected fake HTTP function: no network."""
import json

import pytest

from cne.extractors import OpenAICompatExtractor

NOTE = "Patient: Maria Alvarez   MRN-123456\nHPI: 58 y/o F. Assessment: knee OA. Plan: MRI knee."
GOOD = {
    "age": 58, "sex": "F", "diagnosis": "knee osteoarthritis", "requested_procedure": "mri knee",
    "conservative_therapy_weeks": None, "red_flag": None, "medications": [], "allergies": [],
}


def reply(text):
    return {"choices": [{"message": {"content": text}}]}


class FakePost:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, headers, payload):
        self.calls.append((url, headers, payload))
        return self.replies.pop(0)


def test_request_shape_and_redaction():
    post = FakePost([reply(json.dumps(GOOD))])
    out = OpenAICompatExtractor("qwen2.5:7b", post=post).extract(NOTE)
    url, headers, payload = post.calls[0]
    assert url == "http://localhost:11434/v1/chat/completions"
    assert "Authorization" not in headers
    assert payload["model"] == "qwen2.5:7b"
    sent = payload["messages"][1]["content"]
    assert "Maria" not in sent and "123456" not in sent
    assert out.diagnosis == "knee osteoarthritis"


def test_api_key_is_sent_as_bearer_token():
    post = FakePost([reply(json.dumps(GOOD))])
    OpenAICompatExtractor("m", base_url="https://example.test/v1/", api_key="k", post=post).extract(NOTE)
    url, headers, _ = post.calls[0]
    assert url == "https://example.test/v1/chat/completions"
    assert headers["Authorization"] == "Bearer k"


def test_invalid_json_is_retried_then_counted():
    post = FakePost([reply("sorry, here is prose"), reply("still no json")])
    ex = OpenAICompatExtractor("m", post=post)
    assert ex.extract(NOTE).age is None
    assert len(post.calls) == 2 and ex.failures == 1


def test_unexpected_response_shape_raises_instead_of_scoring_zero():
    with pytest.raises(RuntimeError):
        OpenAICompatExtractor("m", post=FakePost([{"error": "model not found"}])).extract(NOTE)


def test_result_file_name_is_filesystem_safe():
    assert OpenAICompatExtractor("qwen2.5:7b").name == "compat-qwen2.5_7b"
