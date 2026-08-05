from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from clarity_agent_evals.genq.claude_query_generator import (
    ClaudeHaikuQueryGenerator,
    _messages_url,
)


class FakeMessages:
    def __init__(self, query_batches: list[list[Any]]) -> None:
        self.query_batches = query_batches
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        queries = self.query_batches.pop(0)
        block = SimpleNamespace(type="tool_use", name="return_queries", input={"queries": queries})
        return SimpleNamespace(content=[block])


class FakeClient:
    def __init__(self, query_batches: list[list[Any]]) -> None:
        self.messages = FakeMessages(query_batches)


def test_claude_generates_one_structured_result_per_passage() -> None:
    client = FakeClient([["question a", "question b"], ["question c", "question d"]])
    generator = ClaudeHaikuQueryGenerator("test-haiku", client=client)

    result = generator.generate(
        ["first passage", "second passage"],
        queries_per_passage=2,
        max_input_tokens=300,
        max_query_tokens=64,
        top_p=0.9,
        seed=42,
    )

    assert result == [["question a", "question b"], ["question c", "question d"]]
    assert len(client.messages.calls) == 2
    assert client.messages.calls[0]["model"] == "test-haiku"
    assert client.messages.calls[0]["tool_choice"]["name"] == "return_queries"
    schema = client.messages.calls[0]["tools"][0]["input_schema"]["properties"]["queries"]
    assert schema["minItems"] == 2
    assert schema["maxItems"] == 2


def test_claude_passage_is_bounded_by_approximate_input_tokens() -> None:
    client = FakeClient([["question"]])
    generator = ClaudeHaikuQueryGenerator(client=client)

    generator.generate(
        ["x" * 100],
        queries_per_passage=1,
        max_input_tokens=5,
        max_query_tokens=10,
        top_p=1.0,
        seed=0,
    )

    content = client.messages.calls[0]["messages"][0]["content"]
    assert content.endswith("x" * 20)
    assert "x" * 21 not in content


@pytest.mark.parametrize("queries", [["only one"], ["one", 2]])
def test_claude_rejects_wrong_or_non_string_query_values(queries: list[Any]) -> None:
    generator = ClaudeHaikuQueryGenerator(client=FakeClient([queries]))

    with pytest.raises(RuntimeError, match="expected 2"):
        generator.generate(
            ["passage"],
            queries_per_passage=2,
            max_input_tokens=300,
            max_query_tokens=64,
            top_p=0.95,
            seed=42,
        )


def test_claude_requires_tool_result() -> None:
    client = FakeClient([["unused"]])
    client.messages.create = lambda **_kwargs: SimpleNamespace(content=[])  # type: ignore[method-assign]
    generator = ClaudeHaikuQueryGenerator(client=client)

    with pytest.raises(RuntimeError, match="required return_queries"):
        generator.generate(
            ["passage"],
            queries_per_passage=1,
            max_input_tokens=300,
            max_query_tokens=64,
            top_p=0.95,
            seed=42,
        )


def test_claude_accepts_raw_http_response_mapping() -> None:
    class DictMessages:
        def create(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "return_queries",
                        "input": {"queries": ["question a", "question b"]},
                    }
                ]
            }

    client = SimpleNamespace(messages=DictMessages())
    generator = ClaudeHaikuQueryGenerator(client=client)

    result = generator.generate(
        ["passage"],
        queries_per_passage=2,
        max_input_tokens=300,
        max_query_tokens=64,
        top_p=0.95,
        seed=42,
    )

    assert result == [["question a", "question b"]]


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.anthropic.com", "https://api.anthropic.com/v1/messages"),
        ("https://example.test/v1", "https://example.test/v1/messages"),
        ("https://example.test/v1/messages", "https://example.test/v1/messages"),
    ],
)
def test_messages_url_normalizes_supported_base_urls(base_url: str, expected: str) -> None:
    assert _messages_url(base_url) == expected


class FakeBatchedMessages:
    """Return a batched `results` payload, optionally corrupted for failure tests."""

    def __init__(self, payloads: list[Any]) -> None:
        self.payloads = payloads
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        block = SimpleNamespace(type="tool_use", name="return_queries", input=payload)
        return SimpleNamespace(content=[block])


class FakeBatchedClient:
    def __init__(self, payloads: list[Any]) -> None:
        self.messages = FakeBatchedMessages(payloads)


def _results(*groups: list[str]) -> dict[str, Any]:
    return {
        "results": [
            {"passage_index": index, "queries": queries} for index, queries in enumerate(groups)
        ]
    }


def _generate(generator: ClaudeHaikuQueryGenerator, passages: list[str], count: int = 2):
    return generator.generate(
        passages,
        queries_per_passage=count,
        max_input_tokens=300,
        max_query_tokens=64,
        top_p=0.9,
        seed=42,
    )


def test_batched_requests_group_passages_into_one_call() -> None:
    client = FakeBatchedClient([_results(["a1", "a2"], ["b1", "b2"], ["c1", "c2"])])
    generator = ClaudeHaikuQueryGenerator("test-haiku", client=client, passages_per_request=3)

    result = _generate(generator, ["first", "second", "third"])

    assert result == [["a1", "a2"], ["b1", "b2"], ["c1", "c2"]]
    assert len(client.messages.calls) == 1
    content = client.messages.calls[0]["messages"][0]["content"]
    assert "<passage index=0>" in content
    assert "<passage index=2>" in content
    schema = client.messages.calls[0]["tools"][0]["input_schema"]["properties"]["results"]
    assert schema["minItems"] == 3
    assert schema["maxItems"] == 3


def test_batched_results_are_reordered_by_passage_index() -> None:
    scrambled = {
        "results": [
            {"passage_index": 2, "queries": ["c1", "c2"]},
            {"passage_index": 0, "queries": ["a1", "a2"]},
            {"passage_index": 1, "queries": ["b1", "b2"]},
        ]
    }
    client = FakeBatchedClient([scrambled])
    generator = ClaudeHaikuQueryGenerator(client=client, passages_per_request=3)

    assert _generate(generator, ["first", "second", "third"]) == [
        ["a1", "a2"],
        ["b1", "b2"],
        ["c1", "c2"],
    ]


def test_batched_generation_splits_across_several_requests() -> None:
    client = FakeBatchedClient(
        [_results(["a1", "a2"], ["b1", "b2"]), _results(["c1", "c2"], ["d1", "d2"])]
    )
    generator = ClaudeHaikuQueryGenerator(client=client, passages_per_request=2)

    result = _generate(generator, ["one", "two", "three", "four"])

    assert len(result) == 4
    assert len(client.messages.calls) == 2


def test_batch_remainder_falls_back_to_the_single_passage_shape() -> None:
    client = FakeBatchedClient([_results(["a1", "a2"], ["b1", "b2"]), {"queries": ["c1", "c2"]}])
    generator = ClaudeHaikuQueryGenerator(client=client, passages_per_request=2)

    result = _generate(generator, ["one", "two", "three"])

    assert result == [["a1", "a2"], ["b1", "b2"], ["c1", "c2"]]
    # A one-passage remainder must use the original schema, not a 1-item batch.
    assert "queries" in client.messages.calls[1]["tools"][0]["input_schema"]["properties"]


def test_missing_passage_index_fails_closed() -> None:
    payload = {
        "results": [
            {"passage_index": 0, "queries": ["a1", "a2"]},
            {"passage_index": 0, "queries": ["b1", "b2"]},
        ]
    }
    client = FakeBatchedClient([payload])
    generator = ClaudeHaikuQueryGenerator(client=client, passages_per_request=2)

    with pytest.raises(RuntimeError, match="more than once"):
        _generate(generator, ["one", "two"])


def test_out_of_range_passage_index_fails_closed() -> None:
    payload = {
        "results": [
            {"passage_index": 0, "queries": ["a1", "a2"]},
            {"passage_index": 7, "queries": ["b1", "b2"]},
        ]
    }
    client = FakeBatchedClient([payload])
    generator = ClaudeHaikuQueryGenerator(client=client, passages_per_request=2)

    with pytest.raises(RuntimeError, match=r"unexpected=\[7\]"):
        _generate(generator, ["one", "two"])


def test_short_query_list_in_a_batch_fails_closed() -> None:
    client = FakeBatchedClient([_results(["a1", "a2"], ["b1"])])
    generator = ClaudeHaikuQueryGenerator(client=client, passages_per_request=2)

    with pytest.raises(RuntimeError, match="expected 2"):
        _generate(generator, ["one", "two"])


def test_non_integer_passage_index_fails_closed() -> None:
    payload = {
        "results": [
            {"passage_index": "0", "queries": ["a1", "a2"]},
            {"passage_index": 1, "queries": ["b1", "b2"]},
        ]
    }
    client = FakeBatchedClient([payload])
    generator = ClaudeHaikuQueryGenerator(client=client, passages_per_request=2)

    with pytest.raises(RuntimeError, match="non-integer passage_index"):
        _generate(generator, ["one", "two"])


def test_passages_per_request_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ClaudeHaikuQueryGenerator(client=FakeBatchedClient([]), passages_per_request=0)
