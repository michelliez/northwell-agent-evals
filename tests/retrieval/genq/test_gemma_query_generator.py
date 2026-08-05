from __future__ import annotations

import pytest

from clarity_agent_evals.genq.gemma_query_generator import GemmaMLXQueryGenerator


def test_parse_queries_accepts_json_code_fence() -> None:
    queries = GemmaMLXQueryGenerator._parse_queries(
        '```json\n{"queries": ["What is A0H_MAP?", "Which columns are available?"]}\n```'
    )

    assert queries == ["What is A0H_MAP?", "Which columns are available?"]


def test_parse_queries_accepts_tagged_lines() -> None:
    queries = GemmaMLXQueryGenerator._parse_queries(
        "<query>What is A0H_MAP?</query>\n<query>Which columns are available?</query>"
    )

    assert queries == ["What is A0H_MAP?", "Which columns are available?"]


def test_parse_queries_accepts_one_plain_question() -> None:
    queries = GemmaMLXQueryGenerator._parse_queries(
        "What information does the A0H_MAP table contain?"
    )

    assert queries == ["What information does the A0H_MAP table contain?"]


def test_parse_queries_accepts_question_wrapped_in_smart_quotes() -> None:
    queries = GemmaMLXQueryGenerator._parse_queries(
        "“What records were deleted from the A0H_DELETE table?”"
    )

    assert queries == ["What records were deleted from the A0H_DELETE table?"]


def test_parse_queries_rejects_duplicates() -> None:
    with pytest.raises(RuntimeError, match="duplicate"):
        GemmaMLXQueryGenerator._parse_queries(
            '{"queries": ["What is A0H_MAP?", "What is A0H_MAP?"]}'
        )


def test_generation_retries_invalid_response_and_generates_queries_individually(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = GemmaMLXQueryGenerator("local-gemma")
    responses: list[RuntimeError | list[str]] = [
        RuntimeError("invalid JSON"),
        ["What is query one?"],
        ["What is query two?"],
    ]

    def fake_request(*args: object, **kwargs: object) -> list[str]:
        del args, kwargs
        result = responses.pop(0)
        if isinstance(result, RuntimeError):
            raise result
        return result

    monkeypatch.setattr(generator, "_request_queries", fake_request)

    queries = generator._generate_for_passage(
        "A sufficiently detailed table passage.",
        queries_per_passage=2,
        max_query_tokens=64,
        top_p=0.95,
    )

    assert queries == ["What is query one?", "What is query two?"]


def test_generation_returns_empty_result_after_all_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = GemmaMLXQueryGenerator("local-gemma")
    generator.max_attempts = 2

    def always_fail(*args: object, **kwargs: object) -> list[str]:
        del args, kwargs
        raise RuntimeError("invalid response")

    monkeypatch.setattr(generator, "_request_queries", always_fail)

    queries = generator._generate_for_passage(
        "A sufficiently detailed table passage.",
        queries_per_passage=2,
        max_query_tokens=64,
        top_p=0.95,
    )

    assert queries == []
