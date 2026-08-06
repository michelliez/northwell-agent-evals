from __future__ import annotations

from collections.abc import Sequence

import pytest

from clarity_agent_evals.genq.qwen_query_generator import QwenQueryGenerator, _parse_queries


class ScriptedGenerator(QwenQueryGenerator):
    """Drive the retry and batching logic without loading a model.

    Only the decode step is replaced, so prompt construction, batching, parsing,
    retry accounting, and the fail-closed exit all run as they do in production.
    """

    def __init__(self, scripted: list[list[str]], **kwargs: object) -> None:
        super().__init__("fake-qwen", model=object(), tokenizer=object(), **kwargs)  # type: ignore[arg-type]
        self.scripted = scripted
        self.decode_calls: list[int] = []

    def _build_prompt(self, passage: str, queries_per_passage: int) -> str:
        return f"PROMPT::{passage}::{queries_per_passage}"

    def _decode_batch(self, prompts, *, max_new_tokens, top_p, temperature):  # type: ignore[no-untyped-def]
        self.decode_calls.append(len(prompts))
        return self.scripted.pop(0)


def _generate(generator: QwenQueryGenerator, passages: Sequence[str], count: int = 2):
    return generator.generate(
        passages,
        queries_per_passage=count,
        max_input_tokens=300,
        max_query_tokens=64,
        top_p=0.9,
        seed=42,
    )


def test_parses_a_clean_json_array() -> None:
    generator = ScriptedGenerator([['["what is X?", "where is Y?"]', '["how about Z?", "and W?"]']])

    assert _generate(generator, ["first", "second"]) == [
        ["what is X?", "where is Y?"],
        ["how about Z?", "and W?"],
    ]
    assert generator.decode_calls == [2]


def test_tolerates_markdown_fences_and_surrounding_prose() -> None:
    noisy = 'Sure! Here you go:\n```json\n["first question?", "second question?"]\n```\nHope that helps.'
    generator = ScriptedGenerator([[noisy]])

    assert _generate(generator, ["passage"]) == [["first question?", "second question?"]]


def test_extra_queries_are_truncated_to_the_requested_count() -> None:
    generator = ScriptedGenerator([['["a?", "b?", "c?", "d?"]']])

    assert _generate(generator, ["passage"]) == [["a?", "b?"]]


def test_unparsable_output_is_retried_for_that_passage_only() -> None:
    generator = ScriptedGenerator(
        [
            ['["good one?", "good two?"]', "I am afraid I cannot do that."],
            ['["recovered one?", "recovered two?"]'],
        ]
    )

    result = _generate(generator, ["first", "second"])

    assert result == [["good one?", "good two?"], ["recovered one?", "recovered two?"]]
    # The retry must resend only the failed passage, not the whole batch.
    assert generator.decode_calls == [2, 1]


def test_persistent_failure_fails_closed(mocker: object) -> None:
    generator = ScriptedGenerator([["nope"], ["still nope"], ["nope again"]], max_retries=2)

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        _generate(generator, ["passage"])
    assert generator.decode_calls == [1, 1, 1]


def test_too_few_queries_counts_as_unparsable() -> None:
    generator = ScriptedGenerator([['["only one?"]'], ['["one?", "two?"]']])

    assert _generate(generator, ["passage"]) == [["one?", "two?"]]
    assert generator.decode_calls == [1, 1]


def test_passages_are_split_across_gpu_batches() -> None:
    generator = ScriptedGenerator(
        [
            ['["a1?", "a2?"]', '["b1?", "b2?"]'],
            ['["c1?", "c2?"]'],
        ],
        batch_size=2,
    )

    result = _generate(generator, ["one", "two", "three"])

    assert len(result) == 3
    assert generator.decode_calls == [2, 1]


def test_results_stay_positionally_aligned_after_a_partial_retry() -> None:
    # First and third parse; the middle one needs a retry. Order must survive.
    generator = ScriptedGenerator(
        [
            ['["a1?", "a2?"]', "garbage", '["c1?", "c2?"]'],
            ['["b1?", "b2?"]'],
        ],
        batch_size=3,
    )

    assert _generate(generator, ["one", "two", "three"]) == [
        ["a1?", "a2?"],
        ["b1?", "b2?"],
        ["c1?", "c2?"],
    ]


def test_passage_is_bounded_by_approximate_input_tokens() -> None:
    captured: list[str] = []

    class Capturing(ScriptedGenerator):
        def _build_prompt(self, passage: str, queries_per_passage: int) -> str:
            captured.append(passage)
            return passage

    generator = Capturing([['["a?", "b?"]']])
    generator.generate(
        ["x" * 500],
        queries_per_passage=2,
        max_input_tokens=10,
        max_query_tokens=64,
        top_p=0.9,
        seed=1,
    )

    assert captured == ["x" * 40]


def test_whitespace_in_generated_queries_is_normalized() -> None:
    generator = ScriptedGenerator([['["  what   is\\tX?  ", "where is Y?"]']])

    assert _generate(generator, ["passage"]) == [["what is X?", "where is Y?"]]


def test_invalid_construction_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        QwenQueryGenerator("fake", model=object(), tokenizer=object(), batch_size=0)
    with pytest.raises(ValueError, match="max_retries"):
        QwenQueryGenerator("fake", model=object(), tokenizer=object(), max_retries=-1)


@pytest.mark.parametrize(
    "completion",
    [
        "",
        "no array here",
        "[not, valid, json]",
        "[1, 2]",
        '["", "  "]',
    ],
)
def test_parse_queries_rejects_unusable_output(completion: str) -> None:
    assert _parse_queries(completion, 2) is None


def test_parse_queries_accepts_exact_count() -> None:
    assert _parse_queries('["a?", "b?"]', 2) == ["a?", "b?"]


def test_parse_queries_recovers_an_array_wrapped_in_an_object() -> None:
    # Instruct models often ignore "bare array" and emit a keyed object anyway.
    # Recovering the array beats burning a retry on a usable answer.
    assert _parse_queries('{"queries": ["a?", "b?"]}', 2) == ["a?", "b?"]


def test_parse_queries_accepts_brackets_inside_a_query() -> None:
    """A bracketed term in the text must not truncate the surrounding array.

    This killed a corpus run 32% of the way in. The HMM_MAP passage reads
    "Carryover Mapping [HMM] database", the model quoted it faithfully, and the
    old non-greedy `\\[.*?\\]` stopped at the "]" of "[HMM]" -- handing the
    decoder a fragment. Because the bracket comes from the passage, every retry
    reproduced it, so a valid answer failed closed three times running.

    Epic documentation uses bracketed tags routinely: 24 of 31,060 eligible
    passages contain one, which made a completed run essentially impossible.
    """
    completion = (
        '["What community IDs are associated with the Carryover Mapping [HMM] database?", '
        '"How many unique community IDs are listed in the HMM_MAP table?"]'
    )
    assert _parse_queries(completion, 2) == [
        "What community IDs are associated with the Carryover Mapping [HMM] database?",
        "How many unique community IDs are listed in the HMM_MAP table?",
    ]


def test_parse_queries_skips_a_bracket_that_opens_nothing_decodable() -> None:
    """Scanning must resume past a "[" that begins no complete value.

    Prose can contain a stray bracket before the real array, so finding one is
    not the same as finding the answer.
    """
    assert _parse_queries('see [note] below: ["a?", "b?"]', 2) == ["a?", "b?"]
