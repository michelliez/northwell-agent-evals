from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

import clarity_agent_evals.genq.query_generation as query_generation
from clarity_agent_evals.genq.query_generation import (
    DEFAULT_CLAUDE_MODEL,
    GenerationConfig,
    generate_queries,
)

HASH = hashlib.sha256(b"value").hexdigest()


class FakeGenerator:
    model_name = DEFAULT_CLAUDE_MODEL
    device_name = "fake-cpu"

    def __init__(
        self,
        *,
        duplicate: bool = False,
        wrong_count: bool = False,
        skip: bool = False,
    ) -> None:
        self.duplicate = duplicate
        self.wrong_count = wrong_count
        self.skip = skip
        self.calls: list[tuple[list[str], int]] = []

    def generate(
        self,
        passages: Sequence[str],
        *,
        queries_per_passage: int,
        max_input_tokens: int,
        max_query_tokens: int,
        top_p: float,
        seed: int,
    ) -> list[list[str]]:
        del max_input_tokens, max_query_tokens, top_p
        self.calls.append((list(passages), seed))
        if self.skip:
            return [[] for _passage in passages]
        count = queries_per_passage - 1 if self.wrong_count else queries_per_passage
        return [
            [
                " repeated query " if self.duplicate else f" What is passage {i}? \t"
                for i in range(count)
            ]
            for _passage in passages
        ]


def split_chunk(
    chunk_id: str,
    source_file: str,
    split: str,
    *,
    text: str,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_file": source_file,
        "source_hash": HASH,
        "table_name": Path(source_file).stem,
        "column_name": "ID",
        "chunk_type": "column_definition",
        "section_name": "Column Information",
        "text": text,
        "text_hash": HASH,
        "parser_version": "epic-genq-html-v1",
        "split": split,
        "split_version": "source-hash-split-v1",
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def config(tmp_path: Path, input_path: Path, **overrides: object) -> GenerationConfig:
    values: dict[str, object] = {
        "input_path": input_path,
        "output_path": tmp_path / "generated_queries.jsonl",
        "report_path": tmp_path / "generation_report.json",
        "batch_size": 1,
        "queries_per_chunk": 2,
        "min_passage_chars": 20,
    }
    values.update(overrides)
    return GenerationConfig(**values)  # type: ignore[arg-type]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_queries_keep_positive_chunk_and_inherit_split(tmp_path: Path) -> None:
    input_path = tmp_path / "split_chunks.jsonl"
    write_jsonl(
        input_path,
        [
            split_chunk(
                "TABLE_A__COLUMN__ID",
                "TABLE_A.html",
                "train",
                text="A sufficiently long passage for query generation.",
            ),
            split_chunk(
                "TABLE_B__COLUMN__ID",
                "TABLE_B.html",
                "test",
                text="Another sufficiently long passage for query generation.",
            ),
        ],
    )
    fake = FakeGenerator()
    settings = config(tmp_path, input_path)

    report = generate_queries(settings, generator=fake)
    rows = read_jsonl(settings.output_path)

    assert len(rows) == 4
    assert {row["relevant_chunk_id"] for row in rows} == {
        "TABLE_A__COLUMN__ID",
        "TABLE_B__COLUMN__ID",
    }
    assert {row["split"] for row in rows} == {"train", "test"}
    assert all(not str(row["query"]).startswith(" ") for row in rows)
    assert all("\t" not in str(row["query"]) for row in rows)
    assert report.query_counts_by_split == {"train": 2, "validation": 0, "test": 2}
    assert [seed for _passages, seed in fake.calls] == [42, 43]


def test_short_passages_are_reported_and_not_generated(tmp_path: Path) -> None:
    input_path = tmp_path / "split_chunks.jsonl"
    write_jsonl(
        input_path,
        [
            split_chunk("SHORT", "SHORT.html", "train", text="Too short"),
            split_chunk(
                "LONG",
                "LONG.html",
                "validation",
                text="This passage is comfortably above the configured minimum.",
            ),
        ],
    )
    settings = config(tmp_path, input_path, min_passage_chars=30)

    report = generate_queries(settings, generator=FakeGenerator())
    rows = read_jsonl(settings.output_path)

    assert report.input_chunk_count == 2
    assert report.eligible_chunk_count == 1
    assert report.skipped_short_chunk_count == 1
    assert {row["relevant_chunk_id"] for row in rows} == {"LONG"}


def test_limit_is_applied_after_length_filter(tmp_path: Path) -> None:
    input_path = tmp_path / "split_chunks.jsonl"
    write_jsonl(
        input_path,
        [
            split_chunk("SHORT", "SHORT.html", "train", text="short"),
            split_chunk("FIRST", "FIRST.html", "train", text="First eligible passage text."),
            split_chunk("SECOND", "SECOND.html", "test", text="Second eligible passage text."),
        ],
    )
    settings = config(tmp_path, input_path, limit=1)

    report = generate_queries(settings, generator=FakeGenerator())
    rows = read_jsonl(settings.output_path)

    assert report.selected_chunk_count == 1
    assert {row["relevant_chunk_id"] for row in rows} == {"FIRST"}


def test_duplicate_raw_queries_are_counted_but_retained(tmp_path: Path) -> None:
    input_path = tmp_path / "split_chunks.jsonl"
    write_jsonl(
        input_path,
        [
            split_chunk(
                "ONE",
                "ONE.html",
                "train",
                text="One eligible passage for duplicate query testing.",
            )
        ],
    )
    settings = config(tmp_path, input_path, queries_per_chunk=3)

    report = generate_queries(settings, generator=FakeGenerator(duplicate=True))

    assert report.generated_query_count == 3
    assert report.duplicate_query_count == 2
    assert len(read_jsonl(settings.output_path)) == 3


def test_output_and_query_ids_are_deterministic_with_fake_generator(tmp_path: Path) -> None:
    input_path = tmp_path / "split_chunks.jsonl"
    write_jsonl(
        input_path,
        [
            split_chunk(
                "ONE",
                "ONE.html",
                "train",
                text="One eligible passage for deterministic generation.",
            )
        ],
    )
    settings = config(tmp_path, input_path)

    first = generate_queries(settings, generator=FakeGenerator())
    first_bytes = settings.output_path.read_bytes()
    second = generate_queries(settings, generator=FakeGenerator())

    assert first.output_query_hash == second.output_query_hash
    assert first_bytes == settings.output_path.read_bytes()


def test_generator_returning_wrong_query_count_fails_before_output(tmp_path: Path) -> None:
    input_path = tmp_path / "split_chunks.jsonl"
    write_jsonl(
        input_path,
        [
            split_chunk(
                "ONE",
                "ONE.html",
                "train",
                text="One eligible passage for invalid generation.",
            )
        ],
    )
    settings = config(tmp_path, input_path)

    with pytest.raises(RuntimeError, match="expected 2"):
        generate_queries(settings, generator=FakeGenerator(wrong_count=True))
    assert not settings.output_path.exists()


def test_generator_exhaustion_skips_chunk_and_is_reported(tmp_path: Path) -> None:
    input_path = tmp_path / "split_chunks.jsonl"
    write_jsonl(
        input_path,
        [
            split_chunk(
                "ONE",
                "ONE.html",
                "train",
                text="One eligible passage whose generation will be skipped.",
            )
        ],
    )
    settings = config(tmp_path, input_path)

    report = generate_queries(settings, generator=FakeGenerator(skip=True))

    assert report.failed_generation_chunk_count == 1
    assert report.generated_query_count == 0
    assert read_jsonl(settings.output_path) == []


def test_pipeline_builds_the_claude_generator_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "split_chunks.jsonl"
    write_jsonl(
        input_path,
        [
            split_chunk(
                "ONE",
                "ONE.html",
                "test",
                text="One eligible passage for Claude provider selection.",
            )
        ],
    )
    fake = FakeGenerator()
    fake.model_name = "test-claude-haiku"
    fake.device_name = "anthropic-api"
    captured_models: list[str] = []

    captured_strides: list[int] = []

    def make_generator(model_name: str, *, passages_per_request: int = 1) -> FakeGenerator:
        captured_models.append(model_name)
        captured_strides.append(passages_per_request)
        return fake

    monkeypatch.setattr(query_generation, "ClaudeHaikuQueryGenerator", make_generator)
    settings = config(tmp_path, input_path, model_name="test-claude-haiku")

    report = generate_queries(settings)

    assert captured_models == ["test-claude-haiku"]
    assert captured_strides == [1]
    assert report.generator_model == "test-claude-haiku"
    assert report.device == "anthropic-api"
    assert report.passages_per_request == 1


def test_pipeline_builds_the_gemma_generator_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "split_chunks.jsonl"
    write_jsonl(
        input_path,
        [
            split_chunk(
                "ONE",
                "ONE.html",
                "test",
                text="One eligible passage for Gemma provider selection.",
            )
        ],
    )
    fake = FakeGenerator()
    fake.model_name = "local-gemma"
    fake.device_name = "mlx-local-api"
    captured: list[tuple[str, str]] = []

    def make_generator(model_name: str, *, base_url: str) -> FakeGenerator:
        captured.append((model_name, base_url))
        return fake

    monkeypatch.setattr(query_generation, "GemmaMLXQueryGenerator", make_generator)
    settings = config(
        tmp_path,
        input_path,
        provider="gemma",
        gemma_model_name="local-gemma",
        gemma_base_url="http://127.0.0.1:9090",
    )

    report = generate_queries(settings)

    assert captured == [("local-gemma", "http://127.0.0.1:9090")]
    assert report.generator_model == "local-gemma"
    assert report.device == "mlx-local-api"


def test_pipeline_builds_the_qwen_generator_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "split_chunks.jsonl"
    write_jsonl(
        input_path,
        [
            split_chunk(
                "ONE",
                "ONE.html",
                "test",
                text="One eligible passage for Qwen provider selection.",
            )
        ],
    )
    fake = FakeGenerator()
    fake.model_name = "local-qwen"
    fake.device_name = "cuda"
    captured: list[dict[str, object]] = []

    def make_generator(
        model_name: str,
        *,
        device: str,
        load_in_4bit: bool,
        batch_size: int,
        max_retries: int,
    ) -> FakeGenerator:
        captured.append(
            {
                "model_name": model_name,
                "device": device,
                "load_in_4bit": load_in_4bit,
                "batch_size": batch_size,
                "max_retries": max_retries,
            }
        )
        return fake

    monkeypatch.setattr(query_generation, "QwenQueryGenerator", make_generator)
    settings = config(
        tmp_path,
        input_path,
        provider="qwen",
        qwen_model_name="local-qwen",
        qwen_device="cuda",
        qwen_load_in_4bit=False,
        qwen_max_retries=4,
        batch_size=32,
    )

    report = generate_queries(settings)

    # batch_size reaches the generator unchanged: Qwen batches on the GPU, so a
    # separate knob could disagree with the pipeline about VRAM per run.
    assert captured == [
        {
            "model_name": "local-qwen",
            "device": "cuda",
            "load_in_4bit": False,
            "batch_size": 32,
            "max_retries": 4,
        }
    ]
    assert report.generator_model == "local-qwen"
    assert report.device == "cuda"


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size": 0},
        {"queries_per_chunk": 0},
        {"top_p": 1.1},
        {"seed": -1},
        {"limit": 0},
        {"model_name": "  "},
        {"provider": "unknown"},
        {"gemma_model_name": "  "},
        {"gemma_base_url": "  "},
        {"qwen_model_name": "  "},
        {"qwen_device": "tpu"},
        {"qwen_max_retries": -1},
    ],
)
def test_invalid_generation_configuration_is_rejected(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    settings = config(tmp_path, tmp_path / "input.jsonl", **overrides)

    with pytest.raises(ValueError):
        settings.validate()
