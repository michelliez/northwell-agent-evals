from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clarity_agent_evals.genq.train_biencoder import (
    BiEncoderTrainingConfig,
    _resolve_optimizer,
    load_training_pairs,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _chunk_line(chunk_id: str, text: str, split: str = "train") -> str:
    return json.dumps(
        {
            "chunk_id": chunk_id,
            "source_file": "TABLE.html",
            "source_hash": _hash("TABLE.html source"),
            "table_name": "TABLE",
            "column_name": None,
            "chunk_type": "table_metadata",
            "section_name": "TABLE",
            "text": text,
            "text_hash": _hash(text),
            "parser_version": "test-v1",
            "split": split,
            "split_version": "test-split-v1",
        },
        sort_keys=True,
    )


def _query_line(
    query_id: str,
    query: str,
    chunk_id: str,
    text_hash: str,
    split: str = "train",
) -> str:
    return json.dumps(
        {
            "query_id": query_id,
            "query": query,
            "relevant_chunk_id": chunk_id,
            "relevant_text_hash": text_hash,
            "source_file": "TABLE.html",
            "chunk_type": "table_metadata",
            "split": split,
            "split_version": "test-split-v1",
            "generator_model": "test-model",
            "generation_seed": 42,
            "query_index": 0,
            "filter_version": "test-filter-v1",
            "meaningful_overlap_tokens": ["table"],
        },
        sort_keys=True,
    )


def test_loads_train_and_validation_pairs(tmp_path: Path) -> None:
    passage = "Table: FOO\nDescription: A foo table."
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        _chunk_line("c1", passage, "train")
        + "\n"
        + _chunk_line("c2", "Table: BAR", "validation")
        + "\n"
        + _chunk_line("c3", "Table: BAZ", "test")
        + "\n",
        encoding="utf-8",
    )
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(
        _query_line("q1", "What is FOO?", "c1", _hash(passage), "train")
        + "\n"
        + _query_line("q2", "What is BAR?", "c2", _hash("Table: BAR"), "validation")
        + "\n"
        + _query_line("q3", "What is BAZ?", "c3", _hash("Table: BAZ"), "test")
        + "\n",
        encoding="utf-8",
    )

    train, val, excluded = load_training_pairs(queries_path, chunks_path)

    assert len(train) == 1
    assert train[0] == ("What is FOO?", passage)
    assert len(val) == 1
    assert val[0] == ("What is BAR?", "Table: BAR")
    assert excluded == 1


def test_rejects_missing_chunk_reference(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        _chunk_line("c1", "Table: FOO", "train") + "\n",
        encoding="utf-8",
    )
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(
        _query_line("q1", "What is FOO?", "MISSING", _hash("Table: FOO"), "train") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown chunk"):
        load_training_pairs(queries_path, chunks_path)


def test_rejects_empty_train_split(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        _chunk_line("c1", "Table: FOO", "test") + "\n",
        encoding="utf-8",
    )
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(
        _query_line("q1", "What is FOO?", "c1", _hash("Table: FOO"), "test") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No train-split"):
        load_training_pairs(queries_path, chunks_path)


def test_config_rejects_invalid_settings() -> None:
    config = BiEncoderTrainingConfig(
        base_model="test",
        retained_queries_path=Path("q.jsonl"),
        chunks_path=Path("c.jsonl"),
        output_dir=Path("out"),
        epochs=0,
    )
    with pytest.raises(ValueError, match="epochs"):
        config.validate()


def test_memory_savings_are_on_by_default() -> None:
    """A 0.6B encoder does not fit on a 10 GB card in fp32 with fp32 moments.

    Weights, gradients, and AdamW moments come to 8.9 GB before any activation,
    against 8.9 GB free, so opting in would mean the documented command fails
    for the encoder the project actually uses.
    """
    config = BiEncoderTrainingConfig(
        base_model="test",
        retained_queries_path=Path("q.jsonl"),
        chunks_path=Path("c.jsonl"),
        output_dir=Path("out"),
    )
    assert config.use_bfloat16 is True
    assert config.use_8bit_optimizer is True


def test_sequence_length_is_capped_well_below_the_model_default() -> None:
    """The encoder's own 32768 default is the wrong number for this corpus.

    Clarity passages are 31 tokens at the median and 460 at the longest, so the
    declared maximum is roughly 300x what any passage needs. Attention memory
    grows with sequence length, and at the model default a 10 GB card spills
    into system RAM rather than failing fast: 32 seconds per iteration.

    The cap must still clear the longest real passage, or training silently
    truncates the passages it is meant to learn.
    """
    config = BiEncoderTrainingConfig(
        base_model="test",
        retained_queries_path=Path("q.jsonl"),
        chunks_path=Path("c.jsonl"),
        output_dir=Path("out"),
    )
    assert config.max_seq_length == 512
    assert config.max_seq_length > 460


class _FakeTorch:
    class optim:
        class AdamW:
            pass


def test_optimizer_is_plain_adamw_when_8bit_is_not_wanted() -> None:
    optimizer, name = _resolve_optimizer(_FakeTorch, want_8bit=False)

    assert optimizer is _FakeTorch.optim.AdamW
    assert name == "adamw"


def test_missing_bitsandbytes_costs_memory_not_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent optional dependency must not abort a multi-hour job.

    The reported name has to change with it, or a later reader cannot tell a
    run that used 8-bit moments from one that quietly fell back.
    """
    import builtins

    real_import = builtins.__import__

    def deny_bitsandbytes(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("bitsandbytes"):
            raise ImportError("bitsandbytes is not installed")
        return real_import(name, *args, **kwargs)  # pyright: ignore[reportCallIssue, reportArgumentType]

    monkeypatch.setattr(builtins, "__import__", deny_bitsandbytes)

    optimizer, name = _resolve_optimizer(_FakeTorch, want_8bit=True)

    assert optimizer is _FakeTorch.optim.AdamW
    assert name == "adamw"
