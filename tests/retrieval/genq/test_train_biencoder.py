from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clarity_agent_evals.genq.train_biencoder import (
    BiEncoderTrainingConfig,
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
