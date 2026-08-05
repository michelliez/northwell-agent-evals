from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clarity_agent_evals.dataset_split import (
    SPLIT_VERSION,
    SplitConfig,
    assign_source_split,
    build_splits,
)

HASH = hashlib.sha256(b"value").hexdigest()


def chunk(chunk_id: str, source_file: str, *, table_name: str | None = None) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_file": source_file,
        "source_hash": HASH,
        "table_name": table_name or Path(source_file).stem,
        "column_name": "ID",
        "chunk_type": "column_definition",
        "section_name": "Column Information",
        "text": f"Table {table_name or Path(source_file).stem}. Column ID.",
        "text_hash": HASH,
        "parser_version": "epic-genq-html-v1",
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_config(tmp_path: Path, input_path: Path, **overrides: object) -> SplitConfig:
    values: dict[str, object] = {
        "input_path": input_path,
        "output_path": tmp_path / "chunks_with_splits.jsonl",
        "report_path": tmp_path / "split_report.json",
    }
    values.update(overrides)
    return SplitConfig(**values)  # type: ignore[arg-type]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_all_chunks_from_one_source_receive_one_split(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        input_path,
        [
            chunk("TABLE_A__COLUMN__ONE", "TABLE_A.html"),
            chunk("TABLE_A__COLUMN__TWO", "TABLE_A.html"),
            chunk("TABLE_B__COLUMN__ONE", "TABLE_B.html"),
        ],
    )
    config = make_config(tmp_path, input_path)

    report = build_splits(config)
    rows = read_jsonl(config.output_path)

    table_a_splits = {row["split"] for row in rows if row["source_file"] == "TABLE_A.html"}
    assert len(table_a_splits) == 1
    assert all(row["split_version"] == SPLIT_VERSION for row in rows)
    assert report.leakage_source_count == 0
    assert report.duplicate_chunk_id_count == 0
    assert sum(report.source_counts_by_split.values()) == 2
    assert sum(report.chunk_counts_by_split.values()) == 3


def test_same_seed_is_byte_deterministic(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        input_path,
        [chunk(f"TABLE_{index}__COLUMN__ID", f"TABLE_{index}.html") for index in range(20)],
    )
    config = make_config(tmp_path, input_path)

    first = build_splits(config)
    first_bytes = config.output_path.read_bytes()
    second = build_splits(config)

    assert first.output_corpus_hash == second.output_corpus_hash
    assert first.split_manifest_hash == second.split_manifest_hash
    assert first_bytes == config.output_path.read_bytes()


def test_adding_sources_does_not_move_existing_sources(tmp_path: Path) -> None:
    config = make_config(tmp_path, tmp_path / "unused.jsonl", seed="stable-seed")
    initial_sources = [f"TABLE_{index}.html" for index in range(30)]
    initial = {source: assign_source_split(source, config) for source in initial_sources}

    expanded_sources = initial_sources + [f"NEW_{index}.html" for index in range(30)]
    expanded = {source: assign_source_split(source, config) for source in expanded_sources}

    assert all(expanded[source] == split for source, split in initial.items())


def test_hash_thresholds_populate_all_splits_for_representative_sample(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, tmp_path / "unused.jsonl")
    counts = {
        split: sum(
            assign_source_split(f"TABLE_{index}.html", config) == split for index in range(1_000)
        )
        for split in ("train", "validation", "test")
    }

    assert 740 <= counts["train"] <= 860
    assert 60 <= counts["validation"] <= 140
    assert 60 <= counts["test"] <= 140


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_ratio": 0.7, "validation_ratio": 0.1, "test_ratio": 0.1},
        {"train_ratio": 0.8, "validation_ratio": 0.2, "test_ratio": 0.0},
        {"seed": "  "},
    ],
)
def test_invalid_configuration_is_rejected(tmp_path: Path, overrides: dict[str, object]) -> None:
    config = make_config(tmp_path, tmp_path / "chunks.jsonl", **overrides)

    with pytest.raises(ValueError):
        config.validate()


def test_duplicate_chunk_ids_are_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        input_path,
        [
            chunk("DUPLICATE", "TABLE_A.html"),
            chunk("DUPLICATE", "TABLE_B.html"),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate chunk IDs"):
        build_splits(make_config(tmp_path, input_path))


def test_inconsistent_metadata_within_source_is_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        input_path,
        [
            chunk("ONE", "TABLE_A.html", table_name="TABLE_A"),
            chunk("TWO", "TABLE_A.html", table_name="WRONG_TABLE"),
        ],
    )

    with pytest.raises(ValueError, match="Inconsistent source metadata"):
        build_splits(make_config(tmp_path, input_path))


def test_invalid_json_reports_line_number(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    input_path.write_text(
        json.dumps(chunk("ONE", "TABLE_A.html")) + "\n{broken\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"chunks\.jsonl:2: invalid chunk record"):
        build_splits(make_config(tmp_path, input_path))
