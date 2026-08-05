from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pytest
from retrieval.chunk_models import SplitChunkRecord
from retrieval.index_contract import INDEX_CHUNKER_VERSION, INDEX_SCHEMA_VERSION
from retrieval.indexer import create_schema

from clarity_agent_evals.genq.column_extractor import (
    EXTRACTOR_VERSION,
    ColumnExtractorConfig,
    build_embedding_text,
    extract_column_chunks,
    extract_description,
    stratified_sample,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _column_text(
    table: str, column: str, description: str, column_type: str = "VARCHAR (25)"
) -> str:
    """Render a chunk in the exact shape the indexer writes."""
    return (
        f"Table {table}. Column {column}. INI: {table[:3]}. Item: 17. Type: {column_type}. "
        f"Deprecated?: No. Discontinued?: No. Preserved?: No. Character Replacement?: No. "
        f"EHI Status: Not Exported. Description: {description}"
    )


def _create_index(path: Path, documents: list[tuple[str, list[str]]]) -> None:
    """Create (source path, column chunk texts) documents for extraction."""
    with sqlite3.connect(path) as conn:
        create_schema(conn)
        conn.executemany(
            "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
            [
                ("index_version", "test-index-v1"),
                ("schema_version", INDEX_SCHEMA_VERSION),
                ("chunker_version", INDEX_CHUNKER_VERSION),
            ],
        )
        for source_path, texts in documents:
            document_id = _hash(source_path)
            conn.execute(
                "INSERT INTO docs (doc_id, source_path, title, source_hash) VALUES (?, ?, ?, ?)",
                (document_id, source_path, f"{source_path} - Dictionary", _hash(source_path)),
            )
            for chunk_index, text in enumerate(texts):
                # Mirror the indexer: column chunks carry column_parser's readable
                # logical ID, not a content hash.
                header = re.match(r"^Table (\S+)\. Column (\S+)\.", text)
                logical_id = (
                    f"{header.group(1)}__COLUMN_DEFINITION__{header.group(2)}"
                    if header
                    else _hash(f"{source_path}:{chunk_index}")
                )
                conn.execute(
                    """INSERT INTO chunks
                       (chunk_id, doc_id, chunk_index, category, heading_path,
                        text, token_count, text_hash)
                       VALUES (?, ?, ?, 'column_info', ?, ?, ?, ?)""",
                    (
                        logical_id,
                        document_id,
                        chunk_index,
                        f"{source_path.removesuffix('.html')} > Column-Information > COL",
                        text,
                        len(text.split()),
                        _hash(text),
                    ),
                )


def _run(tmp_path: Path, documents: list[tuple[str, list[str]]], **kwargs: object):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "index.sqlite"
    _create_index(db_path, documents)
    report = extract_column_chunks(
        ColumnExtractorConfig(
            db_path=db_path,
            output_path=tmp_path / "columns.jsonl",
            report_path=tmp_path / "report.json",
            **kwargs,  # type: ignore[arg-type]
        )
    )
    records = [
        SplitChunkRecord.model_validate_json(line)
        for line in (tmp_path / "columns.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return report, records


def test_extracts_column_records_and_strips_constant_scaffolding(tmp_path: Path) -> None:
    report, records = _run(
        tmp_path,
        [
            (
                "A0H_MAP.html",
                [
                    _column_text(
                        "A0H_MAP",
                        "INTERNAL_ID",
                        "The unique identifier assigned to this hyperspace access record.",
                    ),
                    _column_text(
                        "A0H_MAP",
                        "LINE",
                        "The line number for the information associated with this record.",
                    ),
                ],
            )
        ],
    )

    assert report.output_record_count == 2
    assert report.unparsed_header_count == 0
    assert [record.column_name for record in records] == ["INTERNAL_ID", "LINE"]
    assert all(record.chunk_type == "column_definition" for record in records)
    assert all(record.parser_version == EXTRACTOR_VERSION for record in records)
    # The boilerplate that is identical across the corpus must not survive.
    for record in records:
        assert "Deprecated?" not in record.text
        assert "EHI Status" not in record.text
        assert "Character Replacement?" not in record.text
    assert records[0].text == (
        "Table: A0H_MAP\nColumn: INTERNAL_ID\nType: VARCHAR (25)\n"
        "Description: The unique identifier assigned to this hyperspace access record."
    )


def test_collapses_columns_that_share_a_description(tmp_path: Path) -> None:
    shared = "The Community ID (CID) of the instance that owns this record or line."
    report, records = _run(
        tmp_path,
        [
            ("A.html", [_column_text("A", "CID", shared)]),
            ("B.html", [_column_text("B", "CID", shared)]),
            ("C.html", [_column_text("C", "CID", shared)]),
            (
                "D.html",
                [
                    _column_text(
                        "D",
                        "OTHER",
                        "A genuinely different description that clears the length bar.",
                    )
                ],
            ),
        ],
    )

    assert report.output_record_count == 2
    assert report.duplicate_description_count == 2
    assert report.duplicate_group_count == 1
    # The surviving representative is the first in deterministic source order.
    assert [record.table_name for record in records] == ["A", "D"]


def test_no_dedup_keeps_every_column(tmp_path: Path) -> None:
    shared = "The Community ID (CID) of the instance that owns this record or line."
    report, records = _run(
        tmp_path,
        [
            ("A.html", [_column_text("A", "CID", shared)]),
            ("B.html", [_column_text("B", "CID", shared)]),
        ],
        deduplicate=False,
    )

    assert report.output_record_count == 2
    assert report.duplicate_description_count == 1
    assert len(records) == 2


def test_skips_short_and_missing_descriptions(tmp_path: Path) -> None:
    report, records = _run(
        tmp_path,
        [
            ("A.html", [_column_text("A", "SHORT", "Too brief.")]),
            (
                "B.html",
                ["Table B. Column NODESC. INI: B. Item: 1. Type: INTEGER. Deprecated?: No."],
            ),
            (
                "C.html",
                [_column_text("C", "GOOD", "A description that comfortably clears the bar.")],
            ),
        ],
    )

    assert report.short_description_count == 1
    assert report.missing_description_count == 1
    assert report.output_record_count == 1
    assert records[0].column_name == "GOOD"


def test_unparsable_header_is_counted_not_crashed(tmp_path: Path) -> None:
    report, records = _run(
        tmp_path,
        [
            ("A.html", ["Totally unexpected shape. Description: This description is long enough."]),
            (
                "B.html",
                [_column_text("B", "GOOD", "A description that comfortably clears the bar.")],
            ),
        ],
    )

    assert report.unparsed_header_count == 1
    assert report.output_record_count == 1
    assert records[0].table_name == "B"


def test_output_hash_is_stable_across_runs(tmp_path: Path) -> None:
    documents = [
        ("A.html", [_column_text("A", "ONE", "A description that comfortably clears the bar.")]),
        ("B.html", [_column_text("B", "TWO", "Another description that clears the bar nicely.")]),
    ]
    first, _ = _run(tmp_path / "first", documents)
    second, _ = _run(tmp_path / "second", documents)
    assert first.output_hash == second.output_hash


def test_limit_applies_after_deduplication(tmp_path: Path) -> None:
    shared = "The Community ID (CID) of the instance that owns this record or line."
    report, records = _run(
        tmp_path,
        [
            ("A.html", [_column_text("A", "CID", shared)]),
            ("B.html", [_column_text("B", "CID", shared)]),
            (
                "C.html",
                [
                    _column_text(
                        "C",
                        "OTHER",
                        "A genuinely different description that clears the length bar.",
                    )
                ],
            ),
        ],
        limit=2,
    )

    # Two distinct descriptions exist; the duplicate must not consume the cap.
    assert report.output_record_count == 2
    assert {record.table_name for record in records} == {"A", "C"}


def test_config_rejects_overlapping_paths(tmp_path: Path) -> None:
    shared = tmp_path / "same.jsonl"
    config = ColumnExtractorConfig(
        db_path=tmp_path / "index.sqlite", output_path=shared, report_path=shared
    )
    with pytest.raises(ValueError, match="must be different"):
        config.validate()


def test_extract_description_handles_absent_marker() -> None:
    assert extract_description("Table A. Column B. Type: INTEGER.") is None
    assert extract_description("Table A. Description: ") is None
    assert extract_description("Table A. Description:  spaced   out ") == "spaced out"


def test_build_embedding_text_omits_unknown_type() -> None:
    assert build_embedding_text("T", "C", None, "A description.") == (
        "Table: T\nColumn: C\nDescription: A description."
    )


def test_stratified_sample_spreads_across_table_families(tmp_path: Path) -> None:
    documents = [
        (
            f"{family}_{index}.html",
            [
                _column_text(
                    f"{family}_{index}",
                    f"COL{index}",
                    f"Description number {index} describing the {family} table family in detail.",
                )
            ],
        )
        for family, count in (("CLARITY", 10), ("PAT", 4), ("HSP", 2))
        for index in range(count)
    ]
    _, records = _run(tmp_path, documents)

    sample = stratified_sample(records, 6)
    assert len(sample) == 6
    families = {record.table_name.split("_", 1)[0] for record in sample}
    # A uniform draw would return CLARITY rows only; every family must appear.
    assert families == {"CLARITY", "PAT", "HSP"}
    assert len({record.chunk_id for record in sample}) == 6


def test_stratified_sample_is_deterministic(tmp_path: Path) -> None:
    documents = [
        (
            f"T_{index}.html",
            [
                _column_text(
                    f"T_{index}",
                    "COL",
                    f"Description number {index} describing this particular column in detail.",
                )
            ],
        )
        for index in range(20)
    ]
    _, records = _run(tmp_path, documents)
    assert [r.chunk_id for r in stratified_sample(records, 5)] == [
        r.chunk_id for r in stratified_sample(records, 5)
    ]


def test_stratified_sample_rejects_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        stratified_sample([], 0)


def test_report_round_trips_as_json(tmp_path: Path) -> None:
    report, _ = _run(
        tmp_path,
        [("A.html", [_column_text("A", "ONE", "A description that comfortably clears the bar.")])],
    )
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["output_hash"] == report.output_hash
    assert written["extractor_version"] == EXTRACTOR_VERSION


def test_stratified_sample_flag_spreads_across_families(tmp_path: Path) -> None:
    documents = [
        (
            f"{family}_{index}.html",
            [
                _column_text(
                    f"{family}_{index}",
                    f"COL{index}",
                    f"Description number {index} describing the {family} table family in detail.",
                )
            ],
        )
        for family, count in (("CLARITY", 12), ("PAT", 4), ("HSP", 2))
        for index in range(count)
    ]
    report, records = _run(tmp_path, documents, sample_size=6)

    assert report.output_record_count == 6
    assert report.requested_sample == 6
    assert {r.table_name.split("_", 1)[0] for r in records} == {"CLARITY", "PAT", "HSP"}


def test_sample_size_and_limit_cannot_be_combined(tmp_path: Path) -> None:
    config = ColumnExtractorConfig(
        db_path=tmp_path / "index.sqlite",
        output_path=tmp_path / "out.jsonl",
        report_path=tmp_path / "report.json",
        limit=10,
        sample_size=10,
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        config.validate()


def test_chunk_id_is_the_indexer_id_not_a_locally_minted_one(tmp_path: Path) -> None:
    """Generation records must join back to the index.

    `column_parser` assigns a readable logical chunk_id that `INDEX_CHUNKER_VERSION`
    pins, so two people indexing the same Epic HTML get identical IDs. Deriving a
    private ID here would create a second identifier space that cannot be joined
    to the index and could diverge between machines.
    """
    report, records = _run(
        tmp_path,
        [
            (
                "A0H_MAP.html",
                [
                    _column_text(
                        "A0H_MAP",
                        "INTERNAL_ID",
                        "The unique identifier assigned to this hyperspace access record.",
                    )
                ],
            )
        ],
    )

    assert report.output_record_count == 1
    assert records[0].chunk_id == "A0H_MAP__COLUMN_DEFINITION__INTERNAL_ID"
    assert not records[0].chunk_id.startswith("column:")


def test_chunk_ids_survive_a_rebuild_of_the_same_html(tmp_path: Path) -> None:
    documents = [
        (
            "A0H_MAP.html",
            [
                _column_text("A0H_MAP", "CID", "The community ID for the campaign record here."),
                _column_text("A0H_MAP", "LINE", "The line number for information on this record."),
            ],
        )
    ]
    _, first = _run(tmp_path / "one", documents)
    _, second = _run(tmp_path / "two", documents)
    assert [r.chunk_id for r in first] == [r.chunk_id for r in second]
