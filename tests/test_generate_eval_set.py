from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from clarity_agent_evals.generate_eval_set import (
    NonRelationshipEvalConfig,
    RelationshipEvalConfig,
    run_non_relationship_eval,
    run_relationship_eval,
)

HASH = hashlib.sha256(b"test-value").hexdigest()


class FakeQueryGenerator:
    model_name = "fake-haiku"
    device_name = "fake-api"

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
        del max_input_tokens, max_query_tokens, top_p, seed
        return [
            [
                f"Which documented concept is described in example {index}?"
                for index in range(queries_per_passage)
            ]
            for _passage in passages
        ]


def _write_chunks(path: Path, count: int = 30) -> None:
    rows = []
    for index in range(count):
        table = f"TABLE_{index}"
        rows.append(
            {
                "chunk_id": f"{table}__COLUMN__ID",
                "source_file": f"{table}.html",
                "source_hash": HASH,
                "table_name": table,
                "column_name": "ID",
                "chunk_type": "column_definition",
                "section_name": "Column Information",
                "text": (
                    f"The {table} documentation describes a stable business identifier "
                    "used for approved analytical workflows and reporting."
                ),
                "text_hash": hashlib.sha256(table.encode()).hexdigest(),
                "parser_version": "epic-genq-html-v1",
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_relationship_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE docs (doc_id TEXT PRIMARY KEY, source_path TEXT, title TEXT, source_hash TEXT)"
        )
        conn.execute(
            """CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY, doc_id TEXT, chunk_index INTEGER,
                category TEXT, heading_path TEXT, text TEXT, token_count INTEGER,
                text_hash TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE table_relationships (
                relationship_id TEXT PRIMARY KEY, source_doc_id TEXT NOT NULL,
                target_doc_id TEXT, source_table TEXT NOT NULL, target_table TEXT NOT NULL,
                source_column TEXT NOT NULL, target_column TEXT NOT NULL,
                ordinal INTEGER NOT NULL, relationship_type TEXT NOT NULL,
                evidence_chunk_id TEXT
            )"""
        )
        for index in range(30):
            source_doc = f"source-{index}"
            target_doc = f"target-{index}"
            source_table = f"SOURCE_{index}"
            target_table = f"TARGET_{index}"
            for doc_id, table in ((source_doc, source_table), (target_doc, target_table)):
                conn.execute(
                    "INSERT INTO docs VALUES (?, ?, ?, ?)",
                    (doc_id, f"{table}.html", table, f"hash-{doc_id}"),
                )
                conn.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"metadata-{doc_id}",
                        doc_id,
                        0,
                        "metadata",
                        f"{table} > Metadata",
                        f"Table {table}. Approved business documentation.",
                        8,
                        f"text-hash-{doc_id}",
                    ),
                )
            conn.execute(
                "INSERT INTO table_relationships VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"relationship-{index}",
                    source_doc,
                    target_doc,
                    source_table,
                    target_table,
                    "SOURCE_ID",
                    "TARGET_ID",
                    1,
                    "foreign_key",
                    f"metadata-{source_doc}",
                ),
            )


def test_non_relationship_workflow_creates_reproducible_artifacts(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    output_dir = tmp_path / "ordinary"

    report = run_non_relationship_eval(
        NonRelationshipEvalConfig(
            chunks_path=chunks_path,
            output_dir=output_dir,
            splits=("train", "validation", "test"),
            min_passage_chars=20,
        ),
        generator=FakeQueryGenerator(),
    )

    assert report["selected_chunk_count"] == 30
    assert report["generated_query_count"] == 60
    assert (output_dir / "corpus/chunks_with_splits.jsonl").is_file()
    assert (output_dir / "queries/generated.jsonl").is_file()
    assert (output_dir / "reviewed/filter_report.json").is_file()
    assert json.loads((output_dir / "run_report.json").read_text())["mode"] == "non-relationship"


def test_relationship_manifest_only_workflow_avoids_model_calls(tmp_path: Path) -> None:
    db_path = tmp_path / "relationships.sqlite"
    _make_relationship_database(db_path)
    output_dir = tmp_path / "relationship"

    report = run_relationship_eval(
        RelationshipEvalConfig(
            db_path=db_path,
            output_dir=output_dir,
            train_size=3,
            validation_size=3,
            test_size=3,
            manifest_only=True,
        )
    )

    assert report["manifest_only"] is True
    assert report["generation_reports"] == {}
    assert (output_dir / "manifests/train_relationships.jsonl").is_file()
    assert (output_dir / "manifests/validation_relationships.jsonl").is_file()
    assert (output_dir / "manifests/test_relationships.jsonl").is_file()
    assert all(count == 0 for count in report["dataset_report"]["pair_overlap_counts"].values())
