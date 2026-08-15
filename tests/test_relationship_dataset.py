from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from clarity_agent_evals.relationship_dataset import (
    RelationshipDatasetConfig,
    assign_pair_split,
    build_relationship_dataset,
    canonical_pair,
)


def make_database(path: Path, pair_count: int = 200) -> None:
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
                relationship_id TEXT PRIMARY KEY,
                source_doc_id TEXT NOT NULL,
                target_doc_id TEXT,
                source_table TEXT NOT NULL,
                target_table TEXT NOT NULL,
                source_column TEXT NOT NULL,
                target_column TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                relationship_type TEXT NOT NULL,
                evidence_chunk_id TEXT
            )"""
        )
        rows = []
        docs = []
        chunks = []
        for index in range(pair_count):
            values = (
                f"rel-{index}",
                f"doc-source-{index}",
                f"doc-target-{index}",
                f"SOURCE_{index}",
                f"TARGET_{index}",
                "SOURCE_ID",
                "TARGET_ID",
                1,
                "foreign_key",
                f"evidence-{index}",
            )
            rows.append(values)
            for side in ("source", "target"):
                doc_id = f"doc-{side}-{index}"
                table = f"{side.upper()}_{index}"
                docs.append((doc_id, f"{table}.html", table, f"hash-{doc_id}"))
                chunks.append(
                    (
                        f"metadata-{doc_id}",
                        doc_id,
                        0,
                        "metadata",
                        f"{table} > Metadata",
                        f"Table {table}. Description: business context for {table}.",
                        10,
                        f"text-hash-{doc_id}",
                    )
                )
        # This row is an exact semantic duplicate with a different storage ID.
        rows.append(("duplicate", *rows[0][1:]))
        # Unresolved relationships are intentionally excluded.
        rows.append(
            (
                "unresolved",
                "doc-unresolved",
                None,
                "UNRESOLVED_A",
                "UNRESOLVED_B",
                "ID",
                "ID",
                1,
                "foreign_key",
                None,
            )
        )
        docs.extend(
            [
                ("doc-unresolved", "UNRESOLVED_A.html", "UNRESOLVED_A", "hash-a"),
                ("doc-unresolved-target", "UNRESOLVED_B.html", "UNRESOLVED_B", "hash-b"),
            ]
        )
        conn.executemany("INSERT INTO docs VALUES (?, ?, ?, ?)", docs)
        conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)", chunks)
        conn.executemany(
            "INSERT INTO table_relationships VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_pair_identity_is_order_independent() -> None:
    assert canonical_pair(" table_b ", "TABLE_A") == ("TABLE_A", "TABLE_B")


def test_dataset_is_deterministic_and_validation_test_do_not_leak(tmp_path: Path) -> None:
    db_path = tmp_path / "index.sqlite"
    make_database(db_path)
    config = RelationshipDatasetConfig(
        db_path=db_path,
        output_dir=tmp_path / "relationships",
        train_size=7,
        validation_size=8,
        test_size=9,
    )

    first = build_relationship_dataset(config)
    first_validation = (config.output_dir / "validation_relationships.jsonl").read_bytes()
    first_test = (config.output_dir / "test_relationships.jsonl").read_bytes()
    second = build_relationship_dataset(config)

    train = read_jsonl(config.output_dir / "train_relationships.jsonl")
    validation = read_jsonl(config.output_dir / "validation_relationships.jsonl")
    test = read_jsonl(config.output_dir / "test_relationships.jsonl")
    train_ids = {str(row["relationship_group_id"]) for row in train}
    validation_ids = {str(row["relationship_group_id"]) for row in validation}
    test_ids = {str(row["relationship_group_id"]) for row in test}

    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids
    assert all(value == 0 for value in first["pair_overlap_counts"].values())
    assert first["pair_counts_by_split"] == second["pair_counts_by_split"]
    assert first_validation == (config.output_dir / "validation_relationships.jsonl").read_bytes()
    assert first_test == (config.output_dir / "test_relationships.jsonl").read_bytes()
    assert all(row["split"] == "validation" for row in validation)
    assert all(row["split"] == "test" for row in test)


def test_exact_edges_are_deduplicated_and_unresolved_edges_are_excluded(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "index.sqlite"
    make_database(db_path, pair_count=1)
    config = RelationshipDatasetConfig(
        db_path=db_path,
        output_dir=tmp_path / "relationships",
        train_ratio=0.01,
        validation_ratio=0.98,
        test_ratio=0.01,
        train_size=5,
        validation_size=5,
        test_size=5,
    )
    # Pick a seed that places the fixture pair in validation.
    for index in range(1_000):
        candidate = RelationshipDatasetConfig(
            **{**config.__dict__, "seed": f"seed-{index}"}
        )
        if assign_pair_split(("SOURCE_0", "TARGET_0"), candidate) == "validation":
            config = candidate
            break

    report = build_relationship_dataset(config)
    validation = read_jsonl(config.output_dir / "validation_relationships.jsonl")

    assert report["pair_counts_by_split"]["validation"] == 1
    assert report["edge_counts_by_split"]["validation"] == 1
    assert len(validation) == 1
    assert len(validation[0]["edges"]) == 1  # type: ignore[arg-type]
    assert len(validation[0]["table_contexts"]) == 2  # type: ignore[arg-type]
