"""Export deterministic, leakage-safe relationship evaluation datasets."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

SplitName = Literal["train", "validation", "test"]

SPLIT_VERSION = "relationship-pair-split-v1"
DEFAULT_SEED = "clarity-relationships-v1"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VALIDATION_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1


@dataclass(frozen=True)
class RelationshipEdge:
    """One distinct, explicitly documented foreign-key column edge."""

    edge_id: str
    source_doc_id: str
    target_doc_id: str
    source_table: str
    target_table: str
    source_column: str
    target_column: str
    ordinal: int
    relationship_type: str
    evidence_chunk_id: str | None


@dataclass(frozen=True)
class RelationshipTableContext:
    """Bounded document metadata supplied to a relationship query generator."""

    table_name: str
    doc_id: str
    title: str
    source_path: str
    metadata_text: str


@dataclass(frozen=True)
class RelationshipGroup:
    """All directional column edges belonging to one unordered table pair."""

    relationship_group_id: str
    tables: tuple[str, str]
    split: SplitName
    split_version: str
    table_contexts: tuple[RelationshipTableContext, ...]
    edges: tuple[RelationshipEdge, ...]


@dataclass(frozen=True)
class RelationshipDatasetConfig:
    """Configuration for deterministic pair splitting and bounded sampling."""

    db_path: Path
    output_dir: Path
    seed: str = DEFAULT_SEED
    train_ratio: float = DEFAULT_TRAIN_RATIO
    validation_ratio: float = DEFAULT_VALIDATION_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO
    train_size: int = 8_000
    validation_size: int = 200
    test_size: int = 500

    def validate(self) -> None:
        if not self.db_path.is_file():
            raise ValueError(f"RAG index does not exist: {self.db_path}")
        if not self.seed.strip():
            raise ValueError("seed must not be blank")
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if not all(math.isfinite(ratio) and ratio > 0 for ratio in ratios):
            raise ValueError("all split ratios must be finite and greater than zero")
        if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("split ratios must sum to 1.0")
        if self.train_size < 1 or self.validation_size < 1 or self.test_size < 1:
            raise ValueError("train_size, validation_size, and test_size must be at least 1")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_identifier(value: str) -> str:
    return value.strip().upper()


def canonical_pair(table_a: str, table_b: str) -> tuple[str, str]:
    """Return an order-independent, normalized table-pair identity."""
    return tuple(sorted((_normalize_identifier(table_a), _normalize_identifier(table_b))))  # type: ignore[return-value]


def assign_pair_split(pair: tuple[str, str], config: RelationshipDatasetConfig) -> SplitName:
    """Assign a pair by stable hash threshold so future additions do not move it."""
    digest = hashlib.sha256(f"{config.seed}\0{pair[0]}\0{pair[1]}".encode()).digest()
    score = int.from_bytes(digest[:8], "big") / 2**64
    if score < config.train_ratio:
        return "train"
    if score < config.train_ratio + config.validation_ratio:
        return "validation"
    return "test"


def _validate_schema(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'table_relationships'"
    ).fetchone()
    if row is None:
        raise ValueError("RAG index has no table_relationships table")
    required = {
        "relationship_id",
        "source_doc_id",
        "target_doc_id",
        "source_table",
        "target_table",
        "source_column",
        "target_column",
        "ordinal",
        "relationship_type",
        "evidence_chunk_id",
    }
    actual = {str(row[1]) for row in conn.execute("PRAGMA table_info(table_relationships)")}
    missing = sorted(required - actual)
    if missing:
        raise ValueError(f"table_relationships is missing columns: {', '.join(missing)}")
    existing_tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('docs', 'chunks')"
        )
    }
    if existing_tables != {"docs", "chunks"}:
        raise ValueError("RAG index must contain docs and chunks tables")


def _load_table_contexts(conn: sqlite3.Connection) -> dict[str, RelationshipTableContext]:
    """Load the single metadata chunk for each table once, keyed by document ID."""
    contexts: dict[str, RelationshipTableContext] = {}
    for row in conn.execute(
        """SELECT d.doc_id, d.source_path, d.title, c.text
             FROM docs d
             LEFT JOIN chunks c ON c.doc_id = d.doc_id AND c.category = 'metadata'
            ORDER BY d.doc_id, c.chunk_index"""
    ):
        doc_id = str(row[0])
        if doc_id in contexts:
            continue
        source_path = str(row[1])
        table_name = _normalize_identifier(Path(source_path).stem)
        contexts[doc_id] = RelationshipTableContext(
            table_name=table_name,
            doc_id=doc_id,
            title=str(row[2]),
            source_path=source_path,
            metadata_text=str(row[3]) if row[3] is not None else "",
        )
    return contexts


def _edge_id(values: tuple[Any, ...]) -> str:
    return _sha256_text("\0".join(str(value) for value in values))


def iter_relationship_groups(
    conn: sqlite3.Connection,
    config: RelationshipDatasetConfig,
) -> Iterator[RelationshipGroup]:
    """Stream exact-edge-deduplicated groups without loading the graph into memory."""
    _validate_schema(conn)
    contexts_by_doc_id = _load_table_contexts(conn)
    rows = conn.execute(
        """
        SELECT MIN(source_doc_id) AS source_doc_id,
               MIN(target_doc_id) AS target_doc_id,
               UPPER(TRIM(source_table)) AS source_table,
               UPPER(TRIM(target_table)) AS target_table,
               UPPER(TRIM(source_column)) AS source_column,
               UPPER(TRIM(target_column)) AS target_column,
               ordinal,
               relationship_type,
               MIN(evidence_chunk_id) AS evidence_chunk_id
          FROM table_relationships
         WHERE target_doc_id IS NOT NULL
           AND TRIM(source_table) <> ''
           AND TRIM(target_table) <> ''
           AND TRIM(source_column) <> ''
           AND TRIM(target_column) <> ''
         GROUP BY UPPER(TRIM(source_table)), UPPER(TRIM(target_table)),
                  UPPER(TRIM(source_column)), UPPER(TRIM(target_column)),
                  ordinal, relationship_type
         ORDER BY MIN(UPPER(TRIM(source_table)), UPPER(TRIM(target_table))),
                  MAX(UPPER(TRIM(source_table)), UPPER(TRIM(target_table))),
                  source_table, target_table, ordinal, source_column, target_column
        """
    )

    active_pair: tuple[str, str] | None = None
    active_edges: list[RelationshipEdge] = []

    def build_group() -> RelationshipGroup:
        assert active_pair is not None
        group_id = _sha256_text(f"pair\0{active_pair[0]}\0{active_pair[1]}")
        contexts_by_table: dict[str, RelationshipTableContext] = {}
        for edge in active_edges:
            for table_name, doc_id in (
                (edge.source_table, edge.source_doc_id),
                (edge.target_table, edge.target_doc_id),
            ):
                context = contexts_by_doc_id.get(doc_id)
                if context is None:
                    raise ValueError(f"No document context found for relationship table {table_name}")
                contexts_by_table.setdefault(
                    table_name,
                    RelationshipTableContext(
                        table_name=table_name,
                        doc_id=context.doc_id,
                        title=context.title,
                        source_path=context.source_path,
                        metadata_text=context.metadata_text,
                    ),
                )
        return RelationshipGroup(
            relationship_group_id=group_id,
            tables=active_pair,
            split=assign_pair_split(active_pair, config),
            split_version=SPLIT_VERSION,
            table_contexts=tuple(
                contexts_by_table[table] for table in sorted(contexts_by_table)
            ),
            edges=tuple(active_edges),
        )

    for row in rows:
        source_table = str(row[2])
        target_table = str(row[3])
        pair = canonical_pair(source_table, target_table)
        if active_pair is not None and pair != active_pair:
            yield build_group()
            active_edges = []
        active_pair = pair
        semantic_identity = (
            source_table,
            target_table,
            str(row[4]),
            str(row[5]),
            int(row[6]),
            str(row[7]),
        )
        active_edges.append(
            RelationshipEdge(
                edge_id=_edge_id(semantic_identity),
                source_doc_id=str(row[0]),
                target_doc_id=str(row[1]),
                source_table=source_table,
                target_table=target_table,
                source_column=str(row[4]),
                target_column=str(row[5]),
                ordinal=int(row[6]),
                relationship_type=str(row[7]),
                evidence_chunk_id=str(row[8]) if row[8] is not None else None,
            )
        )
    if active_pair is not None:
        yield build_group()


def _sample_priority(group: RelationshipGroup, seed: str) -> int:
    digest = hashlib.sha256(
        f"{seed}\0sample\0{group.relationship_group_id}".encode()
    ).digest()
    return int.from_bytes(digest[:16], "big")


def _retain_sample(
    heap: list[tuple[int, str, RelationshipGroup]],
    group: RelationshipGroup,
    size: int,
    seed: str,
) -> None:
    # Negative priority turns heapq into a bounded max-heap. The lowest stable
    # hashes win, making the sample independent of SQLite iteration order.
    priority = _sample_priority(group, seed)
    item = (-priority, group.relationship_group_id, group)
    if len(heap) < size:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _write_atomic(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            for line in lines:
                output.write(line)
                output.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _json_line(group: RelationshipGroup) -> str:
    return json.dumps(asdict(group), sort_keys=True, ensure_ascii=False)


def build_relationship_dataset(config: RelationshipDatasetConfig) -> dict[str, Any]:
    """Build bounded validation/test samples and an auditable split report."""
    config.validate()
    split_pair_counts: Counter[str] = Counter()
    split_edge_counts: Counter[str] = Counter()
    train_heap: list[tuple[int, str, RelationshipGroup]] = []
    validation_heap: list[tuple[int, str, RelationshipGroup]] = []
    test_heap: list[tuple[int, str, RelationshipGroup]] = []

    uri = f"file:{config.db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        for group in iter_relationship_groups(conn, config):
            split_pair_counts[group.split] += 1
            split_edge_counts[group.split] += len(group.edges)
            if group.split == "train":
                _retain_sample(train_heap, group, config.train_size, config.seed)
            elif group.split == "validation":
                _retain_sample(
                    validation_heap, group, config.validation_size, config.seed
                )
            elif group.split == "test":
                _retain_sample(test_heap, group, config.test_size, config.seed)

    train = sorted((item[2] for item in train_heap), key=lambda item: item.relationship_group_id)
    validation = sorted((item[2] for item in validation_heap), key=lambda item: item.relationship_group_id)
    test = sorted((item[2] for item in test_heap), key=lambda item: item.relationship_group_id)
    train_ids = {group.relationship_group_id for group in train}
    validation_ids = {group.relationship_group_id for group in validation}
    test_ids = {group.relationship_group_id for group in test}
    overlaps = {
        "train_validation": train_ids & validation_ids,
        "train_test": train_ids & test_ids,
        "validation_test": validation_ids & test_ids,
    }
    if any(overlaps.values()):
        raise RuntimeError("Relationship pair leakage detected between dataset splits")

    train_path = config.output_dir / "train_relationships.jsonl"
    validation_path = config.output_dir / "validation_relationships.jsonl"
    test_path = config.output_dir / "test_relationships.jsonl"
    report_path = config.output_dir / "split_report.json"
    _write_atomic(train_path, (_json_line(group) for group in train))
    _write_atomic(validation_path, (_json_line(group) for group in validation))
    _write_atomic(test_path, (_json_line(group) for group in test))

    report: dict[str, Any] = {
        "split_version": SPLIT_VERSION,
        "seed": config.seed,
        "db_path": str(config.db_path),
        "ratios": {
            "train": config.train_ratio,
            "validation": config.validation_ratio,
            "test": config.test_ratio,
        },
        "pair_counts_by_split": {
            split: split_pair_counts[split] for split in ("train", "validation", "test")
        },
        "edge_counts_by_split": {
            split: split_edge_counts[split] for split in ("train", "validation", "test")
        },
        "requested_sample_pairs": {
            "train": config.train_size,
            "validation": config.validation_size,
            "test": config.test_size,
        },
        "sample_pair_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "sample_edge_counts": {
            "train": sum(len(group.edges) for group in train),
            "validation": sum(len(group.edges) for group in validation),
            "test": sum(len(group.edges) for group in test),
        },
        "pair_overlap_counts": {
            name: len(overlap) for name, overlap in overlaps.items()
        },
        "train_output": str(train_path),
        "validation_output": str(validation_path),
        "test_output": str(test_path),
        "train_output_hash": _sha256_text(train_path.read_text(encoding="utf-8")),
        "validation_output_hash": _sha256_text(validation_path.read_text(encoding="utf-8")),
        "test_output_hash": _sha256_text(test_path.read_text(encoding="utf-8")),
    }
    _write_atomic(
        report_path,
        [json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export leakage-safe relationship validation and test datasets."
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--train-size", type=int, default=8_000)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=500)
    args = parser.parse_args()
    config = RelationshipDatasetConfig(
        db_path=args.db,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
    try:
        report = build_relationship_dataset(config)
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(
        "Exported "
        f"{report['sample_pair_counts']['train']:,} train, "
        f"{report['sample_pair_counts']['validation']:,} validation, and "
        f"{report['sample_pair_counts']['test']:,} test relationship pairs; "
        "cross-split overlap: 0."
    )


if __name__ == "__main__":
    main()
