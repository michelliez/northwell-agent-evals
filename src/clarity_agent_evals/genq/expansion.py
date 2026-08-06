"""Build an FTS doc2query expansion corpus from retained synthetic queries.

Expansion text is index-side vocabulary, never an evaluation label: a retained
query is appended to the FTS row of its table's first chunk so lexical search
can match analyst phrasing the documentation lacks. The quality bar is
therefore novel-term contribution, not leakage: a query whose every meaningful
token already appears in the indexed text cannot change any match and is
dropped. Never evaluate synthetic queries against an index expanded from the
same queries; the reviewed human benchmark remains the only judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from clarity_agent_evals.query_filter import _meaningful_tokens

LOGGER = logging.getLogger(__name__)

EXPANSION_VERSION = "fts-doc2query-expansion-v1"
DEFAULT_MIN_NOVEL_TERMS = 2


@dataclass(frozen=True)
class ExpansionConfig:
    queries_path: Path
    db_path: Path
    output_path: Path
    report_path: Path
    min_novel_terms: int = DEFAULT_MIN_NOVEL_TERMS

    def validate(self) -> None:
        if self.min_novel_terms < 1:
            raise ValueError("min_novel_terms must be at least 1")
        if self.output_path in {self.queries_path, self.db_path}:
            raise ValueError("output_path must differ from the input paths")


@dataclass(frozen=True)
class _TargetChunk:
    doc_id: str
    source_path: str
    chunk_id: str
    indexed_tokens: frozenset[str]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_queries(path: Path) -> tuple[list[dict[str, str]], str]:
    """Read retained query records, requiring only the fields expansion uses.

    Retained sets carry filter-version-specific fields, so this accepts any
    superset and fails closed on the three fields that matter.
    """
    raw_bytes = path.read_bytes()
    if not raw_bytes.strip():
        raise ValueError(f"retained query JSONL is empty: {path}")
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(raw_bytes.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        row = json.loads(line)
        record = {
            "query_id": str(row.get("query_id", "")).strip(),
            "query": str(row.get("query", "")).strip(),
            "source_file": str(row.get("source_file", "")).strip(),
        }
        if not all(record.values()):
            raise ValueError(f"{path}:{line_number}: record needs query_id, query, and source_file")
        records.append(record)
    return records, _sha256_bytes(raw_bytes)


def _load_target_chunks(conn: sqlite3.Connection) -> dict[str, _TargetChunk]:
    """Map upper-cased document file names to each document's first chunk.

    The first chunk carries the table's description section, so document-level
    retrieval benefits as soon as any expansion term matches. Tokens already
    indexed for that row (text, heading, title, path, category) are recorded so
    novelty is measured against what FTS can already reach, not just the
    passage the query was generated from. Name collisions keep the smallest
    source path, mirroring the application's relationship resolver.
    """
    documents = {
        str(row[0]): (str(row[1]), str(row[2]))
        for row in conn.execute("SELECT doc_id, source_path, title FROM docs")
    }

    # One streaming pass instead of a correlated MIN subquery: the chunks table
    # holds ~half a million rows and carries no index on doc_id.
    first_chunk_by_doc: dict[str, tuple[int, str, str, str, str]] = {}
    for row in conn.execute(
        "SELECT doc_id, chunk_id, chunk_index, heading_path, category, text FROM chunks"
    ):
        doc_id = str(row[0])
        chunk_index = int(row[2])
        best = first_chunk_by_doc.get(doc_id)
        if best is None or chunk_index < best[0]:
            first_chunk_by_doc[doc_id] = (
                chunk_index,
                str(row[1]),
                str(row[3]),
                str(row[4]),
                str(row[5]),
            )

    targets: dict[str, _TargetChunk] = {}
    for doc_id, (_, chunk_id, heading_path, category, text) in first_chunk_by_doc.items():
        document = documents.get(doc_id)
        if document is None:
            continue
        source_path, title = document
        file_name = source_path.rsplit("/", 1)[-1].upper()
        existing = targets.get(file_name)
        if existing is not None and existing.source_path <= source_path:
            continue
        indexed_text = " ".join(
            (text, heading_path, title, category, source_path.replace("/", " "))
        )
        targets[file_name] = _TargetChunk(
            doc_id=doc_id,
            source_path=source_path,
            chunk_id=chunk_id,
            indexed_tokens=frozenset(_meaningful_tokens(indexed_text)),
        )
    if not targets:
        raise ValueError("index has no chunks - wrong database?")
    return targets


def build_expansion_corpus(config: ExpansionConfig) -> dict[str, object]:
    config.validate()
    queries, queries_hash = _load_queries(config.queries_path)

    conn = sqlite3.connect(f"file:{config.db_path.as_posix()}?mode=ro", uri=True)
    try:
        index_metadata = dict(conn.execute("SELECT key, value FROM index_metadata"))
        targets = _load_target_chunks(conn)
    finally:
        conn.close()

    retained_by_chunk: dict[str, list[dict[str, object]]] = defaultdict(list)
    target_by_chunk: dict[str, _TargetChunk] = {}
    novel_term_histogram: Counter[int] = Counter()
    unmatched_documents = 0
    dropped_low_novelty = 0

    for record in queries:
        target = targets.get(record["source_file"].rsplit("/", 1)[-1].upper())
        if target is None:
            unmatched_documents += 1
            continue
        novel_terms = sorted(_meaningful_tokens(record["query"]) - target.indexed_tokens)
        novel_term_histogram[len(novel_terms)] += 1
        if len(novel_terms) < config.min_novel_terms:
            dropped_low_novelty += 1
            continue
        target_by_chunk[target.chunk_id] = target
        retained_by_chunk[target.chunk_id].append(
            {
                "query_id": record["query_id"],
                "query": record["query"],
                "novel_terms": novel_terms,
            }
        )

    if not retained_by_chunk:
        raise ValueError("no queries survived expansion filtering - wrong inputs?")

    output_lines = [
        json.dumps(
            {
                "chunk_id": chunk_id,
                "doc_id": target_by_chunk[chunk_id].doc_id,
                "source_path": target_by_chunk[chunk_id].source_path,
                "queries": retained_by_chunk[chunk_id],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        for chunk_id in sorted(retained_by_chunk)
    ]
    output_text = "\n".join(output_lines) + "\n"
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(output_text, encoding="utf-8", newline="\n")

    retained_query_count = sum(len(rows) for rows in retained_by_chunk.values())
    report = {
        "expansion_version": EXPANSION_VERSION,
        "min_novel_terms": config.min_novel_terms,
        "queries_path": str(config.queries_path),
        "input_queries_hash": queries_hash,
        "input_query_count": len(queries),
        "index_version": index_metadata.get("index_version"),
        "chunker_version": index_metadata.get("chunker_version"),
        "retained_query_count": retained_query_count,
        "dropped_low_novelty_count": dropped_low_novelty,
        "unmatched_document_count": unmatched_documents,
        "expanded_chunk_count": len(retained_by_chunk),
        "novel_term_histogram": {
            str(count): occurrences for count, occurrences in sorted(novel_term_histogram.items())
        },
        "output_hash": _sha256_bytes(output_text.encode("utf-8")),
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Filter retained synthetic queries into an FTS expansion corpus."
    )
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-novel-terms", type=int, default=DEFAULT_MIN_NOVEL_TERMS)
    args = parser.parse_args()

    report = build_expansion_corpus(
        ExpansionConfig(
            queries_path=args.queries,
            db_path=args.db,
            output_path=args.output,
            report_path=args.report,
            min_novel_terms=args.min_novel_terms,
        )
    )
    print(
        f"Retained {report['retained_query_count']:,}/{report['input_query_count']:,} queries "
        f"over {report['expanded_chunk_count']:,} chunks "
        f"(dropped {report['dropped_low_novelty_count']:,} low-novelty, "
        f"{report['unmatched_document_count']:,} unmatched documents)"
    )
    print(f"Corpus: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
