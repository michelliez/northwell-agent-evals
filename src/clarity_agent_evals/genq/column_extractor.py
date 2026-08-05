"""Produce deterministic column-level generation input from the RAG index.

The table-level path (``retrieval.metadata_extractor`` then
``clarity_agent_evals.genq.metadata_to_chunks``) covers one chunk per document. This module
is its column-level counterpart: it reads the ``column_info`` chunks the indexer
already wrote, strips the constant scaffolding, and collapses columns that share
a description.

Deduplication is the whole point. Epic reuses column descriptions across the
schema -- one sentence about Community IDs appears on 14,638 distinct columns --
so generating queries for every member buys no new information and trains an
encoder to separate passages that no encoder can separate. The retrieval corpus
still indexes every column; only generation input is collapsed.

Nothing here runs in the request path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from clarity_agent_evals.dataset_split import SplitConfig, assign_source_split
from retrieval.chunk_models import SplitChunkRecord
from retrieval.metadata_extractor import _write_atomic
from retrieval.search import get_index_version, open_connection

EXTRACTOR_VERSION = "column-info-v1"
SPLIT_VERSION = "metadata-source-hash-split-v1"
DEFAULT_SEED = "epic-genq-v1"
DEFAULT_MIN_DESCRIPTION_CHARS = 40

# `Table A0H_MAP. Column CM_PHY_OWNER_ID. INI: A0H. Item: 17. Type: VARCHAR (25). ...`
# The heading path cannot be used instead: the indexer truncates its leaf, so
# `CM_PHY_OWNER_ID` is stored as `CM_PHY_OWNER_`.
_HEADER = re.compile(
    r"^Table\s+(?P<table>[A-Za-z0-9_#$]+)\.\s+Column\s+(?P<column>[A-Za-z0-9_#$]+)\."
)
_TYPE = re.compile(r"\bType:\s*(?P<type>[A-Za-z ]+(?:\s*\([^)]*\))?)")
_DESCRIPTION_MARKER = "Description: "


class ColumnExtractionReport(BaseModel):
    """Audit record for one column-chunk extraction run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extractor_version: str
    split_version: str
    db_path: str
    output_path: str
    report_path: str
    index_version: str = Field(min_length=1)
    seed: str = Field(min_length=1)
    min_description_chars: int = Field(ge=0)
    requested_limit: int | None
    requested_sample: int | None = None
    input_chunk_count: int = Field(ge=0)
    unparsed_header_count: int = Field(ge=0)
    missing_description_count: int = Field(ge=0)
    short_description_count: int = Field(ge=0)
    duplicate_description_count: int = Field(ge=0)
    duplicate_group_count: int = Field(ge=0)
    output_record_count: int = Field(ge=0)
    source_counts_by_split: dict[str, int]
    output_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ColumnExtractorConfig:
    """Paths and settings for one column-chunk extraction run."""

    db_path: Path
    output_path: Path
    report_path: Path
    seed: str = DEFAULT_SEED
    min_description_chars: int = DEFAULT_MIN_DESCRIPTION_CHARS
    limit: int | None = None
    deduplicate: bool = True
    sample_size: int | None = None

    def validate(self) -> None:
        """Reject configurations that would overwrite an input or empty the output."""
        resolved = {
            "db_path": self.db_path.resolve(),
            "output_path": self.output_path.resolve(),
            "report_path": self.report_path.resolve(),
        }
        if len(set(resolved.values())) != len(resolved):
            raise ValueError("db_path, output_path, and report_path must be different")
        if not self.seed.strip():
            raise ValueError("seed must not be blank")
        if self.min_description_chars < 0:
            raise ValueError("min_description_chars must be non-negative")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be at least 1")
        if self.sample_size is not None and self.sample_size < 1:
            raise ValueError("sample_size must be at least 1")
        if self.limit is not None and self.sample_size is not None:
            # --limit truncates in corpus order and --stratified-sample spreads
            # across families; combining them silently makes the second operate
            # on an already-biased subset.
            raise ValueError("limit and sample_size cannot be combined")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_description(chunk_text: str) -> str | None:
    """Return the normalized description, or None when the chunk carries none."""
    _prefix, marker, remainder = chunk_text.partition(_DESCRIPTION_MARKER)
    if not marker:
        return None
    description = " ".join(remainder.split()).strip()
    return description or None


def build_embedding_text(
    table_name: str,
    column_name: str,
    column_type: str | None,
    description: str,
) -> str:
    """Render the passage, dropping fields that are constant across the corpus.

    `Deprecated?`, `Discontinued?`, `Preserved?`, `Character Replacement?`, and
    `EHI Status` are near-identical on all 356,056 column chunks. Keeping them
    would place most of the corpus in one dense region and give a query generator
    boilerplate to paraphrase instead of meaning.
    """
    lines = [f"Table: {table_name}", f"Column: {column_name}"]
    if column_type:
        lines.append(f"Type: {column_type}")
    lines.append(f"Description: {description}")
    return "\n".join(lines)


def _load_column_rows(conn: sqlite3.Connection) -> tuple[str, list[dict[str, str]]]:
    """Read every column chunk joined to its document, in a deterministic order."""
    index_version = get_index_version(conn)
    rows = [
        {
            "chunk_id": str(row["chunk_id"]),
            "source_path": str(row["source_path"]),
            "source_hash": str(row["source_hash"]),
            "text": str(row["text"]),
        }
        for row in conn.execute(
            "SELECT c.chunk_id AS chunk_id, d.source_path AS source_path, "
            "d.source_hash AS source_hash, c.text AS text "
            "FROM chunks AS c JOIN docs AS d ON d.doc_id = c.doc_id "
            "WHERE c.category = 'column_info' "
            "ORDER BY d.source_path COLLATE BINARY, c.chunk_index, c.chunk_id"
        )
    ]
    return index_version, rows


def _to_split_chunk(
    chunk_id: str,
    source_path: str,
    source_hash: str,
    table_name: str,
    column_name: str,
    text: str,
    split_config: SplitConfig,
) -> SplitChunkRecord:
    """Build a generation record, preserving the index's own chunk identity.

    The chunk_id is the indexer's, not a new one derived here. Column chunks
    carry a readable logical ID (`A0H_MAP__COLUMN_DEFINITION__INTERNAL_ID`)
    assigned by `column_parser` and pinned by `INDEX_CHUNKER_VERSION`, so two
    people indexing the same Epic HTML at the same chunker version get identical
    IDs. Minting a private ID here would produce a second identifier space that
    cannot be joined back to the index and would silently diverge if anything
    about the derivation differed between machines.

    The table-level path mints `table:<document_id>` only because metadata chunks
    have no logical ID to inherit -- their index IDs are content hashes.
    """
    return SplitChunkRecord(
        chunk_id=chunk_id,
        source_file=source_path,
        source_hash=source_hash,
        table_name=table_name,
        column_name=column_name,
        chunk_type="column_definition",
        section_name=column_name,
        text=text,
        text_hash=_sha256_text(text),
        parser_version=EXTRACTOR_VERSION,
        split=assign_source_split(source_path, split_config),
        split_version=SPLIT_VERSION,
    )


def extract_column_chunks(config: ColumnExtractorConfig) -> ColumnExtractionReport:
    """Extract deduplicated column-level generation input from a RAG index."""
    config.validate()
    conn = open_connection(config.db_path)
    try:
        index_version, rows = _load_column_rows(conn)
    finally:
        conn.close()

    split_config = SplitConfig(
        input_path=config.db_path,
        output_path=config.output_path,
        report_path=config.report_path,
        seed=config.seed,
    )

    unparsed = missing = short = 0
    description_counts: dict[str, int] = {}
    seen_chunk_ids: set[str] = set()
    records: list[SplitChunkRecord] = []

    for row in rows:
        header = _HEADER.match(row["text"])
        if header is None:
            unparsed += 1
            continue
        description = extract_description(row["text"])
        if description is None:
            missing += 1
            continue
        if len(description) < config.min_description_chars:
            short += 1
            continue

        # Deduplication runs before the cap, so --limit N yields N chunks with
        # distinct descriptions rather than N raw rows.
        previously_seen = description in description_counts
        description_counts[description] = description_counts.get(description, 0) + 1
        if config.deduplicate and previously_seen:
            continue

        type_match = _TYPE.search(row["text"])
        record = _to_split_chunk(
            chunk_id=row["chunk_id"],
            source_path=row["source_path"],
            source_hash=row["source_hash"],
            table_name=header.group("table").upper(),
            column_name=header.group("column").upper(),
            text=build_embedding_text(
                header.group("table").upper(),
                header.group("column").upper(),
                type_match.group("type").strip() if type_match else None,
                description,
            ),
            split_config=split_config,
        )
        # A table can list the same column twice; the chunk_id must stay unique
        # or the generation stage rejects the whole corpus.
        if record.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(record.chunk_id)
        records.append(record)
        if config.limit is not None and len(records) >= config.limit:
            break

    if config.sample_size is not None:
        records = stratified_sample(
            records, min(config.sample_size, len(records)), seed=config.seed
        )

    split_counts: Counter[str] = Counter(record.split for record in records)
    lines = [
        json.dumps(record.model_dump(), sort_keys=True, ensure_ascii=False) for record in records
    ]
    output_hash = _sha256_text("\n".join(lines) + "\n") if lines else _sha256_text("")

    report = ColumnExtractionReport(
        extractor_version=EXTRACTOR_VERSION,
        split_version=SPLIT_VERSION,
        db_path=str(config.db_path),
        output_path=str(config.output_path),
        report_path=str(config.report_path),
        index_version=index_version,
        seed=config.seed,
        min_description_chars=config.min_description_chars,
        requested_limit=config.limit,
        requested_sample=config.sample_size,
        input_chunk_count=len(rows),
        unparsed_header_count=unparsed,
        missing_description_count=missing,
        short_description_count=short,
        duplicate_description_count=sum(count - 1 for count in description_counts.values()),
        duplicate_group_count=sum(1 for count in description_counts.values() if count > 1),
        output_record_count=len(records),
        source_counts_by_split=dict(split_counts),
        output_hash=output_hash,
    )

    _write_atomic(config.output_path, lines)
    _write_atomic(
        config.report_path,
        [json.dumps(report.model_dump(), indent=2, sort_keys=True, ensure_ascii=False)],
    )
    return report


def stratified_sample(
    records: Sequence[SplitChunkRecord],
    sample_size: int,
    *,
    seed: str = DEFAULT_SEED,
) -> list[SplitChunkRecord]:
    """Take a deterministic sample spread across Chronicles INI table families.

    Sampling uniformly would draw hundreds of columns from the largest table
    families and none from small ones. Bucketing by the table-name prefix keeps
    the Claude overlap sample representative of the schema rather than of its
    biggest tables.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    buckets: dict[str, list[SplitChunkRecord]] = {}
    for record in records:
        buckets.setdefault(record.table_name.split("_", 1)[0], []).append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda record: _sha256_text(f"{seed}\0{record.chunk_id}"))

    selected: list[SplitChunkRecord] = []
    depth = 0
    ordered_families = sorted(buckets)
    while len(selected) < sample_size:
        added = False
        for family in ordered_families:
            bucket = buckets[family]
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) == sample_size:
                    break
        if not added:
            break
        depth += 1
    return selected


def main() -> None:
    """Extract deduplicated column chunks from the command line."""
    parser = argparse.ArgumentParser(
        description="Extract deduplicated column-level generation input from a RAG index."
    )
    parser.add_argument("db_path", type=Path, help="RAG SQLite index")
    parser.add_argument("--output", type=Path, required=True, help="Output SplitChunkRecord JSONL")
    parser.add_argument("--report", type=Path, required=True, help="Output JSON audit report")
    parser.add_argument("--seed", default=DEFAULT_SEED, help="Split assignment seed")
    parser.add_argument(
        "--min-description-chars",
        type=int,
        default=DEFAULT_MIN_DESCRIPTION_CHARS,
        help="Skip columns whose description is shorter than this",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional record cap")
    parser.add_argument(
        "--stratified-sample",
        type=int,
        default=None,
        help=(
            "Emit this many columns spread across Chronicles table families "
            "instead of the whole corpus. Use for a representative Claude sample."
        ),
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        default=False,
        help="Keep every column instead of one per distinct description",
    )
    args = parser.parse_args()
    try:
        report = extract_column_chunks(
            ColumnExtractorConfig(
                db_path=args.db_path,
                output_path=args.output,
                report_path=args.report,
                seed=args.seed,
                min_description_chars=args.min_description_chars,
                limit=args.limit,
                deduplicate=not args.no_dedup,
                sample_size=args.stratified_sample,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Extracted {report.output_record_count:,} column chunks from "
        f"{report.input_chunk_count:,} (train={report.source_counts_by_split.get('train', 0):,}, "
        f"validation={report.source_counts_by_split.get('validation', 0):,}, "
        f"test={report.source_counts_by_split.get('test', 0):,})."
    )
    print(
        f"Collapsed {report.duplicate_description_count:,} duplicate descriptions across "
        f"{report.duplicate_group_count:,} groups; skipped {report.short_description_count:,} "
        f"short and {report.missing_description_count:,} missing."
    )
    print(f"output_hash={report.output_hash}")


if __name__ == "__main__":
    main()
