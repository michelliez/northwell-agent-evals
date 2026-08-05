"""Filter raw GenQ output into retained candidates and an audit ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import ValidationError

from retrieval.chunk_models import (
    FilterDecision,
    FilteredQueryRecord,
    FilterReport,
    GeneratedQueryRecord,
    QueryReviewRecord,
    SplitChunkRecord,
)

LOGGER = logging.getLogger(__name__)
FILTER_VERSION = "deterministic-query-quality-v1"
DEFAULT_MIN_QUERY_WORDS = 3
DEFAULT_MAX_QUERY_WORDS = 32
DEFAULT_MAX_QUERY_CHARS = 256
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.88
DEFAULT_PASSAGE_COPY_THRESHOLD = 0.86
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "can",
        "column",
        "data",
        "define",
        "description",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "information",
        "of",
        "on",
        "or",
        "the",
        "this",
        "table",
        "to",
        "what",
        "where",
        "which",
        "with",
    }
)
MODEL_ARTIFACTS = ("<pad>", "</s>", "<unk>", "[pad]", "[sep]")


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for deterministic query-quality decisions."""

    queries_path: Path
    chunks_path: Path
    retained_output_path: Path
    review_output_path: Path
    report_path: Path
    min_query_words: int = DEFAULT_MIN_QUERY_WORDS
    max_query_words: int = DEFAULT_MAX_QUERY_WORDS
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD
    passage_copy_threshold: float = DEFAULT_PASSAGE_COPY_THRESHOLD

    def validate(self) -> None:
        """Reject path collisions and invalid quality thresholds."""
        paths = {
            self.queries_path,
            self.chunks_path,
            self.retained_output_path,
            self.review_output_path,
            self.report_path,
        }
        if len(paths) != 5:
            raise ValueError("all input and output paths must be different")
        if self.min_query_words < 1:
            raise ValueError("min_query_words must be at least 1")
        if self.max_query_words < self.min_query_words:
            raise ValueError("max_query_words must be at least min_query_words")
        if self.max_query_chars < 1:
            raise ValueError("max_query_chars must be at least 1")
        for name, value in (
            ("near_duplicate_threshold", self.near_duplicate_threshold),
            ("passage_copy_threshold", self.passage_copy_threshold),
        ):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be greater than 0 and at most 1")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", value.casefold()))


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", value.casefold())


def _meaningful_tokens(value: str) -> set[str]:
    return {
        token
        for token in _tokens(value)
        if token not in STOP_WORDS and (len(token) >= 3 or any(char.isdigit() for char in token))
    }


def _load_jsonl[T](path: Path, model: type[T], label: str) -> tuple[list[T], str]:
    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} JSONL does not exist: {path}") from exc
    if not raw_bytes.strip():
        raise ValueError(f"{label} JSONL is empty: {path}")
    records: list[T] = []
    for line_number, line in enumerate(raw_bytes.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        try:
            records.append(model.model_validate_json(line))  # type: ignore[attr-defined]
        except ValidationError as exc:
            raise ValueError(f"{path}:{line_number}: invalid {label} record: {exc}") from exc
    return records, _sha256_bytes(raw_bytes)


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


def _validate_provenance(
    queries: list[GeneratedQueryRecord],
    chunks: list[SplitChunkRecord],
) -> dict[str, SplitChunkRecord]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunk_by_id) != len(chunks):
        raise ValueError("Duplicate chunk IDs found in source chunks")
    query_ids = Counter(query.query_id for query in queries)
    duplicate_queries = sorted(query_id for query_id, count in query_ids.items() if count > 1)
    if duplicate_queries:
        raise ValueError(f"Duplicate query IDs found: {', '.join(duplicate_queries[:5])}")

    for query in queries:
        chunk = chunk_by_id.get(query.relevant_chunk_id)
        if chunk is None:
            raise ValueError(
                f"Query {query.query_id} references missing chunk {query.relevant_chunk_id}"
            )
        expected = (
            chunk.text_hash,
            chunk.source_file,
            chunk.chunk_type,
            chunk.split,
            chunk.split_version,
        )
        actual = (
            query.relevant_text_hash,
            query.source_file,
            query.chunk_type,
            query.split,
            query.split_version,
        )
        if actual != expected:
            raise ValueError(f"Query {query.query_id} has inconsistent chunk provenance")
    return chunk_by_id


def _ambiguous_exact_queries(
    queries: list[GeneratedQueryRecord],
) -> set[str]:
    targets_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for query in queries:
        targets_by_key[(query.split, _normalized_text(query.query))].add(query.relevant_chunk_id)
    return {
        query.query_id
        for query in queries
        if len(targets_by_key[(query.split, _normalized_text(query.query))]) > 1
    }


def filter_queries(config: FilterConfig) -> FilterReport:
    """Apply conservative deterministic gates and write both outputs atomically."""
    config.validate()
    queries, raw_hash = _load_jsonl(config.queries_path, GeneratedQueryRecord, "generated query")
    chunks, chunks_hash = _load_jsonl(config.chunks_path, SplitChunkRecord, "split chunk")
    chunk_by_id = _validate_provenance(queries, chunks)
    ambiguous_query_ids = _ambiguous_exact_queries(queries)

    reviews: list[QueryReviewRecord] = []
    retained: list[FilteredQueryRecord] = []
    prior_by_chunk: dict[tuple[str, str], list[GeneratedQueryRecord]] = defaultdict(list)
    exact_seen_by_chunk: dict[tuple[str, str, str], str] = {}

    for query in queries:
        chunk = chunk_by_id[query.relevant_chunk_id]
        normalized = _normalized_text(query.query)
        query_tokens = _tokens(query.query)
        meaningful_query = _meaningful_tokens(query.query)
        overlap = sorted(meaningful_query & _meaningful_tokens(chunk.text))
        reject_reasons: list[str] = []
        review_reasons: list[str] = []
        near_duplicate_of: str | None = None

        if len(query_tokens) < config.min_query_words:
            reject_reasons.append("too_few_words")
        if len(query_tokens) > config.max_query_words:
            reject_reasons.append("too_many_words")
        if len(query.query) > config.max_query_chars:
            reject_reasons.append("too_many_characters")
        filename = Path(query.source_file).name.casefold()
        if filename in query.query.casefold() or ".html" in query.query.casefold():
            reject_reasons.append("mentions_source_filename")
        if any(artifact in query.query.casefold() for artifact in MODEL_ARTIFACTS):
            reject_reasons.append("contains_model_artifact")

        normalized_passage = _normalized_text(chunk.text)
        copied_substring = len(query_tokens) >= 8 and normalized in normalized_passage
        copy_ratio = SequenceMatcher(None, normalized, normalized_passage).ratio()
        if copied_substring or copy_ratio >= config.passage_copy_threshold:
            reject_reasons.append("copies_source_passage")

        exact_key = (query.split, query.relevant_chunk_id, normalized)
        if exact_key in exact_seen_by_chunk:
            reject_reasons.append("exact_duplicate_same_chunk")
            near_duplicate_of = exact_seen_by_chunk[exact_key]
        else:
            exact_seen_by_chunk[exact_key] = query.query_id

        if not reject_reasons:
            for previous in prior_by_chunk[(query.split, query.relevant_chunk_id)]:
                similarity = SequenceMatcher(
                    None, normalized, _normalized_text(previous.query)
                ).ratio()
                if similarity >= config.near_duplicate_threshold:
                    reject_reasons.append("near_duplicate_same_chunk")
                    near_duplicate_of = previous.query_id
                    break

        if query.query_id in ambiguous_query_ids:
            review_reasons.append("same_query_multiple_positive_chunks")
        if not overlap:
            review_reasons.append("no_meaningful_lexical_support")

        if reject_reasons:
            decision: FilterDecision = "reject"
            reasons = reject_reasons + review_reasons
        elif review_reasons:
            decision = "review"
            reasons = review_reasons
        else:
            decision = "retain"
            reasons = []
            retained.append(
                FilteredQueryRecord(
                    **query.model_dump(),
                    filter_version=FILTER_VERSION,
                    meaningful_overlap_tokens=overlap,
                )
            )
        reviews.append(
            QueryReviewRecord(
                **query.model_dump(),
                filter_version=FILTER_VERSION,
                decision=decision,
                reason_codes=reasons,
                normalized_query=normalized,
                meaningful_overlap_tokens=overlap,
                near_duplicate_of=near_duplicate_of,
            )
        )
        prior_by_chunk[(query.split, query.relevant_chunk_id)].append(query)

    retained_lines = [
        json.dumps(record.model_dump(), sort_keys=True, ensure_ascii=False) for record in retained
    ]
    review_lines = [
        json.dumps(record.model_dump(), sort_keys=True, ensure_ascii=False) for record in reviews
    ]
    retained_hash = _sha256_text("\n".join(retained_lines) + "\n")
    review_hash = _sha256_text("\n".join(review_lines) + "\n")
    decision_counts = Counter(review.decision for review in reviews)
    reason_counts = Counter(reason for review in reviews for reason in review.reason_codes)
    split_counts = Counter(record.split for record in retained)
    report = FilterReport(
        filter_version=FILTER_VERSION,
        queries_path=str(config.queries_path),
        chunks_path=str(config.chunks_path),
        retained_output_path=str(config.retained_output_path),
        review_output_path=str(config.review_output_path),
        min_query_words=config.min_query_words,
        max_query_words=config.max_query_words,
        max_query_chars=config.max_query_chars,
        near_duplicate_threshold=config.near_duplicate_threshold,
        passage_copy_threshold=config.passage_copy_threshold,
        input_query_count=len(queries),
        retained_query_count=len(retained),
        rejected_query_count=decision_counts.get("reject", 0),
        manual_review_query_count=decision_counts.get("review", 0),
        decision_counts={
            decision: decision_counts.get(decision, 0)
            for decision in ("retain", "reject", "review")
        },
        reason_counts=dict(sorted(reason_counts.items())),
        retained_counts_by_split={
            split: split_counts.get(split, 0) for split in ("train", "validation", "test")
        },
        raw_queries_hash=raw_hash,
        source_chunks_hash=chunks_hash,
        retained_queries_hash=retained_hash,
        review_ledger_hash=review_hash,
    )
    _write_atomic(config.retained_output_path, retained_lines)
    _write_atomic(config.review_output_path, review_lines)
    _write_atomic(
        config.report_path,
        [json.dumps(report.model_dump(), indent=2, sort_keys=True, ensure_ascii=False)],
    )
    LOGGER.info(
        "Filtered %d queries: %d retained, %d rejected, %d review",
        len(queries),
        len(retained),
        report.rejected_query_count,
        report.manual_review_query_count,
    )
    return report


def main() -> None:
    """Run Stage 4 deterministic query filtering."""
    parser = argparse.ArgumentParser(
        description="Filter raw generated queries and write an auditable review ledger."
    )
    parser.add_argument("queries_path", type=Path)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--retained-output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-query-words", type=int, default=DEFAULT_MIN_QUERY_WORDS)
    parser.add_argument("--max-query-words", type=int, default=DEFAULT_MAX_QUERY_WORDS)
    parser.add_argument("--max-query-chars", type=int, default=DEFAULT_MAX_QUERY_CHARS)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    )
    parser.add_argument(
        "--passage-copy-threshold",
        type=float,
        default=DEFAULT_PASSAGE_COPY_THRESHOLD,
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        report = filter_queries(
            FilterConfig(
                queries_path=args.queries_path,
                chunks_path=args.chunks,
                retained_output_path=args.retained_output,
                review_output_path=args.review_output,
                report_path=args.report,
                min_query_words=args.min_query_words,
                max_query_words=args.max_query_words,
                max_query_chars=args.max_query_chars,
                near_duplicate_threshold=args.near_duplicate_threshold,
                passage_copy_threshold=args.passage_copy_threshold,
            )
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Reviewed {report.input_query_count} raw queries: "
        f"{report.retained_query_count} retained, "
        f"{report.rejected_query_count} rejected, "
        f"{report.manual_review_query_count} require manual review"
    )


if __name__ == "__main__":
    main()
