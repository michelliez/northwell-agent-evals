"""Assign complete Epic source files to deterministic ML dataset splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from retrieval.chunk_models import (
    ChunkRecord,
    SplitChunkRecord,
    SplitName,
    SplitReport,
)

LOGGER = logging.getLogger(__name__)
SPLIT_VERSION = "source-hash-split-v1"
DEFAULT_SEED = "epic-genq-v1"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VALIDATION_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for deterministic, source-grouped dataset splitting."""

    input_path: Path
    output_path: Path
    report_path: Path
    seed: str = DEFAULT_SEED
    train_ratio: float = DEFAULT_TRAIN_RATIO
    validation_ratio: float = DEFAULT_VALIDATION_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO

    def validate(self) -> None:
        """Reject settings that could produce missing or ambiguous assignments."""
        if self.input_path in {self.output_path, self.report_path}:
            raise ValueError("input_path, output_path, and report_path must be different")
        if self.output_path == self.report_path:
            raise ValueError("output_path and report_path must be different")
        if not self.seed.strip():
            raise ValueError("seed must not be blank")
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if not all(math.isfinite(ratio) and ratio > 0 for ratio in ratios):
            raise ValueError("all split ratios must be finite and greater than zero")
        if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("split ratios must sum to 1.0")

    def ratios(self) -> dict[SplitName, float]:
        """Return ratios with stable names and ordering."""
        return {
            "train": self.train_ratio,
            "validation": self.validation_ratio,
            "test": self.test_ratio,
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def assign_source_split(source_file: str, config: SplitConfig) -> SplitName:
    """Map one source group to a stable split using a seeded SHA-256 score.

    Hash thresholds keep existing assignments stable when new source files are
    added. Exact 80/10/10 counts are not guaranteed for a small sample, but the
    proportions converge as the corpus grows.
    """
    digest = hashlib.sha256(f"{config.seed}\0{source_file}".encode()).digest()
    score = int.from_bytes(digest[:8], "big") / 2**64
    if score < config.train_ratio:
        return "train"
    if score < config.train_ratio + config.validation_ratio:
        return "validation"
    return "test"


def load_chunks(path: Path) -> tuple[list[ChunkRecord], str]:
    """Load and validate a Stage 1 JSONL artifact with line-specific errors."""
    try:
        input_bytes = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"Input JSONL does not exist: {path}") from exc
    if not input_bytes.strip():
        raise ValueError(f"Input JSONL is empty: {path}")

    records: list[ChunkRecord] = []
    for line_number, raw_line in enumerate(input_bytes.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        try:
            value = json.loads(raw_line)
            records.append(ChunkRecord.model_validate(value))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid chunk record: {exc}") from exc
    return records, _sha256_bytes(input_bytes)


def _validate_input_records(records: list[ChunkRecord]) -> None:
    chunk_ids = Counter(record.chunk_id for record in records)
    duplicates = sorted(chunk_id for chunk_id, count in chunk_ids.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate chunk IDs found: {', '.join(duplicates[:5])}")

    source_metadata: dict[str, tuple[str, str]] = {}
    for record in records:
        identity = (record.source_hash, record.table_name)
        previous = source_metadata.setdefault(record.source_file, identity)
        if previous != identity:
            raise ValueError(
                f"Inconsistent source metadata for {record.source_file}: "
                "source_hash and table_name must agree"
            )


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


def build_splits(config: SplitConfig) -> SplitReport:
    """Create leakage-safe split records and an auditable assignment report."""
    config.validate()
    chunks, input_hash = load_chunks(config.input_path)
    _validate_input_records(chunks)

    source_files = sorted({chunk.source_file for chunk in chunks})
    source_assignments: dict[str, SplitName] = {
        source_file: assign_source_split(source_file, config) for source_file in source_files
    }
    split_records = [
        SplitChunkRecord(
            **chunk.model_dump(),
            split=source_assignments[chunk.source_file],
            split_version=SPLIT_VERSION,
        )
        for chunk in chunks
    ]

    sources_seen_by_split: dict[str, set[SplitName]] = defaultdict(set)
    for record in split_records:
        sources_seen_by_split[record.source_file].add(record.split)
    leaking_sources = [
        source for source, splits in sources_seen_by_split.items() if len(splits) > 1
    ]
    if leaking_sources:
        raise RuntimeError(f"Source-group leakage detected: {', '.join(leaking_sources[:5])}")

    jsonl_lines = [
        json.dumps(record.model_dump(), sort_keys=True, ensure_ascii=False)
        for record in split_records
    ]
    output_hash = _sha256_text("\n".join(jsonl_lines) + "\n")
    manifest_lines = [
        f"{source_file}\t{source_assignments[source_file]}" for source_file in source_files
    ]
    manifest_hash = _sha256_text("\n".join(manifest_lines) + "\n")

    source_counts = Counter(source_assignments.values())
    chunk_counts = Counter(record.split for record in split_records)
    type_counts: dict[SplitName, Counter[str]] = {
        split: Counter(record.chunk_type for record in split_records if record.split == split)
        for split in SPLIT_NAMES
    }
    report = SplitReport(
        split_version=SPLIT_VERSION,
        input_path=str(config.input_path),
        output_path=str(config.output_path),
        seed=config.seed,
        ratios=config.ratios(),
        source_file_count=len(source_files),
        chunk_count=len(split_records),
        source_counts_by_split={split: source_counts.get(split, 0) for split in SPLIT_NAMES},
        chunk_counts_by_split={split: chunk_counts.get(split, 0) for split in SPLIT_NAMES},
        chunk_type_counts_by_split={
            split: dict(sorted(type_counts[split].items())) for split in SPLIT_NAMES
        },
        leakage_source_count=len(leaking_sources),
        duplicate_chunk_id_count=0,
        input_corpus_hash=input_hash,
        output_corpus_hash=output_hash,
        split_manifest_hash=manifest_hash,
    )

    _write_atomic(config.output_path, jsonl_lines)
    _write_atomic(
        config.report_path,
        [json.dumps(report.model_dump(), indent=2, sort_keys=True, ensure_ascii=False)],
    )
    LOGGER.info(
        "Assigned %d source files and %d chunks: %s",
        len(source_files),
        len(split_records),
        dict(report.source_counts_by_split),
    )
    return report


def main() -> None:
    """Run the leakage-safe Stage 2 splitter from the command line."""
    parser = argparse.ArgumentParser(
        description="Assign Stage 1 chunks to deterministic source-group dataset splits."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
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
        report = build_splits(
            SplitConfig(
                input_path=args.input_path,
                output_path=args.output,
                report_path=args.report,
                seed=args.seed,
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
                test_ratio=args.test_ratio,
            )
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Assigned {report.source_file_count} files / {report.chunk_count} chunks "
        f"to splits {dict(report.source_counts_by_split)}; "
        f"manifest_hash={report.split_manifest_hash}"
    )


if __name__ == "__main__":
    main()
