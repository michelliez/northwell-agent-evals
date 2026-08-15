"""Portable umbrella CLI for non-relationship and relationship eval generation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError
from retrieval.chunk_models import SplitChunkRecord

from clarity_agent_evals.dataset_split import SplitConfig, build_splits
from clarity_agent_evals.genq.claude_query_generator import DEFAULT_CLAUDE_MODEL
from clarity_agent_evals.genq.query_generation import (
    GenerationConfig,
    QueryGenerator,
    generate_queries,
)
from clarity_agent_evals.query_filter import FilterConfig, filter_queries
from clarity_agent_evals.relationship_dataset import (
    DEFAULT_SEED as DEFAULT_RELATIONSHIP_SEED,
)
from clarity_agent_evals.relationship_dataset import (
    RelationshipDatasetConfig,
    build_relationship_dataset,
)
from clarity_agent_evals.relationship_query_generation import (
    RelationshipClaudeGenerator,
    RelationshipGenerationConfig,
    generate_relationship_queries,
)

LOGGER = logging.getLogger(__name__)
SPLIT_CHOICES = ("train", "validation", "test")


@dataclass(frozen=True)
class NonRelationshipEvalConfig:
    chunks_path: Path
    output_dir: Path
    splits: tuple[str, ...] = ("validation", "test")
    split_seed: str = "epic-genq-v1"
    generation_seed: int = 42
    model_name: str = DEFAULT_CLAUDE_MODEL
    queries_per_chunk: int = 2
    batch_size: int = 20
    passages_per_request: int = 10
    max_input_tokens: int = 200
    max_query_tokens: int = 48
    top_p: float = 0.95
    min_passage_chars: int = 100
    limit: int | None = None
    apply_filter: bool = True


@dataclass(frozen=True)
class RelationshipEvalConfig:
    db_path: Path
    output_dir: Path
    splits: tuple[str, ...] = ("validation", "test")
    split_seed: str = DEFAULT_RELATIONSHIP_SEED
    train_size: int = 8_000
    validation_size: int = 1_000
    test_size: int = 1_000
    generation_seed: int = 42
    model_name: str = DEFAULT_CLAUDE_MODEL
    batch_size: int = 20
    relationships_per_request: int = 10
    max_input_tokens: int = 400
    max_query_tokens: int = 64
    top_p: float = 0.95
    max_attempts: int = 3
    limit: int | None = None
    manifest_only: bool = False


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


def _validate_splits(splits: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(splits))
    invalid = sorted(set(normalized) - set(SPLIT_CHOICES))
    if invalid:
        raise ValueError(f"Unknown splits: {', '.join(invalid)}")
    if not normalized:
        raise ValueError("At least one split must be selected")
    return normalized


def _select_split_chunks(
    source_path: Path,
    output_path: Path,
    splits: Sequence[str],
) -> int:
    allowed = set(_validate_splits(splits))
    selected: list[str] = []
    for line_number, line in enumerate(source_path.read_bytes().splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{source_path}:{line_number}: blank JSONL line")
        try:
            record = SplitChunkRecord.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(f"{source_path}:{line_number}: invalid split chunk") from exc
        if record.split in allowed:
            selected.append(json.dumps(record.model_dump(), sort_keys=True, ensure_ascii=False))
    if not selected:
        raise ValueError(f"No chunks belong to selected splits: {', '.join(splits)}")
    _write_atomic(output_path, selected)
    return len(selected)


def run_non_relationship_eval(
    config: NonRelationshipEvalConfig,
    *,
    generator: QueryGenerator | None = None,
) -> dict[str, Any]:
    """Split ordinary chunks, select eval splits, generate, and optionally filter."""
    splits = _validate_splits(config.splits)
    corpus_dir = config.output_dir / "corpus"
    queries_dir = config.output_dir / "queries"
    reviewed_dir = config.output_dir / "reviewed"
    split_path = corpus_dir / "chunks_with_splits.jsonl"
    selected_path = corpus_dir / "selected_chunks.jsonl"
    split_report_path = corpus_dir / "split_report.json"

    split_report = build_splits(
        SplitConfig(
            input_path=config.chunks_path,
            output_path=split_path,
            report_path=split_report_path,
            seed=config.split_seed,
        )
    )
    selected_count = _select_split_chunks(split_path, selected_path, splits)
    generation_report = generate_queries(
        GenerationConfig(
            input_path=selected_path,
            output_path=queries_dir / "generated.jsonl",
            report_path=queries_dir / "generation_report.json",
            provider="claude",
            model_name=config.model_name,
            seed=config.generation_seed,
            batch_size=config.batch_size,
            passages_per_request=config.passages_per_request,
            queries_per_chunk=config.queries_per_chunk,
            max_input_tokens=config.max_input_tokens,
            max_query_tokens=config.max_query_tokens,
            top_p=config.top_p,
            min_passage_chars=config.min_passage_chars,
            limit=config.limit,
        ),
        generator=generator,
    )
    filter_summary: dict[str, Any] | None = None
    if config.apply_filter:
        filter_report = filter_queries(
            FilterConfig(
                queries_path=queries_dir / "generated.jsonl",
                chunks_path=selected_path,
                retained_output_path=reviewed_dir / "retained.jsonl",
                review_output_path=reviewed_dir / "reviews.jsonl",
                report_path=reviewed_dir / "filter_report.json",
            )
        )
        filter_summary = filter_report.model_dump()

    report: dict[str, Any] = {
        "mode": "non-relationship",
        "config": asdict(config),
        "splits": list(splits),
        "source_count": split_report.source_file_count,
        "source_chunk_count": split_report.chunk_count,
        "selected_chunk_count": selected_count,
        "generated_query_count": generation_report.generated_query_count,
        "generation_report": generation_report.model_dump(),
        "filter_report": filter_summary,
    }
    _write_atomic(
        config.output_dir / "run_report.json",
        [json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, default=str)],
    )
    return report


def run_relationship_eval(
    config: RelationshipEvalConfig,
    *,
    generator: RelationshipClaudeGenerator | None = None,
) -> dict[str, Any]:
    """Export relationship manifests and generate selected split queries."""
    splits = _validate_splits(config.splits)
    manifests_dir = config.output_dir / "manifests"
    queries_dir = config.output_dir / "queries"
    dataset_report = build_relationship_dataset(
        RelationshipDatasetConfig(
            db_path=config.db_path,
            output_dir=manifests_dir,
            seed=config.split_seed,
            train_size=config.train_size,
            validation_size=config.validation_size,
            test_size=config.test_size,
        )
    )
    generation_reports: dict[str, Any] = {}
    if not config.manifest_only:
        for split in splits:
            LOGGER.info("Generating relationship queries for split %s", split)
            report = generate_relationship_queries(
                RelationshipGenerationConfig(
                    input_path=manifests_dir / f"{split}_relationships.jsonl",
                    output_path=queries_dir / f"{split}.generated.jsonl",
                    report_path=queries_dir / f"{split}.generation_report.json",
                    model_name=config.model_name,
                    seed=config.generation_seed,
                    batch_size=config.batch_size,
                    relationships_per_request=config.relationships_per_request,
                    max_input_tokens=config.max_input_tokens,
                    max_query_tokens=config.max_query_tokens,
                    top_p=config.top_p,
                    max_attempts=config.max_attempts,
                    limit=config.limit,
                ),
                generator=generator,
            )
            generation_reports[split] = report

    report = {
        "mode": "relationship",
        "config": asdict(config),
        "splits": list(splits),
        "manifest_only": config.manifest_only,
        "dataset_report": dataset_report,
        "generation_reports": generation_reports,
    }
    _write_atomic(
        config.output_dir / "run_report.json",
        [json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, default=str)],
    )
    return report


def _add_shared_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=SPLIT_CHOICES, default=["validation", "test"]
    )
    parser.add_argument("--model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-query-tokens", type=int)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create reproducible Clarity non-relationship or relationship eval sets."
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    ordinary = subparsers.add_parser(
        "non-relationship", help="Split ordinary chunks, generate queries, and filter them."
    )
    ordinary.add_argument("--chunks", type=Path, required=True)
    _add_shared_generation_args(ordinary)
    ordinary.add_argument("--split-seed", default="epic-genq-v1")
    ordinary.add_argument("--queries-per-chunk", type=int, default=2)
    ordinary.add_argument("--items-per-request", type=int, default=10)
    ordinary.add_argument("--min-passage-chars", type=int, default=100)
    ordinary.add_argument("--skip-filter", action="store_true")

    relationship = subparsers.add_parser(
        "relationship", help="Export graph-pair manifests and generate structured queries."
    )
    relationship.add_argument("--db", type=Path, required=True)
    _add_shared_generation_args(relationship)
    relationship.add_argument("--split-seed", default=DEFAULT_RELATIONSHIP_SEED)
    relationship.add_argument("--train-size", type=int, default=8_000)
    relationship.add_argument("--validation-size", type=int, default=1_000)
    relationship.add_argument("--test-size", type=int, default=1_000)
    relationship.add_argument("--items-per-request", type=int, default=10)
    relationship.add_argument("--max-attempts", type=int, default=3)
    relationship.add_argument("--manifest-only", action="store_true")
    return parser


def main() -> None:
    load_dotenv()
    parser = _parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.mode == "non-relationship":
            report = run_non_relationship_eval(
                NonRelationshipEvalConfig(
                    chunks_path=args.chunks,
                    output_dir=args.output_dir,
                    splits=tuple(args.splits),
                    split_seed=args.split_seed,
                    generation_seed=args.generation_seed,
                    model_name=args.model,
                    queries_per_chunk=args.queries_per_chunk,
                    batch_size=args.batch_size,
                    passages_per_request=args.items_per_request,
                    max_input_tokens=args.max_input_tokens or 200,
                    max_query_tokens=args.max_query_tokens or 48,
                    top_p=args.top_p,
                    min_passage_chars=args.min_passage_chars,
                    limit=args.limit,
                    apply_filter=not args.skip_filter,
                )
            )
        else:
            report = run_relationship_eval(
                RelationshipEvalConfig(
                    db_path=args.db,
                    output_dir=args.output_dir,
                    splits=tuple(args.splits),
                    split_seed=args.split_seed,
                    train_size=args.train_size,
                    validation_size=args.validation_size,
                    test_size=args.test_size,
                    generation_seed=args.generation_seed,
                    model_name=args.model,
                    batch_size=args.batch_size,
                    relationships_per_request=args.items_per_request,
                    max_input_tokens=args.max_input_tokens or 400,
                    max_query_tokens=args.max_query_tokens or 64,
                    top_p=args.top_p,
                    max_attempts=args.max_attempts,
                    limit=args.limit,
                    manifest_only=args.manifest_only,
                )
            )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    generated = sum(
        value.get("generated_query_count", 0)
        for value in report.get("generation_reports", {}).values()
    )
    if report["mode"] == "non-relationship":
        generated = report["generated_query_count"]
    print(
        f"Completed {report['mode']} eval generation; generated_queries={generated}; "
        f"report={args.output_dir / 'run_report.json'}"
    )


if __name__ == "__main__":
    main()
