"""Generate raw synthetic queries from leakage-safe Epic documentation chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from pydantic import ValidationError
from retrieval.chunk_models import (
    GeneratedQueryRecord,
    GenerationReport,
    SplitChunkRecord,
)

from clarity_agent_evals.genq.claude_query_generator import (
    DEFAULT_CLAUDE_MODEL,
    ClaudeHaikuQueryGenerator,
)
from clarity_agent_evals.genq.gemma_query_generator import GemmaMLXQueryGenerator
from clarity_agent_evals.genq.qwen_query_generator import (
    DEFAULT_MAX_RETRIES as DEFAULT_QWEN_MAX_RETRIES,
)
from clarity_agent_evals.genq.qwen_query_generator import (
    DEFAULT_QWEN_MODEL,
    QwenQueryGenerator,
)

PROVIDER_CHOICES = ("claude", "gemma", "qwen")

DEFAULT_GEMMA_MODEL = ".local/models/gemma-3-1b-it-4bit"
DEFAULT_GEMMA_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_QWEN_DEVICE = "auto"
QWEN_DEVICE_CHOICES = ("auto", "cuda", "cpu")

LOGGER = logging.getLogger(__name__)
GENERATION_VERSION = "synthetic-query-generation-v3"
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 8
DEFAULT_PASSAGES_PER_REQUEST = 1
DEFAULT_QUERIES_PER_CHUNK = 5
DEFAULT_MAX_INPUT_TOKENS = 300
DEFAULT_MAX_QUERY_TOKENS = 64
DEFAULT_TOP_P = 0.95
DEFAULT_MIN_PASSAGE_CHARS = 100


class QueryGenerator(Protocol):
    """Small interface that keeps model downloads out of deterministic tests."""

    model_name: str
    device_name: str

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
        """Return exactly `queries_per_passage` raw queries for each passage."""
        ...


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for raw synthetic-query generation."""

    input_path: Path
    output_path: Path
    report_path: Path
    provider: str = "claude"
    model_name: str = DEFAULT_CLAUDE_MODEL
    gemma_model_name: str = DEFAULT_GEMMA_MODEL
    gemma_base_url: str = DEFAULT_GEMMA_BASE_URL
    qwen_model_name: str = DEFAULT_QWEN_MODEL
    qwen_device: str = DEFAULT_QWEN_DEVICE
    qwen_load_in_4bit: bool = True
    qwen_max_retries: int = DEFAULT_QWEN_MAX_RETRIES
    seed: int = DEFAULT_SEED
    batch_size: int = DEFAULT_BATCH_SIZE
    passages_per_request: int = DEFAULT_PASSAGES_PER_REQUEST
    queries_per_chunk: int = DEFAULT_QUERIES_PER_CHUNK
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    max_query_tokens: int = DEFAULT_MAX_QUERY_TOKENS
    top_p: float = DEFAULT_TOP_P
    min_passage_chars: int = DEFAULT_MIN_PASSAGE_CHARS
    limit: int | None = None

    def validate(self) -> None:
        """Reject configurations that could corrupt or silently empty output."""
        if self.input_path in {self.output_path, self.report_path}:
            raise ValueError("input_path, output_path, and report_path must be different")
        if self.output_path == self.report_path:
            raise ValueError("output_path and report_path must be different")
        if not self.model_name.strip():
            raise ValueError("model_name must not be blank")
        if self.provider not in PROVIDER_CHOICES:
            raise ValueError(f"provider must be one of: {', '.join(PROVIDER_CHOICES)}")
        if not self.gemma_model_name.strip():
            raise ValueError("gemma_model_name must not be blank")
        if not self.gemma_base_url.strip():
            raise ValueError("gemma_base_url must not be blank")
        if not self.qwen_model_name.strip():
            raise ValueError("qwen_model_name must not be blank")
        if self.qwen_device not in QWEN_DEVICE_CHOICES:
            raise ValueError(f"qwen_device must be one of: {', '.join(QWEN_DEVICE_CHOICES)}")
        if self.qwen_max_retries < 0:
            raise ValueError("qwen_max_retries must be non-negative")
        for name, value in (
            ("batch_size", self.batch_size),
            ("passages_per_request", self.passages_per_request),
            ("queries_per_chunk", self.queries_per_chunk),
            ("max_input_tokens", self.max_input_tokens),
            ("max_query_tokens", self.max_query_tokens),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if self.min_passage_chars < 0:
            raise ValueError("min_passage_chars must be non-negative")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be at least 1 or None")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _normalize_query(value: str) -> str:
    return " ".join(value.replace("\t", " ").split()).strip()


def load_split_chunks(path: Path) -> tuple[list[SplitChunkRecord], str]:
    """Load Stage 2 JSONL and provide line-specific validation failures."""
    try:
        input_bytes = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"Input JSONL does not exist: {path}") from exc
    if not input_bytes.strip():
        raise ValueError(f"Input JSONL is empty: {path}")

    records: list[SplitChunkRecord] = []
    for line_number, raw_line in enumerate(input_bytes.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        try:
            records.append(SplitChunkRecord.model_validate_json(raw_line))
        except ValidationError as exc:
            raise ValueError(f"{path}:{line_number}: invalid split chunk record: {exc}") from exc
    duplicate_ids = [
        chunk_id
        for chunk_id, count in Counter(record.chunk_id for record in records).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"Duplicate chunk IDs found: {', '.join(sorted(duplicate_ids)[:5])}")
    return records, _sha256_bytes(input_bytes)


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


def generate_queries(
    config: GenerationConfig,
    *,
    generator: QueryGenerator | None = None,
) -> GenerationReport:
    """Generate raw queries while preserving chunk identity and split provenance."""
    config.validate()
    chunks, input_hash = load_split_chunks(config.input_path)
    eligible = [chunk for chunk in chunks if len(chunk.text) >= config.min_passage_chars]
    selected = eligible[: config.limit] if config.limit is not None else eligible
    if not selected:
        raise ValueError("No chunks satisfy the configured passage length and limit")

    if generator is not None:
        active_generator = generator
    elif config.provider == "gemma":
        active_generator = GemmaMLXQueryGenerator(
            config.gemma_model_name,
            base_url=config.gemma_base_url,
        )
    elif config.provider == "qwen":
        active_generator = QwenQueryGenerator(
            config.qwen_model_name,
            device=config.qwen_device,
            load_in_4bit=config.qwen_load_in_4bit,
            # Qwen batches on the GPU itself, so the pipeline batch size only
            # sets log granularity. One knob drives both rather than two that
            # can disagree about how much VRAM a run needs.
            batch_size=config.batch_size,
            max_retries=config.qwen_max_retries,
        )
    else:
        active_generator = ClaudeHaikuQueryGenerator(
            config.model_name, passages_per_request=config.passages_per_request
        )
    records: list[GeneratedQueryRecord] = []
    failed_generation_chunk_count = 0
    for batch_number, start in enumerate(range(0, len(selected), config.batch_size)):
        batch = selected[start : start + config.batch_size]
        LOGGER.info(
            "Generating batch %d (%d/%d chunks)",
            batch_number + 1,
            min(start + len(batch), len(selected)),
            len(selected),
        )
        generated = active_generator.generate(
            [chunk.text for chunk in batch],
            queries_per_passage=config.queries_per_chunk,
            max_input_tokens=config.max_input_tokens,
            max_query_tokens=config.max_query_tokens,
            top_p=config.top_p,
            seed=config.seed + batch_number,
        )
        if len(generated) != len(batch):
            raise RuntimeError(
                f"Generator returned results for {len(generated)} passages; expected {len(batch)}"
            )
        for chunk, raw_queries in zip(batch, generated, strict=True):
            if not raw_queries:
                failed_generation_chunk_count += 1
                LOGGER.warning(
                    "Skipping chunk %s because the generator returned no queries",
                    chunk.chunk_id,
                )
                continue
            if len(raw_queries) != config.queries_per_chunk:
                raise RuntimeError(
                    f"Generator returned {len(raw_queries)} queries for {chunk.chunk_id}; "
                    f"expected {config.queries_per_chunk}"
                )
            for query_index, raw_query in enumerate(raw_queries):
                query = _normalize_query(raw_query)
                if not query:
                    raise RuntimeError(f"Generator returned an empty query for {chunk.chunk_id}")
                query_id = "q_" + _sha256_text(f"{chunk.chunk_id}\0{query_index}")[:16]
                records.append(
                    GeneratedQueryRecord(
                        query_id=query_id,
                        query=query,
                        relevant_chunk_id=chunk.chunk_id,
                        relevant_text_hash=chunk.text_hash,
                        source_file=chunk.source_file,
                        chunk_type=chunk.chunk_type,
                        split=chunk.split,
                        split_version=chunk.split_version,
                        generator_model=active_generator.model_name,
                        generation_seed=config.seed,
                        query_index=query_index,
                    )
                )

    query_duplicates = sum(
        count - 1
        for count in Counter(record.query.casefold() for record in records).values()
        if count > 1
    )
    lines = [
        json.dumps(record.model_dump(), sort_keys=True, ensure_ascii=False) for record in records
    ]
    output_hash = _sha256_text("\n".join(lines) + "\n")
    split_counts = Counter(record.split for record in records)
    type_counts = Counter(record.chunk_type for record in records)
    report = GenerationReport(
        generation_version=GENERATION_VERSION,
        input_path=str(config.input_path),
        output_path=str(config.output_path),
        generator_model=active_generator.model_name,
        device=active_generator.device_name,
        seed=config.seed,
        batch_size=config.batch_size,
        passages_per_request=config.passages_per_request,
        queries_per_chunk=config.queries_per_chunk,
        max_input_tokens=config.max_input_tokens,
        max_query_tokens=config.max_query_tokens,
        top_p=config.top_p,
        min_passage_chars=config.min_passage_chars,
        requested_limit=config.limit,
        input_chunk_count=len(chunks),
        eligible_chunk_count=len(eligible),
        selected_chunk_count=len(selected),
        skipped_short_chunk_count=len(chunks) - len(eligible),
        failed_generation_chunk_count=failed_generation_chunk_count,
        generated_query_count=len(records),
        duplicate_query_count=query_duplicates,
        query_counts_by_split={
            split: split_counts.get(split, 0) for split in ("train", "validation", "test")
        },
        query_counts_by_chunk_type=dict(sorted(type_counts.items())),
        input_corpus_hash=input_hash,
        output_query_hash=output_hash,
    )
    _write_atomic(config.output_path, lines)
    _write_atomic(
        config.report_path,
        [json.dumps(report.model_dump(), indent=2, sort_keys=True, ensure_ascii=False)],
    )
    return report


def main() -> None:
    """Run raw synthetic-query generation from the command line."""
    # Load local defaults before building argparse options so GEMMA_BASE_URL in
    # .env can supply the CLI default. Explicit shell variables still win.
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Generate raw synthetic queries from Stage 2 Epic chunks."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default="claude",
        help="Query-generation provider (default: claude)",
    )
    parser.add_argument("--model", default=DEFAULT_CLAUDE_MODEL, help="Claude model ID")
    parser.add_argument(
        "--gemma-model",
        default=DEFAULT_GEMMA_MODEL,
        help="Gemma model name recorded in generated-query provenance",
    )
    parser.add_argument(
        "--gemma-base-url",
        default=os.getenv("GEMMA_BASE_URL", DEFAULT_GEMMA_BASE_URL),
        help="Base URL of the local MLX-LM server",
    )
    parser.add_argument(
        "--qwen-model", default=DEFAULT_QWEN_MODEL, help="Hugging Face model ID for Qwen"
    )
    parser.add_argument("--qwen-device", default=DEFAULT_QWEN_DEVICE, choices=QWEN_DEVICE_CHOICES)
    parser.add_argument(
        "--qwen-no-4bit",
        action="store_true",
        default=False,
        help="Load Qwen in bf16 instead of 4-bit NF4 (needs roughly 8 GB for a 4B model)",
    )
    parser.add_argument(
        "--qwen-max-retries",
        type=int,
        default=DEFAULT_QWEN_MAX_RETRIES,
        help="Re-sample attempts for a passage whose reply is not parseable JSON",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--passages-per-request",
        type=int,
        default=DEFAULT_PASSAGES_PER_REQUEST,
        help=(
            "Passages to place in a single API request. 1 reproduces earlier runs; "
            "10 cuts input tokens per passage from ~748 to ~135."
        ),
    )
    parser.add_argument("--queries-per-chunk", type=int, default=DEFAULT_QUERIES_PER_CHUNK)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--max-query-tokens", type=int, default=DEFAULT_MAX_QUERY_TOKENS)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--min-passage-chars", type=int, default=DEFAULT_MIN_PASSAGE_CHARS)
    parser.add_argument("--limit", type=int)
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
        report = generate_queries(
            GenerationConfig(
                input_path=args.input_path,
                output_path=args.output,
                report_path=args.report,
                provider=args.provider,
                model_name=args.model,
                gemma_model_name=args.gemma_model,
                gemma_base_url=args.gemma_base_url,
                qwen_model_name=args.qwen_model,
                qwen_device=args.qwen_device,
                qwen_load_in_4bit=not args.qwen_no_4bit,
                qwen_max_retries=args.qwen_max_retries,
                seed=args.seed,
                batch_size=args.batch_size,
                passages_per_request=args.passages_per_request,
                queries_per_chunk=args.queries_per_chunk,
                max_input_tokens=args.max_input_tokens,
                max_query_tokens=args.max_query_tokens,
                top_p=args.top_p,
                min_passage_chars=args.min_passage_chars,
                limit=args.limit,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Generated {report.generated_query_count} raw queries from "
        f"{report.selected_chunk_count - report.failed_generation_chunk_count} chunks "
        f"on {report.device}; skipped_failed={report.failed_generation_chunk_count}; "
        f"output_hash={report.output_query_hash}"
    )


if __name__ == "__main__":
    main()
