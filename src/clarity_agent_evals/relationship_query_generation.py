"""Generate structured relationship-retrieval questions with Claude Haiku."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from clarity_agent_evals.genq.claude_query_generator import (
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_CLAUDE_MODEL,
    MessagesClient,
    _headers_from_env,
    _HTTPMessagesClient,
    _timeout_from_env,
)
from clarity_agent_evals.relationship_dataset import SPLIT_VERSION

LOGGER = logging.getLogger(__name__)
GENERATION_VERSION = "relationship-query-generation-v2"
IDENTIFIER_AWARE_TEMPLATE_VERSION = "relationship-aware-template-v1"
SYSTEM_PROMPT = """\
You create realistic evaluation questions for an Epic Clarity relationship-retrieval system.
Each supplied group contains two documented tables, their bounded business descriptions, and
explicit foreign-key column edges. Generate one identifier-free question only; never answer it
and never invent facts. The question must describe a supported business need without using any
supplied table or column identifier, and must require connecting the two documented concepts.
Do not request patient-level data. Identifier-aware questions are built deterministically outside
the model and are not part of your task.
"""


class InvalidRelationshipResponse(RuntimeError):
    """Claude replied, but its structured relationship questions were invalid."""


@dataclass(frozen=True)
class RelationshipGenerationConfig:
    input_path: Path
    output_path: Path
    report_path: Path
    model_name: str = DEFAULT_CLAUDE_MODEL
    seed: int = 42
    batch_size: int = 20
    relationships_per_request: int = 10
    max_input_tokens: int = 400
    max_query_tokens: int = 64
    top_p: float = 0.95
    max_attempts: int = 3
    limit: int | None = None

    def validate(self) -> None:
        if self.input_path in {self.output_path, self.report_path}:
            raise ValueError("input_path, output_path, and report_path must be different")
        if self.output_path == self.report_path:
            raise ValueError("output_path and report_path must be different")
        if not self.model_name.strip():
            raise ValueError("model_name must not be blank")
        for name, value in (
            ("batch_size", self.batch_size),
            ("relationships_per_request", self.relationships_per_request),
            ("max_input_tokens", self.max_input_tokens),
            ("max_query_tokens", self.max_query_tokens),
            ("max_attempts", self.max_attempts),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be at least 1 or None")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _normalize_query(value: str) -> str:
    return " ".join(value.replace("\t", " ").split()).strip()


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


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _validate_group(value: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number}: relationship group must be an object")
    group_id = _required_string(value.get("relationship_group_id"), "relationship_group_id")
    split = _required_string(value.get("split"), "split")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"line {line_number}: invalid split {split!r}")
    if value.get("split_version") != SPLIT_VERSION:
        raise ValueError(f"line {line_number}: unsupported split_version")
    tables = value.get("tables")
    contexts = value.get("table_contexts")
    edges = value.get("edges")
    if not isinstance(tables, list) or len(tables) != 2:
        raise ValueError(f"line {line_number}: tables must contain exactly two values")
    normalized_tables = [_required_string(table, "table") for table in tables]
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"line {line_number}: table_contexts must not be empty")
    if not isinstance(edges, list) or not edges:
        raise ValueError(f"line {line_number}: edges must not be empty")
    for context in contexts:
        if not isinstance(context, dict):
            raise ValueError(f"line {line_number}: malformed table context")
        _required_string(context.get("table_name"), "table_context.table_name")
        _required_string(context.get("doc_id"), "table_context.doc_id")
        if not isinstance(context.get("metadata_text"), str):
            raise ValueError(f"line {line_number}: metadata_text must be a string")
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError(f"line {line_number}: malformed relationship edge")
        for field in (
            "edge_id",
            "source_table",
            "target_table",
            "source_column",
            "target_column",
            "relationship_type",
        ):
            _required_string(edge.get(field), f"edge.{field}")
        if sorted((edge["source_table"], edge["target_table"])) != normalized_tables:
            raise ValueError(f"line {line_number}: edge tables do not match group tables")
    return value | {"relationship_group_id": group_id, "split": split}


def load_relationship_groups(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"Relationship JSONL does not exist: {path}") from exc
    if not raw.strip():
        raise ValueError(f"Relationship JSONL is empty: {path}")
    groups: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        try:
            groups.append(_validate_group(value, line_number))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    duplicates = [
        group_id
        for group_id, count in Counter(
            group["relationship_group_id"] for group in groups
        ).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate relationship group IDs: {', '.join(sorted(duplicates)[:5])}")
    return groups, _sha256_bytes(raw)


def _bounded_group(group: Mapping[str, Any], max_input_tokens: int) -> dict[str, Any]:
    max_context_chars = max(100, max_input_tokens * 4 // max(1, len(group["table_contexts"])))
    primary = group["edges"][0]
    return {
        "relationship_group_index": None,
        "anchor_table": primary["source_table"],
        "anchor_column": primary["source_column"],
        "related_table": primary["target_table"],
        "related_column": primary["target_column"],
        "table_contexts": [
            {
                "table_name": context["table_name"],
                "description": context["metadata_text"][:max_context_chars],
            }
            for context in group["table_contexts"]
        ],
        "documented_edges": [
            {
                "source_table": edge["source_table"],
                "source_column": edge["source_column"],
                "target_table": edge["target_table"],
                "target_column": edge["target_column"],
                "ordinal": edge["ordinal"],
            }
            for edge in group["edges"]
        ],
    }


def _payload(
    model_name: str,
    groups: Sequence[Mapping[str, Any]],
    max_input_tokens: int,
    max_query_tokens: int,
    top_p: float,
) -> dict[str, Any]:
    bounded = []
    for index, group in enumerate(groups):
        item = _bounded_group(group, max_input_tokens)
        item["relationship_group_index"] = index
        bounded.append(item)
    return {
        "model": model_name,
        "max_tokens": min(8192, 128 + len(groups) * max_query_tokens * 2),
        "top_p": top_p,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Generate exactly one identifier-free question for every relationship "
                    "group. Preserve relationship_group_index. "
                    "Return only through the required tool.\n\n"
                    + json.dumps(bounded, ensure_ascii=False)
                ),
            }
        ],
        "tools": [
            {
                "name": "return_relationship_queries",
                "description": "Return one identifier-free question for every relationship group.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "relationship_group_index": {"type": "integer"},
                                    "identifier_free": {"type": "string", "minLength": 1},
                                },
                                "required": [
                                    "relationship_group_index",
                                    "identifier_free",
                                ],
                                "additionalProperties": False,
                            },
                            "minItems": len(groups),
                            "maxItems": len(groups),
                        }
                    },
                    "required": ["results"],
                    "additionalProperties": False,
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "return_relationship_queries"},
    }


def _extract_results(response: Any, expected_groups: int) -> list[str]:
    content = response.get("content") if isinstance(response, dict) else response.content
    if not isinstance(content, list):
        raise InvalidRelationshipResponse(
            "Claude did not return the relationship query tool result"
        )
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        block_name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
        if block_type != "tool_use" or block_name != "return_relationship_queries":
            continue
        data = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
        entries = data.get("results") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            break
        by_index: dict[int, str] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise InvalidRelationshipResponse(
                    "Claude returned a malformed relationship result"
                )
            index = entry.get("relationship_group_index")
            free = entry.get("identifier_free")
            if not isinstance(index, int) or isinstance(index, bool) or index in by_index:
                raise InvalidRelationshipResponse(
                    f"Claude returned an invalid relationship group index: {index!r}"
                )
            if not isinstance(free, str):
                raise InvalidRelationshipResponse(
                    f"Claude returned a malformed identifier-free question for group {index}"
                )
            by_index[index] = _normalize_query(free)
        expected = set(range(expected_groups))
        if set(by_index) != expected:
            raise InvalidRelationshipResponse(
                f"Claude returned group indexes {sorted(by_index)}; expected {sorted(expected)}"
            )
        return [by_index[index] for index in range(expected_groups)]
    raise InvalidRelationshipResponse("Claude did not return the relationship query tool result")


def _mentions_identifier(query: str, identifier: str) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])",
            query,
            flags=re.IGNORECASE,
        )
    )


def _identifier_aware_query(group: Mapping[str, Any]) -> str:
    """Build a stable query that exposes only the anchor side of the gold edge."""
    primary = group["edges"][0]
    return (
        f"Which documented table is related to {primary['source_table']} through "
        f"{primary['source_column']}, and which column completes the relationship?"
    )


def _validate_identifier_free(
    group: Mapping[str, Any], free: str, max_query_tokens: int
) -> None:
    if not free:
        raise InvalidRelationshipResponse("Claude returned an empty relationship question")
    if len(free) > max_query_tokens * 4:
        raise InvalidRelationshipResponse(
            "Claude returned a relationship question above the length limit"
        )
    identifiers = {
        identifier
        for edge in group["edges"]
        for identifier in (
            edge["source_table"],
            edge["target_table"],
            edge["source_column"],
            edge["target_column"],
        )
    }
    leaked = sorted(identifier for identifier in identifiers if _mentions_identifier(free, identifier))
    if leaked:
        raise InvalidRelationshipResponse(
            "Identifier-free question contains documented identifiers: " + ", ".join(leaked)
        )


class RelationshipClaudeGenerator:
    def __init__(
        self,
        model_name: str,
        *,
        client: MessagesClient | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model_name = model_name
        if client is not None:
            self._client = client
            return
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("AI_HUB_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "Relationship query generation requires ANTHROPIC_API_KEY or AI_HUB_API_KEY"
            )
        self._client = _HTTPMessagesClient(
            api_key=resolved_key,
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC_BASE_URL,
            default_headers=_headers_from_env(),
            timeout_seconds=_timeout_from_env(),
        )

    def generate(
        self,
        groups: Sequence[Mapping[str, Any]],
        *,
        relationships_per_request: int,
        max_input_tokens: int,
        max_query_tokens: int,
        top_p: float,
        max_attempts: int,
        progress_offset: int = 0,
        progress_total: int | None = None,
    ) -> list[str | None]:
        results: list[str | None] = []
        total = progress_total if progress_total is not None else len(groups)

        def request(batch: Sequence[Mapping[str, Any]]) -> list[str]:
            last_error: InvalidRelationshipResponse | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    response = self._client.messages.create(
                        **_payload(
                            self.model_name,
                            batch,
                            max_input_tokens,
                            max_query_tokens,
                            top_p,
                        )
                    )
                    generated = _extract_results(response, len(batch))
                    for group, question in zip(batch, generated, strict=True):
                        _validate_identifier_free(group, question, max_query_tokens)
                    return generated
                except InvalidRelationshipResponse as exc:
                    last_error = exc
                    if attempt < max_attempts:
                        LOGGER.warning(
                            "Invalid relationship response (attempt %d/%d): %s",
                            attempt,
                            max_attempts,
                            exc,
                        )
            assert last_error is not None
            raise last_error

        for start in range(0, len(groups), relationships_per_request):
            batch = groups[start : start + relationships_per_request]
            LOGGER.info(
                "Requesting relationship questions for groups %d-%d/%d",
                progress_offset + start + 1,
                progress_offset + start + len(batch),
                total,
            )
            try:
                results.extend(request(batch))
            except InvalidRelationshipResponse as batch_error:
                if len(batch) > 1:
                    LOGGER.warning(
                        "Batch remained invalid after %d attempts; isolating %d groups",
                        max_attempts,
                        len(batch),
                    )
                for group in batch:
                    group_id = str(group["relationship_group_id"])
                    if len(batch) == 1:
                        error = batch_error
                    else:
                        try:
                            results.extend(request([group]))
                            continue
                        except InvalidRelationshipResponse as exc:
                            error = exc
                    LOGGER.error(
                        "Skipping relationship group %s after %d attempts: %s",
                        group_id,
                        max_attempts,
                        error,
                    )
                    results.append(None)
        return results


def generate_relationship_queries(
    config: RelationshipGenerationConfig,
    *,
    generator: RelationshipClaudeGenerator | None = None,
) -> dict[str, Any]:
    config.validate()
    groups, input_hash = load_relationship_groups(config.input_path)
    selected = groups[: config.limit] if config.limit is not None else groups
    active = generator or RelationshipClaudeGenerator(config.model_name)
    questions: list[str | None] = []
    for start in range(0, len(selected), config.batch_size):
        questions.extend(
            active.generate(
                selected[start : start + config.batch_size],
                relationships_per_request=config.relationships_per_request,
                max_input_tokens=config.max_input_tokens,
                max_query_tokens=config.max_query_tokens,
                top_p=config.top_p,
                max_attempts=config.max_attempts,
                progress_offset=start,
                progress_total=len(selected),
            )
        )
    records: list[dict[str, Any]] = []
    failed_group_ids: list[str] = []
    for group, identifier_free in zip(selected, questions, strict=True):
        if identifier_free is None:
            failed_group_ids.append(str(group["relationship_group_id"]))
            continue
        identifier_aware = _identifier_aware_query(group)
        primary = group["edges"][0]
        for query_index, (query_type, query) in enumerate(
            (
                ("identifier_aware", identifier_aware),
                ("identifier_free", identifier_free),
            )
        ):
            records.append(
                {
                    "query_id": "rq_"
                    + _sha256_text(
                        f"{group['relationship_group_id']}\0{query_type}"
                    )[:16],
                    "query": query,
                    "query_type": query_type,
                    "query_index": query_index,
                    "split": group["split"],
                    "split_version": group["split_version"],
                    "relationship_group_id": group["relationship_group_id"],
                    "anchor_table": primary["source_table"],
                    "anchor_column": primary["source_column"],
                    "related_table": primary["target_table"],
                    "related_column": primary["target_column"],
                    "gold_edges": group["edges"],
                    "table_contexts": group["table_contexts"],
                    "generator_model": (
                        f"deterministic:{IDENTIFIER_AWARE_TEMPLATE_VERSION}"
                        if query_type == "identifier_aware"
                        else active.model_name
                    ),
                    "generation_seed": config.seed,
                    "generation_version": GENERATION_VERSION,
                }
            )
    lines = [json.dumps(record, sort_keys=True, ensure_ascii=False) for record in records]
    output_hash = _sha256_text("\n".join(lines) + "\n")
    _write_atomic(config.output_path, lines)
    report: dict[str, Any] = {
        "generation_version": GENERATION_VERSION,
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "generator_model": active.model_name,
        "identifier_aware_template_version": IDENTIFIER_AWARE_TEMPLATE_VERSION,
        "claude_generated_query_count": len(selected) - len(failed_group_ids),
        "seed": config.seed,
        "input_group_count": len(groups),
        "selected_group_count": len(selected),
        "successful_group_count": len(selected) - len(failed_group_ids),
        "failed_group_count": len(failed_group_ids),
        "failed_relationship_group_ids": failed_group_ids,
        "generated_query_count": len(records),
        "query_counts_by_type": {
            "identifier_aware": len(selected),
            "identifier_free": len(selected),
        },
        "input_hash": input_hash,
        "output_hash": output_hash,
        "relationships_per_request": config.relationships_per_request,
        "max_input_tokens": config.max_input_tokens,
        "max_query_tokens": config.max_query_tokens,
        "top_p": config.top_p,
    }
    _write_atomic(
        config.report_path,
        [json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)],
    )
    return report


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Generate structured relationship-retrieval questions with Claude Haiku."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--relationships-per-request", type=int, default=10)
    parser.add_argument("--max-input-tokens", type=int, default=400)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        report = generate_relationship_queries(
            RelationshipGenerationConfig(
                input_path=args.input_path,
                output_path=args.output,
                report_path=args.report,
                model_name=args.model,
                seed=args.seed,
                batch_size=args.batch_size,
                relationships_per_request=args.relationships_per_request,
                max_input_tokens=args.max_input_tokens,
                max_query_tokens=args.max_query_tokens,
                top_p=args.top_p,
                max_attempts=args.max_attempts,
                limit=args.limit,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Generated {report['generated_query_count']:,} relationship queries from "
        f"{report['selected_group_count']:,} groups; output_hash={report['output_hash']}"
    )


if __name__ == "__main__":
    main()
