from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from clarity_agent_evals.relationship_query_generation import (
    RelationshipClaudeGenerator,
    RelationshipGenerationConfig,
    generate_relationship_queries,
    load_relationship_groups,
)


def relationship_group() -> dict[str, Any]:
    return {
        "relationship_group_id": "group-1",
        "tables": ["SOURCE_A", "TARGET_A"],
        "split": "validation",
        "split_version": "relationship-pair-split-v1",
        "table_contexts": [
            {
                "table_name": "SOURCE_A",
                "doc_id": "source-doc",
                "title": "Source A",
                "source_path": "SOURCE_A.html",
                "metadata_text": "Stores scheduling events for operational reporting.",
            },
            {
                "table_name": "TARGET_A",
                "doc_id": "target-doc",
                "title": "Target A",
                "source_path": "TARGET_A.html",
                "metadata_text": "Stores billing activity associated with operational events.",
            },
        ],
        "edges": [
            {
                "edge_id": "edge-1",
                "source_doc_id": "source-doc",
                "target_doc_id": "target-doc",
                "source_table": "SOURCE_A",
                "target_table": "TARGET_A",
                "source_column": "SOURCE_ID",
                "target_column": "TARGET_ID",
                "ordinal": 1,
                "relationship_type": "foreign_key",
                "evidence_chunk_id": "evidence-1",
            }
        ],
    }


def write_input(path: Path, groups: list[dict[str, Any]] | None = None) -> None:
    path.write_text(
        "".join(json.dumps(group) + "\n" for group in (groups or [relationship_group()])),
        encoding="utf-8",
    )


def tool_response(
    *,
    free: str = "Which datasets connect scheduling events with associated billing activity?",
) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "tool_use",
                "name": "return_relationship_queries",
                "input": {
                    "results": [
                        {
                            "relationship_group_index": 0,
                            "identifier_free": free,
                        }
                    ]
                },
            }
        ]
    }


class FakeMessages:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **payload: Any) -> dict[str, Any]:
        self.calls.append(payload)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.messages = FakeMessages(responses)


def make_config(tmp_path: Path, input_path: Path) -> RelationshipGenerationConfig:
    return RelationshipGenerationConfig(
        input_path=input_path,
        output_path=tmp_path / "generated.jsonl",
        report_path=tmp_path / "report.json",
        relationships_per_request=1,
    )


def test_generator_preserves_gold_relationship_and_query_types(tmp_path: Path) -> None:
    input_path = tmp_path / "relationships.jsonl"
    write_input(input_path)
    client = FakeClient([tool_response()])
    generator = RelationshipClaudeGenerator("haiku-test", client=client)
    config = make_config(tmp_path, input_path)

    report = generate_relationship_queries(config, generator=generator)
    records = [
        json.loads(line) for line in config.output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert report["generated_query_count"] == 2
    assert [record["query_type"] for record in records] == [
        "identifier_aware",
        "identifier_free",
    ]
    assert all(record["relationship_group_id"] == "group-1" for record in records)
    assert all(record["gold_edges"][0]["edge_id"] == "edge-1" for record in records)
    assert records[0]["anchor_table"] == "SOURCE_A"
    assert records[0]["related_table"] == "TARGET_A"
    assert "SOURCE_A" in records[0]["query"]
    assert "TARGET_A" not in records[0]["query"]
    assert records[0]["generator_model"].startswith("deterministic:")
    assert records[1]["generator_model"] == "haiku-test"
    payload = client.messages.calls[0]
    assert payload["tool_choice"]["name"] == "return_relationship_queries"
    assert payload["tools"][0]["input_schema"]["additionalProperties"] is False


def test_identifier_leakage_retries_then_accepts_valid_response(tmp_path: Path) -> None:
    input_path = tmp_path / "relationships.jsonl"
    write_input(input_path)
    client = FakeClient(
        [
            tool_response(free="How does SOURCE_ID connect scheduling and billing?"),
            tool_response(),
        ]
    )
    generator = RelationshipClaudeGenerator("haiku-test", client=client)

    generate_relationship_queries(make_config(tmp_path, input_path), generator=generator)

    assert len(client.messages.calls) == 2


def test_persistent_identifier_free_leakage_is_reported_and_skipped(tmp_path: Path) -> None:
    input_path = tmp_path / "relationships.jsonl"
    write_input(input_path)
    client = FakeClient(
        [tool_response(free="How does SOURCE_ID connect scheduling and billing?")] * 2
    )
    generator = RelationshipClaudeGenerator("haiku-test", client=client)
    config = RelationshipGenerationConfig(
        **{**make_config(tmp_path, input_path).__dict__, "max_attempts": 2}
    )

    report = generate_relationship_queries(config, generator=generator)

    assert report["generated_query_count"] == 0
    assert report["failed_group_count"] == 1
    assert report["failed_relationship_group_ids"] == ["group-1"]
    assert config.output_path.read_text(encoding="utf-8") == ""


def test_input_rejects_edges_from_another_table_pair(tmp_path: Path) -> None:
    input_path = tmp_path / "relationships.jsonl"
    group = relationship_group()
    group["edges"][0]["target_table"] = "WRONG_TABLE"
    write_input(input_path, [group])

    with pytest.raises(ValueError, match="edge tables do not match group tables"):
        load_relationship_groups(input_path)


def test_bad_batch_is_isolated_and_only_bad_group_is_skipped(tmp_path: Path) -> None:
    input_path = tmp_path / "relationships.jsonl"
    first = relationship_group()
    second = deepcopy(first)
    second["relationship_group_id"] = "group-2"
    second["tables"] = ["SOURCE_B", "TARGET_B"]
    for context, name in zip(
        second["table_contexts"], ("SOURCE_B", "TARGET_B"), strict=True
    ):
        context["table_name"] = name
    edge = second["edges"][0]
    edge.update(
        {
            "edge_id": "edge-2",
            "source_table": "SOURCE_B",
            "target_table": "TARGET_B",
            "source_column": "SOURCE_B_ID",
            "target_column": "TARGET_B_ID",
        }
    )
    write_input(input_path, [first, second])
    client = FakeClient(
        [
            tool_response(),  # Invalid for a two-group batch: index 1 is missing.
            tool_response(),  # First isolated group succeeds.
            tool_response(free="How does SOURCE_B_ID connect these records?"),
        ]
    )
    generator = RelationshipClaudeGenerator("haiku-test", client=client)
    base = make_config(tmp_path, input_path)
    config = RelationshipGenerationConfig(
        **{
            **base.__dict__,
            "batch_size": 2,
            "relationships_per_request": 2,
            "max_attempts": 1,
        }
    )

    report = generate_relationship_queries(config, generator=generator)

    assert report["successful_group_count"] == 1
    assert report["failed_group_count"] == 1
    assert report["failed_relationship_group_ids"] == ["group-2"]
    assert report["generated_query_count"] == 2
