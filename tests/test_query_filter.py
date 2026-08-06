from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clarity_agent_evals.query_filter import FilterConfig, filter_queries

HASH = hashlib.sha256(b"value").hexdigest()


def split_chunk(
    chunk_id: str,
    *,
    source_file: str = "TABLE_A.html",
    text: str = "Table TABLE_A stores hyperspace access records and update history.",
    split: str = "train",
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_file": source_file,
        "source_hash": HASH,
        "table_name": Path(source_file).stem,
        "column_name": None,
        "chunk_type": "table_metadata",
        "section_name": "Table Metadata",
        "text": text,
        "text_hash": HASH,
        "parser_version": "epic-genq-html-v1",
        "split": split,
        "split_version": "source-hash-split-v1",
    }


def generated_query(
    query_id: str,
    query: str,
    chunk_id: str,
    *,
    source_file: str = "TABLE_A.html",
    split: str = "train",
    text_hash: str = HASH,
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query": query,
        "relevant_chunk_id": chunk_id,
        "relevant_text_hash": text_hash,
        "source_file": source_file,
        "chunk_type": "table_metadata",
        "split": split,
        "split_version": "source-hash-split-v1",
        "generator_model": "BeIR/query-gen-msmarco-t5-large-v1",
        "generation_seed": 42,
        "query_index": 0,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_config(
    tmp_path: Path,
    queries: list[dict[str, object]],
    chunks: list[dict[str, object]],
    **overrides: object,
) -> FilterConfig:
    queries_path = tmp_path / "queries.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    write_jsonl(queries_path, queries)
    write_jsonl(chunks_path, chunks)
    values: dict[str, object] = {
        "queries_path": queries_path,
        "chunks_path": chunks_path,
        "retained_output_path": tmp_path / "retained.jsonl",
        "review_output_path": tmp_path / "reviews.jsonl",
        "report_path": tmp_path / "report.json",
    }
    values.update(overrides)
    return FilterConfig(**values)  # type: ignore[arg-type]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_grounded_query_is_retained_with_overlap_evidence(tmp_path: Path) -> None:
    settings = make_config(
        tmp_path,
        [
            generated_query(
                "q_good",
                "Where are hyperspace access records stored?",
                "TABLE_A__METADATA",
            )
        ],
        [split_chunk("TABLE_A__METADATA")],
    )

    report = filter_queries(settings)
    retained = read_jsonl(settings.retained_output_path)
    reviews = read_jsonl(settings.review_output_path)

    assert report.retained_query_count == 1
    assert retained[0]["meaningful_overlap_tokens"] == [
        "access",
        "hyperspace",
        "records",
    ]
    assert reviews[0]["decision"] == "retain"
    assert reviews[0]["reason_codes"] == []


def test_too_short_query_is_rejected(tmp_path: Path) -> None:
    settings = make_config(
        tmp_path,
        [generated_query("q_short", "indices table_a", "TABLE_A__METADATA")],
        [split_chunk("TABLE_A__METADATA")],
    )

    report = filter_queries(settings)
    review = read_jsonl(settings.review_output_path)[0]

    assert report.rejected_query_count == 1
    assert report.retained_query_count == 0
    assert review["decision"] == "reject"
    assert "too_few_words" in review["reason_codes"]


def test_no_lexical_support_requires_review_instead_of_rejection(tmp_path: Path) -> None:
    settings = make_config(
        tmp_path,
        [
            generated_query(
                "q_semantic",
                "Which system tracks historical modifications?",
                "TABLE_A__METADATA",
            )
        ],
        [split_chunk("TABLE_A__METADATA")],
    )

    report = filter_queries(settings)
    review = read_jsonl(settings.review_output_path)[0]

    assert report.manual_review_query_count == 1
    assert review["decision"] == "review"
    assert review["reason_codes"] == ["no_meaningful_lexical_support"]


def test_exact_duplicate_for_same_chunk_rejects_later_query(tmp_path: Path) -> None:
    settings = make_config(
        tmp_path,
        [
            generated_query(
                "q_first",
                "Where are hyperspace access records stored?",
                "TABLE_A__METADATA",
            ),
            generated_query(
                "q_second",
                "where are hyperspace access records stored",
                "TABLE_A__METADATA",
            ),
        ],
        [split_chunk("TABLE_A__METADATA")],
    )

    report = filter_queries(settings)
    reviews = read_jsonl(settings.review_output_path)

    assert report.retained_query_count == 1
    assert report.rejected_query_count == 1
    assert reviews[1]["near_duplicate_of"] == "q_first"
    assert "exact_duplicate_same_chunk" in reviews[1]["reason_codes"]


def test_near_duplicate_for_same_chunk_is_rejected(tmp_path: Path) -> None:
    settings = make_config(
        tmp_path,
        [
            generated_query(
                "q_first",
                "Where are hyperspace access records stored?",
                "TABLE_A__METADATA",
            ),
            generated_query(
                "q_second",
                "Where are the hyperspace access records stored?",
                "TABLE_A__METADATA",
            ),
        ],
        [split_chunk("TABLE_A__METADATA")],
    )

    filter_queries(settings)
    second = read_jsonl(settings.review_output_path)[1]

    assert second["decision"] == "reject"
    assert "near_duplicate_same_chunk" in second["reason_codes"]
    assert second["near_duplicate_of"] == "q_first"


def test_same_query_with_different_positives_is_flagged_for_review(tmp_path: Path) -> None:
    query_text = "Where are hyperspace access records stored?"
    settings = make_config(
        tmp_path,
        [
            generated_query("q_a", query_text, "TABLE_A__METADATA"),
            generated_query(
                "q_b",
                query_text,
                "TABLE_B__METADATA",
                source_file="TABLE_B.html",
            ),
        ],
        [
            split_chunk("TABLE_A__METADATA"),
            split_chunk(
                "TABLE_B__METADATA",
                source_file="TABLE_B.html",
                text="Table TABLE_B stores hyperspace access records.",
            ),
        ],
    )

    report = filter_queries(settings)
    reviews = read_jsonl(settings.review_output_path)

    assert report.manual_review_query_count == 2
    assert all(review["decision"] == "review" for review in reviews)
    assert all(
        "same_query_multiple_positive_chunks" in review["reason_codes"] for review in reviews
    )


def test_long_verbatim_passage_copy_is_rejected(tmp_path: Path) -> None:
    passage = "Table TABLE_A stores hyperspace access records and update history."
    settings = make_config(
        tmp_path,
        [generated_query("q_copy", passage, "TABLE_A__METADATA")],
        [split_chunk("TABLE_A__METADATA", text=passage)],
    )

    filter_queries(settings)
    review = read_jsonl(settings.review_output_path)[0]

    assert review["decision"] == "reject"
    assert "copies_source_passage" in review["reason_codes"]


def test_original_question_about_a_terse_table_is_kept(tmp_path: Path) -> None:
    """A short description must not make an original question look copied.

    This is the case the previous rule got wrong. It scored
    SequenceMatcher(query, passage).ratio(), which is
    2*matches/(len(query) + len(passage)); when a passage is only as long as the
    query, that ratio climbs regardless of what the query says. Run at 0.35 it
    discarded 8,352 of 16,000 generated queries, and 40.8% of those reused no
    more vocabulary than the queries it kept.

    The bias had a direction: terse tables were removed preferentially, and they
    are the hardest to retrieve precisely because they offer little text to
    match on.
    """
    passage = "Table: CL_PATEDU_COMM_ID Description: Community IDs for patient education."
    query = "Where would I find the community identifiers assigned to patient education activities?"
    settings = make_config(
        tmp_path,
        [generated_query("q_terse", query, "TABLE_A__METADATA")],
        [split_chunk("TABLE_A__METADATA", text=passage)],
    )

    filter_queries(settings)
    review = read_jsonl(settings.review_output_path)[0]

    assert review["decision"] == "retain"
    assert "copies_source_passage" not in review["reason_codes"]


def test_reordered_passage_wording_is_still_rejected(tmp_path: Path) -> None:
    """Containment does not require a verbatim substring to catch restatement.

    Rearranging a description defeats a substring check while leaving every
    content word intact, so the query still adds no term the passage lacks.
    """
    passage = "Table: ZC_LIFEDOSE_STATUS Description: It is available status for each entry of lifetime dose."
    query = "What is the available status for each lifetime dose entry?"
    settings = make_config(
        tmp_path,
        [generated_query("q_reordered", query, "TABLE_A__METADATA")],
        [split_chunk("TABLE_A__METADATA", text=passage)],
    )

    filter_queries(settings)
    review = read_jsonl(settings.review_output_path)[0]

    assert review["decision"] == "reject"
    assert "copies_source_passage" in review["reason_codes"]


def test_provenance_mismatch_fails_without_writing_outputs(tmp_path: Path) -> None:
    settings = make_config(
        tmp_path,
        [
            generated_query(
                "q_bad_hash",
                "Where are hyperspace access records stored?",
                "TABLE_A__METADATA",
                text_hash=hashlib.sha256(b"wrong").hexdigest(),
            )
        ],
        [split_chunk("TABLE_A__METADATA")],
    )

    with pytest.raises(ValueError, match="inconsistent chunk provenance"):
        filter_queries(settings)
    assert not settings.retained_output_path.exists()
    assert not settings.review_output_path.exists()


def test_filter_outputs_are_byte_deterministic(tmp_path: Path) -> None:
    settings = make_config(
        tmp_path,
        [
            generated_query(
                "q_good",
                "Where are hyperspace access records stored?",
                "TABLE_A__METADATA",
            )
        ],
        [split_chunk("TABLE_A__METADATA")],
    )

    first = filter_queries(settings)
    first_retained = settings.retained_output_path.read_bytes()
    first_reviews = settings.review_output_path.read_bytes()
    second = filter_queries(settings)

    assert first.retained_queries_hash == second.retained_queries_hash
    assert first.review_ledger_hash == second.review_ledger_hash
    assert first_retained == settings.retained_output_path.read_bytes()
    assert first_reviews == settings.review_output_path.read_bytes()


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_query_words": 0},
        {"min_query_words": 5, "max_query_words": 4},
        {"near_duplicate_threshold": 0.0},
        {"passage_copy_threshold": 1.1},
    ],
)
def test_invalid_filter_configuration_is_rejected(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    settings = make_config(
        tmp_path,
        [generated_query("q", "A valid query here", "TABLE_A__METADATA")],
        [split_chunk("TABLE_A__METADATA")],
        **overrides,
    )

    with pytest.raises(ValueError):
        settings.validate()
