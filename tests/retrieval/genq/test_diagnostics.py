from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clarity_agent_evals.genq.diagnostics import (
    LEAKAGE_CONCERN_THRESHOLD,
    DiagnosticsConfig,
    heaps_beta,
    leakage,
    length_profile,
    near_duplicate_median,
    run_diagnostics,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _chunk_line(chunk_id: str, text: str, source_file: str = "TABLE_A.html") -> str:
    return json.dumps(
        {
            "chunk_id": chunk_id,
            "source_file": source_file,
            "source_hash": _hash(source_file),
            "table_name": Path(source_file).stem,
            "column_name": None,
            "chunk_type": "table_metadata",
            "section_name": Path(source_file).stem,
            "text": text,
            "text_hash": _hash(text),
            "parser_version": "test-v1",
            "split": "train",
            "split_version": "test-split-v1",
        },
        sort_keys=True,
    )


def _query_line(query_id: str, query: str, chunk_id: str, decision: str = "retain") -> str:
    return json.dumps(
        {"query_id": query_id, "query": query, "relevant_chunk_id": chunk_id, "decision": decision},
        sort_keys=True,
    )


def test_leakage_is_one_when_the_query_reuses_only_passage_words() -> None:
    passage = "Table FOO stores admission encounter records for inpatient stays."
    query = "admission encounter records inpatient stays"

    all_terms, content_terms = leakage([(query, passage)])

    assert all_terms == pytest.approx(1.0)
    assert content_terms == pytest.approx(1.0)


def test_leakage_ignores_passage_length() -> None:
    """The measure divides by the query, so padding the passage cannot move it.

    This is the property the previous copy check lacked: it scored
    2*matches/(len(query) + len(passage)), so a longer passage lowered the score
    for an identical query and a terse table looked like plagiarism.
    """
    query = "which admissions are recorded"
    short = "Admissions are recorded here."
    padded = short + " " + " ".join(f"filler{i}" for i in range(400))

    short_score = leakage([(query, short)])[1]
    padded_score = leakage([(query, padded)])[1]

    assert short_score == pytest.approx(padded_score)


def test_heaps_beta_separates_a_growing_set_from_a_templated_one() -> None:
    growing = [f"question about unique subject {i} and distinct topic {i}" for i in range(400)]
    templated = ["what does this table store for reporting"] * 400

    growing_beta, growing_vocabulary = heaps_beta(growing)
    templated_beta, templated_vocabulary = heaps_beta(templated)

    assert growing_beta > templated_beta
    assert templated_beta == pytest.approx(0.0, abs=0.05)
    assert growing_vocabulary > templated_vocabulary


def test_near_duplicate_median_is_high_for_a_repeated_query() -> None:
    repeated = ["identical admission encounter question"] * 50
    varied = [f"distinct subject {i} unrelated matter {i}" for i in range(50)]

    repeated_score, _ = near_duplicate_median(repeated)
    varied_score, _ = near_duplicate_median(varied)

    assert repeated_score == pytest.approx(1.0)
    assert varied_score < repeated_score


def test_length_profile_reports_the_mode() -> None:
    profile = length_profile(["one two three"] * 10 + ["one two three four five"])

    assert profile.mode == 3
    assert profile.median == 3


def test_report_flags_a_set_that_only_rewards_term_overlap(tmp_path: Path) -> None:
    """A set whose queries are cut from their passages must be flagged, not scored.

    Nothing downstream can recover from this: an inverted index finds the
    positive passage by term intersection, so every retriever scores near the
    ceiling and the benchmark ranks nothing.
    """
    passage = "Table FOO stores admission encounter records for inpatient stays."
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(_chunk_line("c1", passage) + "\n", encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "\n".join(
            _query_line(f"q{i}", "admission encounter records inpatient stays", "c1")
            for i in range(12)
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_diagnostics(
        DiagnosticsConfig(queries_path=queries, chunks_path=chunks, report_path=tmp_path / "r.json")
    )

    assert report.leakage_content_terms >= LEAKAGE_CONCERN_THRESHOLD
    assert report.leakage_exceeds_threshold is True
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["query_count"] == 12


def test_original_queries_are_not_flagged(tmp_path: Path) -> None:
    passage = "Table FOO stores admission encounter records for inpatient stays."
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(_chunk_line("c1", passage) + "\n", encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "\n".join(
            _query_line(f"q{i}", f"how do I bill overnight observation cases {i}", "c1")
            for i in range(12)
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_diagnostics(DiagnosticsConfig(queries_path=queries, chunks_path=chunks))

    assert report.leakage_exceeds_threshold is False


def test_decision_filter_selects_one_group(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(_chunk_line("c1", "Table FOO stores admissions.") + "\n", encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        _query_line("q1", "how are admissions recorded", "c1", "retain")
        + "\n"
        + _query_line("q2", "table foo stores admissions", "c1", "reject")
        + "\n",
        encoding="utf-8",
    )

    report = run_diagnostics(
        DiagnosticsConfig(queries_path=queries, chunks_path=chunks, decision="retain")
    )

    assert report.query_count == 1


def test_config_rejects_an_unusable_neighbour_sample(tmp_path: Path) -> None:
    config = DiagnosticsConfig(
        queries_path=tmp_path / "q.jsonl", chunks_path=tmp_path / "c.jsonl", neighbour_sample=1
    )
    with pytest.raises(ValueError, match="neighbour_sample"):
        config.validate()


def test_missing_chunks_file_is_reported_by_path(tmp_path: Path) -> None:
    queries = tmp_path / "queries.jsonl"
    queries.write_text(_query_line("q1", "anything at all here", "c1") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="chunks JSONL does not exist"):
        run_diagnostics(DiagnosticsConfig(queries_path=queries, chunks_path=tmp_path / "no.jsonl"))
