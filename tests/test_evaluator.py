from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

import pytest
from retrieval.indexer import build_index

from clarity_agent_evals.retrieval_evaluator import (
    RetrievalChunkTarget,
    RetrievalDocument,
    RetrievalTarget,
    TargetResolution,
    _canonical_relative_path,
    _normalize_text,
    _resolve_chunk_target,
    _resolve_target,
    load_retrieval_dataset,
    ndcg_at_k,
    precision_at_k,
    ranked_metrics,
    recall_at_k,
    reciprocal_rank,
    run_retrieval_evaluation,
)
from clarity_agent_evals.retrieval_gold import validate_semantic_dataset

BENCHMARK_DIR = Path("evals/retrieval/benchmark")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _query(
    *,
    query_id: str = "Q1",
    answerable: bool = True,
    judgment_scope: str = "corpus_complete",
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query": "important value",
        "intent": "test",
        "failure_bucket": "named_table_lookup",
        "answerable": answerable,
        "expected_answer": {
            "type": "string" if answerable else "abstain",
            "value": "answer" if answerable else None,
        },
        "answerability_rationale": "test rationale",
        "difficulty": "easy",
        "tags": ["test"],
        "split": "test",
        "judgment_scope": judgment_scope,
    }


def _target(
    *,
    target_id: str = "Q1-D01",
    query_id: str = "Q1",
    source_path: str = "data.html",
    relevance: int = 3,
) -> dict[str, object]:
    return {
        "target_id": target_id,
        "query_id": query_id,
        "document_key": f"epic_clarity:table:{Path(source_path).stem.upper()}",
        "relevance": relevance,
        "relevance_rationale": "test rationale",
    }


def _document(source_path: str = "data.html") -> dict[str, object]:
    object_name = Path(source_path).stem.upper()
    return {
        "document_key": f"epic_clarity:table:{object_name}",
        "source_system": "epic_clarity",
        "object_type": "table",
        "object_name": object_name,
        "source_path": source_path,
    }


def _chunk_target(
    *,
    target_id: str = "Q1-C01",
    query_id: str = "Q1",
    source_path: str = "data.html",
    relevance: int = 3,
    required_terms: list[str] | None = None,
) -> dict[str, object]:
    return {
        "target_id": target_id,
        "query_id": query_id,
        "document_key": f"epic_clarity:table:{Path(source_path).stem.upper()}",
        "heading_path": f"{Path(source_path).stem} > Info",
        "chunk_category": "column_info",
        "required_terms": required_terms or ["IMPORTANT", "VALUE"],
        "match_policy": "exactly_one",
        "relevance": relevance,
        "relevance_rationale": "test rationale",
    }


def _write_dataset(
    tmp_path: Path,
    queries: list[dict[str, object]],
    targets: list[dict[str, object]],
    documents: list[dict[str, object]] | None = None,
    chunk_targets: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path, Path]:
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    chunk_qrels_path = tmp_path / "chunk_qrels.jsonl"
    catalog_path = tmp_path / "catalog.jsonl"
    _write_jsonl(queries_path, queries)
    _write_jsonl(qrels_path, targets)
    if chunk_targets is None:
        chunk_targets = [
            _chunk_target(
                target_id=str(target["target_id"]).replace("-D", "-C"),
                query_id=str(target["query_id"]),
                source_path=(
                    documents[0]["source_path"]
                    if documents and len(documents) == 1
                    else "data.html"
                ),
                relevance=int(target["relevance"]),
            )
            for target in targets
        ]
    _write_jsonl(chunk_qrels_path, chunk_targets)
    _write_jsonl(catalog_path, documents or [_document()])
    return queries_path, qrels_path, chunk_qrels_path, catalog_path


def _write_html(path: Path, value: str = "IMPORTANT VALUE") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
        <html><head><title>{path.stem}</title></head><body>
        <div class="header">{path.stem}</div><div id="oContent">
          <table class="SubHeader3"><tr><td id="_Info">Info</td></tr></table>
          <table class="SubList"><tr><td>Field</td><td>{value}</td></tr></table>
        </div></body></html>
        """,
        encoding="utf-8",
    )


def test_normalization_helpers() -> None:
    assert _normalize_text("Café_under-score") == "cafe under score"
    assert _normalize_text("path\\to/file") == "path/to/file"
    assert _canonical_relative_path("./tables\\ACCESS_LOG.html") == ("tables/ACCESS_LOG.html")


@pytest.mark.parametrize(
    "source_path",
    ["/private/data.html", "../data.html", "C:\\private\\data.html", ""],
)
def test_source_path_must_be_corpus_relative(source_path: str) -> None:
    with pytest.raises(ValueError, match="source_path"):
        _canonical_relative_path(source_path)


def test_ranking_metrics_use_binary_recall_and_graded_ndcg() -> None:
    grades = [3, 0, 2]
    assert precision_at_k(grades, 2) == 0.5
    assert recall_at_k(grades, relevant_total=3, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(grades, relevant_total=0, k=2) is None
    assert reciprocal_rank(grades) == 1.0
    expected_dcg = 7 / math.log2(2) + 3 / math.log2(4)
    ideal_dcg = 7 / math.log2(2) + 3 / math.log2(3)
    assert ndcg_at_k(grades, [3, 2], 3) == pytest.approx(expected_dcg / ideal_dcg)


def test_public_ranked_metrics_calculator() -> None:
    metrics = ranked_metrics(["wrong", "relevant"], {"relevant": 1}, [1, 5])

    assert metrics["precision@1"] == 0.0
    assert metrics["precision@5"] == pytest.approx(0.2)
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@5"] == 1.0
    assert metrics["hit@5"] is True
    assert metrics["mrr"] == 0.5


def test_positives_only_suppresses_precision_and_ndcg() -> None:
    """Metrics that require judging non-positives must not be reported as measured."""
    metrics = ranked_metrics(["wrong", "relevant"], {"relevant": 1}, [1, 5], positives_only=True)

    assert metrics["recall@5"] == 1.0
    assert metrics["hit@5"] is True
    assert metrics["mrr"] == 0.5
    assert metrics["precision@1"] is None
    assert metrics["precision@5"] is None
    assert metrics["ndcg@5"] is None


def test_dataset_loads_long_term_schema(tmp_path: Path) -> None:
    queries_path, qrels_path, chunk_qrels_path, catalog_path = _write_dataset(
        tmp_path, [_query()], [_target()]
    )
    queries, targets, chunk_targets, documents = load_retrieval_dataset(
        queries_path, qrels_path, chunk_qrels_path, catalog_path
    )
    assert queries[0].intent == "test"
    assert queries[0].failure_bucket == "named_table_lookup"
    assert targets[0].document_key.startswith("epic_clarity:")
    assert chunk_targets[0].heading_path.endswith("Info")
    assert documents["epic_clarity:table:data"].source_path == "data.html"
    validate_semantic_dataset(queries_path, qrels_path, chunk_qrels_path, catalog_path)


def test_reviewed_benchmark_covers_analyst_failure_buckets() -> None:
    queries, targets, chunk_targets, _documents = load_retrieval_dataset(
        BENCHMARK_DIR / "retrieval_queries.jsonl",
        BENCHMARK_DIR / "retrieval_qrels.jsonl",
        BENCHMARK_DIR / "retrieval_chunk_qrels.jsonl",
        BENCHMARK_DIR / "retrieval_catalog.jsonl",
    )

    assert Counter(query.failure_bucket for query in queries) == {
        "named_table_lookup": 10,
        "named_column_schema": 10,
        "business_concept_discovery": 11,
        "cross_table_synthesis": 10,
        "negative_unsupported": 9,
    }
    assert len(targets) == 72
    assert len(chunk_targets) == 115
    assert all("analyst_phrased" in query.tags for query in queries)

    # The original 17 documents were the head and tail of a directory listing.
    # Coverage must include tables an analyst actually queries.
    covered = {document.object_name for document in _documents.values()}
    assert {"PAT_ENC", "CLARITY_ADT", "ORDER_MED", "CLARITY_SER", "HSP_ACCOUNT"} <= covered


def test_reviewed_queries_do_not_reuse_dictionary_probe_phrases() -> None:
    queries, _targets, _chunk_targets, _documents = load_retrieval_dataset(
        BENCHMARK_DIR / "retrieval_queries.jsonl",
        BENCHMARK_DIR / "retrieval_qrels.jsonl",
        BENCHMARK_DIR / "retrieval_chunk_qrels.jsonl",
        BENCHMARK_DIR / "retrieval_catalog.jsonl",
    )
    forbidden_probe_phrases = {
        "load type",
        "chronicles ini",
        "job-divided extract",
        "may contain ehi",
        "paper form records",
    }

    for query in queries:
        normalized = query.query.casefold()
        assert not any(phrase in normalized for phrase in forbidden_probe_phrases)


def test_dataset_rejects_index_specific_ids(tmp_path: Path) -> None:
    query = _query()
    query["doc_id"] = "generated"
    queries_path, qrels_path, chunk_qrels_path, catalog_path = _write_dataset(
        tmp_path, [query], [_target()]
    )
    with pytest.raises(ValueError, match="index-specific"):
        load_retrieval_dataset(queries_path, qrels_path, chunk_qrels_path, catalog_path)


def test_dataset_rejects_duplicate_catalog_paths(tmp_path: Path) -> None:
    second_document = _document("renamed.html")
    second_document["source_path"] = "data.html"
    queries_path, qrels_path, chunk_qrels_path, catalog_path = _write_dataset(
        tmp_path,
        [_query()],
        [_target()],
        [_document(), second_document],
    )
    with pytest.raises(ValueError, match="maps to multiple document keys"):
        load_retrieval_dataset(queries_path, qrels_path, chunk_qrels_path, catalog_path)


def test_unanswerable_query_has_no_positive_qrels(tmp_path: Path) -> None:
    queries_path, qrels_path, chunk_qrels_path, catalog_path = _write_dataset(
        tmp_path,
        [_query(answerable=False)],
        [_target(relevance=0)],
    )
    load_retrieval_dataset(queries_path, qrels_path, chunk_qrels_path, catalog_path)


def test_document_resolves_by_nested_relative_path(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    html_path = corpus / "tables" / "data.html"
    _write_html(html_path)
    db_path = tmp_path / "index.sqlite"
    build_index(corpus, db_path, workers=1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    target = RetrievalTarget.from_dict(_target(source_path="tables\\data.html"))
    document = RetrievalDocument.from_dict(_document(source_path="tables\\data.html"))
    resolved = _resolve_target(conn, target, document)
    conn.close()

    assert resolved.resolution_state == TargetResolution.resolved
    assert resolved.doc_id is not None


def test_document_resolution_survives_changed_runtime_doc_id(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    html_path = corpus / "data.html"
    _write_html(html_path)
    db_path = tmp_path / "index.sqlite"
    build_index(corpus, db_path, workers=1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    original_id = str(conn.execute("SELECT doc_id FROM docs").fetchone()["doc_id"])
    replacement_id = "different-runtime-doc-id"
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("UPDATE docs SET doc_id = ?", (replacement_id,))
    conn.execute("UPDATE chunks SET doc_id = ?", (replacement_id,))
    conn.commit()

    resolved = _resolve_target(
        conn,
        RetrievalTarget.from_dict(_target()),
        RetrievalDocument.from_dict(_document()),
    )
    conn.close()
    assert resolved.doc_id == replacement_id
    assert resolved.doc_id != original_id


def test_chunk_resolution_uses_portable_semantic_selector(tmp_path: Path) -> None:
    html_path = tmp_path / "data.html"
    _write_html(html_path)
    db_path = tmp_path / "index.sqlite"
    build_index(html_path, db_path, workers=1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    resolved = _resolve_chunk_target(
        conn,
        RetrievalChunkTarget.from_dict(_chunk_target()),
        RetrievalDocument.from_dict(_document()),
    )
    conn.close()

    assert resolved.resolution_state == TargetResolution.resolved
    assert len(resolved.chunk_ids) == 1


def test_chunk_resolution_all_policy_keeps_overlapping_matches(tmp_path: Path) -> None:
    html_path = tmp_path / "data.html"
    _write_html(html_path)
    db_path = tmp_path / "index.sqlite"
    build_index(html_path, db_path, workers=1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        INSERT INTO chunks
            (chunk_id, doc_id, chunk_index, category, heading_path, text,
             token_count, text_hash)
        SELECT 'overlap', doc_id, chunk_index + 1, category, heading_path, text,
               token_count, text_hash
        FROM chunks
        LIMIT 1
        """
    )
    conn.commit()
    target_data = _chunk_target()
    target_data["match_policy"] = "all"

    resolved = _resolve_chunk_target(
        conn,
        RetrievalChunkTarget.from_dict(target_data),
        RetrievalDocument.from_dict(_document()),
    )
    conn.close()

    assert resolved.resolution_state == TargetResolution.resolved
    assert len(resolved.chunk_ids) == 2


def test_missing_document_is_unavailable(tmp_path: Path) -> None:
    html_path = tmp_path / "data.html"
    _write_html(html_path)
    db_path = tmp_path / "index.sqlite"
    build_index(html_path, db_path, workers=1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    resolved = _resolve_target(
        conn,
        RetrievalTarget.from_dict(_target(source_path="missing.html")),
        RetrievalDocument.from_dict(_document(source_path="missing.html")),
    )
    conn.close()
    assert resolved.resolution_state == TargetResolution.unavailable_document


def test_evaluation_computes_document_metrics_only(tmp_path: Path) -> None:
    html_path = tmp_path / "data.html"
    _write_html(html_path)
    db_path = tmp_path / "index.sqlite"
    build_index(html_path, db_path, workers=1)
    queries_path, qrels_path, chunk_qrels_path, catalog_path = _write_dataset(
        tmp_path, [_query()], [_target()]
    )

    report = run_retrieval_evaluation(
        db_path=db_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        chunk_qrels_path=chunk_qrels_path,
        catalog_path=catalog_path,
        k_values=(5,),
    )

    assert report["evaluation_levels"] == ["document", "chunk"]
    assert report["resolved_query_count"] == 1
    assert report["metrics"]["document"]["hit@5"] == 1.0
    assert report["metrics"]["chunk"]["hit@5"] == 1.0
    assert report["metrics"]["by_failure_bucket"]["named_table_lookup"]["query_count"] == 1
    assert report["results"][0]["metrics_status"] == "complete"


def test_positive_only_reports_recall_but_not_precision(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_html(corpus / "data.html")
    _write_html(corpus / "other.html", "IMPORTANT OTHER")
    db_path = tmp_path / "index.sqlite"
    build_index(corpus, db_path, workers=1)
    queries_path, qrels_path, chunk_qrels_path, catalog_path = _write_dataset(
        tmp_path,
        [_query(judgment_scope="positive_only")],
        [_target()],
    )

    report = run_retrieval_evaluation(
        db_path=db_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        chunk_qrels_path=chunk_qrels_path,
        catalog_path=catalog_path,
        k_values=(5,),
    )

    # Recall and hit survive incomplete judgments because a known positive
    # either appears in the ranking or does not. Precision and nDCG would have
    # to assume every unjudged result is irrelevant, so they stay None rather
    # than reporting a number that reads as measured.
    result = report["results"][0]
    assert result["unjudged_doc_ids"]
    assert result["metrics_status"] == "recall_only"
    document = result["metrics"]["document"]
    assert document["recall@5"] == 1.0
    assert document["hit@5"] is True
    assert document["precision@5"] is None
    assert document["ndcg@5"] is None


def test_missing_positive_document_excludes_query_from_metrics(tmp_path: Path) -> None:
    html_path = tmp_path / "data.html"
    _write_html(html_path)
    db_path = tmp_path / "index.sqlite"
    build_index(html_path, db_path, workers=1)
    queries_path, qrels_path, chunk_qrels_path, catalog_path = _write_dataset(
        tmp_path,
        [_query()],
        [_target(source_path="missing.html")],
        [_document(source_path="missing.html")],
    )

    report = run_retrieval_evaluation(
        db_path=db_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        chunk_qrels_path=chunk_qrels_path,
        catalog_path=catalog_path,
        k_values=(5,),
    )

    assert report["resolved_query_count"] == 0
    assert report["unresolved_query_count"] == 1
    assert report["metrics"]["document"]["hit@5"] is None
