from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from retrieval.indexer import build_index

from clarity_agent_evals.genq.expansion import (
    EXPANSION_VERSION,
    ExpansionConfig,
    build_expansion_corpus,
)


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


def _write_queries(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _first_chunk_id(db_path: Path, source_path: str) -> str:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT c.chunk_id
            FROM chunks AS c JOIN docs AS d ON d.doc_id = c.doc_id
            WHERE d.source_path = ?
            ORDER BY c.chunk_index
            LIMIT 1
            """,
            (source_path,),
        ).fetchone()
    assert row is not None
    return str(row[0])


def test_expansion_keeps_novel_queries_and_drops_contained_ones(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_html(corpus / "data.html")
    _write_html(corpus / "other.html", "IMPORTANT OTHER")
    db_path = tmp_path / "index.sqlite"
    build_index(corpus, db_path, workers=1)

    queries_path = tmp_path / "retained.jsonl"
    _write_queries(
        queries_path,
        [
            # Extra fields mimic a real retained record; expansion must tolerate them.
            {
                "query_id": "q1",
                "query": "Which encounters capture clinician cancellations",
                "source_file": "data.html",
                "split": "test",
            },
            # Every meaningful token is already indexed for the target row.
            {"query_id": "q2", "query": "important value info", "source_file": "data.html"},
            {"query_id": "q3", "query": "orphan question here", "source_file": "missing.html"},
        ],
    )
    output_path = tmp_path / "expansion.jsonl"
    report_path = tmp_path / "expansion_report.json"

    report = build_expansion_corpus(
        ExpansionConfig(
            queries_path=queries_path,
            db_path=db_path,
            output_path=output_path,
            report_path=report_path,
        )
    )

    assert report["expansion_version"] == EXPANSION_VERSION
    assert report["input_query_count"] == 3
    assert report["retained_query_count"] == 1
    assert report["dropped_low_novelty_count"] == 1
    assert report["unmatched_document_count"] == 1
    assert report["expanded_chunk_count"] == 1
    assert report["chunker_version"]

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["source_path"] == "data.html"
    assert rows[0]["chunk_id"] == _first_chunk_id(db_path, "data.html")
    (retained,) = rows[0]["queries"]
    assert retained["query_id"] == "q1"
    assert set(retained["novel_terms"]) >= {"encounters", "clinician", "cancellations"}

    written_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert written_report["output_hash"] == report["output_hash"]


def test_expansion_rejects_nonpositive_novelty_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_novel_terms"):
        ExpansionConfig(
            queries_path=tmp_path / "queries.jsonl",
            db_path=tmp_path / "index.sqlite",
            output_path=tmp_path / "out.jsonl",
            report_path=tmp_path / "report.json",
            min_novel_terms=0,
        ).validate()
