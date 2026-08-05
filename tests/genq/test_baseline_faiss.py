from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

# GenQ lives behind the optional `genq` dependency group, so a plain `uv sync`
# has neither faiss nor numpy. Skip rather than fail collection for the whole
# suite.
faiss = pytest.importorskip("faiss")
np = pytest.importorskip("numpy")

from clarity_agent_evals.genq import baseline_faiss  # noqa: E402
from clarity_agent_evals.genq.baseline_faiss import (  # noqa: E402
    BaselineConfig,
    TableBaselineConfig,
    _normalize_embeddings,
    run_baseline,
    run_table_baseline,
)

HASH = hashlib.sha256(b"value").hexdigest()


class FakeEncoder:
    model_name = "fake-semantic-model"
    device_name = "fake-mps"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.query_calls = 0

    def encode(self, texts: Sequence[str], *, batch_size: int, is_query: bool = False) -> Any:
        assert batch_size > 0
        if is_query:
            self.query_calls += 1
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def chunk(chunk_id: str, text: str) -> dict[str, object]:
    table_name = chunk_id.split("__", maxsplit=1)[0]
    return {
        "chunk_id": chunk_id,
        "source_file": f"{table_name}.html",
        "source_hash": HASH,
        "table_name": table_name,
        "column_name": None,
        "chunk_type": "table_metadata",
        "section_name": "Table Metadata",
        "text": text,
        "text_hash": HASH,
        "parser_version": "epic-genq-html-v1",
    }


def query(
    query_id: str,
    text: str,
    relevant_chunk_id: str,
) -> dict[str, object]:
    table_name = relevant_chunk_id.split("__", maxsplit=1)[0]
    return {
        "query_id": query_id,
        "query": text,
        "relevant_chunk_id": relevant_chunk_id,
        "relevant_text_hash": HASH,
        "source_file": f"{table_name}.html",
        "chunk_type": "table_metadata",
        "split": "test",
        "split_version": "source-hash-split-v1",
        "generator_model": "BeIR/query-gen-msmarco-t5-large-v1",
        "generation_seed": 42,
        "query_index": 0,
        "filter_version": "deterministic-query-quality-v1",
        "meaningful_overlap_tokens": ["test"],
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def config(
    tmp_path: Path,
    chunks: list[dict[str, object]],
    queries: list[dict[str, object]],
    **overrides: object,
) -> BaselineConfig:
    chunks_path = tmp_path / "chunks.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    write_jsonl(chunks_path, chunks)
    write_jsonl(queries_path, queries)
    values: dict[str, object] = {
        "chunks_path": chunks_path,
        "queries_path": queries_path,
        "output_dir": tmp_path / "baseline",
        "limit": None,
        "batch_size": 2,
        "top_k": 3,
    }
    values.update(overrides)
    return BaselineConfig(**values)  # type: ignore[arg-type]


def table_record(table_name: str, embedding_text: str) -> dict[str, object]:
    document_id = hashlib.sha256(f"document:{table_name}".encode()).hexdigest()
    return {
        "embedding_id": f"table:{document_id}",
        "document_id": document_id,
        "source_path": f"{table_name}.html",
        "source_hash": hashlib.sha256(f"source:{table_name}".encode()).hexdigest(),
        "table_name": table_name,
        "title": table_name,
        "metadata_chunk_id": f"{table_name}__METADATA",
        "metadata_text_hash": hashlib.sha256(f"metadata:{table_name}".encode()).hexdigest(),
        "description_status": "present" if "Description:" in embedding_text else "empty",
        "description": "Test description" if "Description:" in embedding_text else None,
        "embedding_text": embedding_text,
        "embedding_text_hash": hashlib.sha256(embedding_text.encode()).hexdigest(),
        "index_version": "test-index-v1",
        "extractor_version": "table-metadata-v1",
    }


def gold_query(
    query_id: str,
    query_text: str,
    *,
    failure_bucket: str,
    answerable: bool = True,
    judgment_scope: str = "corpus_complete",
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query": query_text,
        "intent": "test",
        "failure_bucket": failure_bucket,
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


def gold_document(table_name: str) -> dict[str, object]:
    return {
        "document_key": f"epic_clarity:table:{table_name}",
        "source_system": "epic_clarity",
        "object_type": "table",
        "object_name": table_name,
        "source_path": f"{table_name}.html",
    }


def gold_target(query_id: str, table_name: str) -> dict[str, object]:
    return {
        "target_id": f"{query_id}-D01",
        "query_id": query_id,
        "document_key": f"epic_clarity:table:{table_name}",
        "relevance": 3,
        "relevance_rationale": "test rationale",
    }


def gold_chunk_target(query_id: str, table_name: str) -> dict[str, object]:
    return {
        "target_id": f"{query_id}-C01",
        "query_id": query_id,
        "document_key": f"epic_clarity:table:{table_name}",
        "heading_path": f"{table_name} > Table Metadata",
        "chunk_category": "table_metadata",
        "required_terms": [table_name],
        "match_policy": "exactly_one",
        "relevance": 3,
        "relevance_rationale": "test rationale",
    }


def table_config(tmp_path: Path, *, limit: int | None = None) -> TableBaselineConfig:
    records_path = tmp_path / "table_records.jsonl"
    write_jsonl(
        records_path,
        [
            table_record("TABLE_A", "Table: TABLE_A\nDescription: Test description"),
            table_record("TABLE_B", "Table: TABLE_B\nDescription: Test description"),
            table_record("TABLE_C", "Table: TABLE_C"),
        ],
    )
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    write_jsonl(
        benchmark_dir / "retrieval_queries.jsonl",
        [
            gold_query("Q_A", "query a", failure_bucket="named_table_lookup"),
            gold_query(
                "Q_B",
                "query b",
                failure_bucket="business_concept_discovery",
                judgment_scope="positive_only",
            ),
            gold_query(
                "Q_NO",
                "query no answer",
                failure_bucket="negative_unsupported",
                answerable=False,
            ),
        ],
    )
    write_jsonl(
        benchmark_dir / "retrieval_qrels.jsonl",
        [gold_target("Q_A", "TABLE_A"), gold_target("Q_B", "TABLE_B")],
    )
    write_jsonl(
        benchmark_dir / "retrieval_chunk_qrels.jsonl",
        [
            gold_chunk_target("Q_A", "TABLE_A"),
            gold_chunk_target("Q_B", "TABLE_B"),
        ],
    )
    write_jsonl(
        benchmark_dir / "retrieval_catalog.jsonl",
        [gold_document(name) for name in ("TABLE_A", "TABLE_B", "TABLE_C")],
    )
    return TableBaselineConfig(
        records_path=records_path,
        benchmark_dir=benchmark_dir,
        output_dir=tmp_path / "table-baseline",
        limit=limit,
        batch_size=2,
        top_k=2,
    )


def test_exact_faiss_mapping_and_metrics(tmp_path: Path) -> None:
    chunks = [
        chunk("TABLE_A__METADATA", "passage a"),
        chunk("TABLE_B__METADATA", "passage b"),
        chunk("TABLE_C__METADATA", "passage c"),
    ]
    queries = [
        query("q_a", "query a", "TABLE_A__METADATA"),
        query("q_b", "query b", "TABLE_B__METADATA"),
    ]
    encoder = FakeEncoder(
        {
            "passage a": [1.0, 0.0],
            "passage b": [0.8, 0.6],
            "passage c": [0.0, 1.0],
            "query a": [1.0, 0.0],
            "query b": [1.0, 0.0],
        }
    )
    settings = config(tmp_path, chunks, queries)

    report = run_baseline(settings, encoder=encoder)

    assert report.evaluated_query_count == 2
    assert report.hit_at_1 == pytest.approx(0.5)
    assert report.hit_at_5 == pytest.approx(1.0)
    assert report.hit_at_10 == pytest.approx(1.0)
    assert report.mrr == pytest.approx(0.75)
    assert report.precision_at_1 == pytest.approx(0.5)
    assert report.precision_at_5 == pytest.approx(0.2)
    assert report.precision_at_10 == pytest.approx(0.1)
    assert report.recall_at_1 == pytest.approx(0.5)
    assert report.recall_at_5 == pytest.approx(1.0)
    assert report.recall_at_10 == pytest.approx(1.0)
    assert [result.positive_rank for result in report.query_results] == [1, 2]

    mapping = [
        json.loads(line)
        for line in (settings.output_dir / "chunk_mapping.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(row["vector_position"], row["chunk_id"]) for row in mapping] == [
        (0, "TABLE_A__METADATA"),
        (1, "TABLE_B__METADATA"),
        (2, "TABLE_C__METADATA"),
    ]
    index = faiss.read_index(str(settings.output_dir / "corpus.faiss"))
    assert index.ntotal == 3
    assert index.d == 2


def test_embeddings_are_normalized_before_inner_product_search(tmp_path: Path) -> None:
    settings = config(
        tmp_path,
        [chunk("TABLE_A__METADATA", "passage a")],
        [query("q_a", "query a", "TABLE_A__METADATA")],
    )
    encoder = FakeEncoder({"passage a": [20.0, 0.0], "query a": [5.0, 0.0]})

    report = run_baseline(settings, encoder=encoder)

    assert report.query_results[0].top_hits[0].score == pytest.approx(1.0)


def test_baseline_accepts_leakage_safe_split_chunks(tmp_path: Path) -> None:
    split_chunk = {
        **chunk("TABLE_A__METADATA", "passage a"),
        "split": "test",
        "split_version": "metadata-source-hash-split-v1",
    }
    settings = config(
        tmp_path,
        [split_chunk],
        [query("q_a", "query a", "TABLE_A__METADATA")],
    )
    encoder = FakeEncoder({"passage a": [1.0, 0.0], "query a": [1.0, 0.0]})

    report = run_baseline(settings, encoder=encoder)

    assert report.evaluated_query_count == 1
    assert report.hit_at_1 == 1.0


def test_table_metadata_baseline_uses_gold_document_qrels(tmp_path: Path) -> None:
    settings = table_config(tmp_path)
    encoder = FakeEncoder(
        {
            "Table: TABLE_A\nDescription: Test description": [1.0, 0.0],
            "Table: TABLE_B\nDescription: Test description": [0.8, 0.6],
            "Table: TABLE_C": [0.0, 1.0],
            "query a": [1.0, 0.0],
            "query b": [1.0, 0.0],
            "query no answer": [0.0, 1.0],
        }
    )

    report = run_table_baseline(settings, encoder=encoder)

    assert report.query_count == 3
    assert report.answerable_query_count == 2
    assert report.unanswerable_query_count == 1
    assert report.candidate_document_count == 3
    assert report.metrics["document"]["hit@1"] == pytest.approx(0.5)
    assert report.metrics["document"]["hit@5"] == pytest.approx(1.0)
    assert report.metrics["document"]["mrr"] == pytest.approx(0.75)
    assert (
        report.metrics["by_failure_bucket"]["business_concept_discovery"]["document"]["precision@1"]
        is None
    )
    assert [result["positive_ranks"] for result in report.results[:2]] == [[1], [2]]
    assert report.results[2]["metrics"] is None
    assert report.results[2]["no_answer_diagnostic"]["top_score"] == pytest.approx(1.0)

    mapping = [
        json.loads(line)
        for line in (settings.output_dir / "table_mapping.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(row["vector_position"], row["table_name"]) for row in mapping] == [
        (0, "TABLE_A"),
        (1, "TABLE_B"),
        (2, "TABLE_C"),
    ]
    assert mapping[2]["description_status"] == "empty"
    assert (settings.output_dir / "gold_evaluation.json").is_file()
    index = faiss.read_index(str(settings.output_dir / "tables.faiss"))
    assert index.ntotal == 3
    assert index.d == 2


def test_bounded_table_index_must_include_every_gold_positive(tmp_path: Path) -> None:
    settings = table_config(tmp_path, limit=1)

    with pytest.raises(ValueError, match="excludes positive gold documents"):
        run_table_baseline(settings, encoder=FakeEncoder({}))
    assert not settings.output_dir.exists()


def test_table_embedding_text_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    settings = table_config(tmp_path)
    rows = [json.loads(line) for line in settings.records_path.read_text().splitlines()]
    rows[0]["embedding_text_hash"] = HASH
    write_jsonl(settings.records_path, rows)

    with pytest.raises(ValueError, match="Embedding text hash disagrees"):
        run_table_baseline(settings, encoder=FakeEncoder({}))


def test_bounded_index_must_include_every_evaluation_positive(tmp_path: Path) -> None:
    settings = config(
        tmp_path,
        [
            chunk("TABLE_A__METADATA", "passage a"),
            chunk("TABLE_B__METADATA", "passage b"),
        ],
        [query("q_b", "query b", "TABLE_B__METADATA")],
        limit=1,
    )

    with pytest.raises(ValueError, match="excludes positive chunks"):
        run_baseline(
            settings,
            encoder=FakeEncoder({"passage a": [1.0], "query b": [1.0]}),
        )
    assert not settings.output_dir.exists()


def test_query_provenance_mismatch_is_rejected(tmp_path: Path) -> None:
    bad_query = query("q_a", "query a", "TABLE_A__METADATA")
    bad_query["relevant_text_hash"] = hashlib.sha256(b"wrong").hexdigest()
    settings = config(
        tmp_path,
        [chunk("TABLE_A__METADATA", "passage a")],
        [bad_query],
    )

    with pytest.raises(ValueError, match="inconsistent chunk provenance"):
        run_baseline(
            settings,
            encoder=FakeEncoder({"passage a": [1.0], "query a": [1.0]}),
        )


@pytest.mark.parametrize(
    "vectors",
    [
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([[float("nan"), 1.0]], dtype=np.float32),
        np.asarray([1.0, 2.0], dtype=np.float32),
    ],
)
def test_invalid_embeddings_are_rejected(vectors: Any) -> None:
    with pytest.raises(ValueError):
        _normalize_embeddings(vectors, expected_rows=1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"limit": 0},
        {"batch_size": 0},
        {"top_k": 0},
        {"device": "quantum"},
        {"model_name": " "},
    ],
)
def test_invalid_baseline_configuration_is_rejected(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    settings = config(
        tmp_path,
        [chunk("TABLE_A__METADATA", "passage a")],
        [query("q_a", "query a", "TABLE_A__METADATA")],
        **overrides,
    )

    with pytest.raises(ValueError):
        settings.validate()


def test_build_encoder_selects_last_token_pooling_for_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pooling must follow the checkpoint, not the caller.

    Loading Qwen3-Embedding through the mean-pooling path yields embeddings that
    look valid and retrieve badly, so selection is asserted explicitly.
    """
    built: list[str] = []

    class FakeQwen:
        def __init__(self, model_name: str, device: str, **kwargs: object) -> None:
            built.append(f"qwen:{model_name}")

    class FakeST:
        def __init__(self, model_name: str, device: str, **kwargs: object) -> None:
            built.append(f"st:{model_name}")

    monkeypatch.setattr(baseline_faiss, "QwenEmbeddingEncoder", FakeQwen)
    monkeypatch.setattr(baseline_faiss, "SentenceTransformerEncoder", FakeST)

    baseline_faiss.build_encoder("Qwen/Qwen3-Embedding-0.6B", "cpu")
    baseline_faiss.build_encoder("qwen/qwen3-embedding-8B", "cpu")
    baseline_faiss.build_encoder("sentence-transformers/all-MiniLM-L6-v2", "cpu")

    assert built == [
        "qwen:Qwen/Qwen3-Embedding-0.6B",
        "qwen:qwen/qwen3-embedding-8B",
        "st:sentence-transformers/all-MiniLM-L6-v2",
    ]


def test_qwen_pooling_takes_the_last_real_token() -> None:
    torch = pytest.importorskip("torch")
    encoder = object.__new__(baseline_faiss.QwenEmbeddingEncoder)
    encoder._torch = torch

    hidden = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
        ]
    )
    # Left padding: the final column is a real token for every row.
    left_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    assert encoder._pool(hidden, left_mask).tolist() == [[3.0, 3.0], [6.0, 6.0]]

    # Right padding: row 0 ends at position 1, so position 2 is padding.
    right_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert encoder._pool(hidden, right_mask).tolist() == [[2.0, 2.0], [6.0, 6.0]]


def test_qwen_query_formatting_wraps_only_queries() -> None:
    encoder = object.__new__(baseline_faiss.QwenEmbeddingEncoder)
    encoder._task = "Find the passage"

    assert encoder._format("A0H_MAP stores access records.", is_query=False) == (
        "A0H_MAP stores access records."
    )
    assert encoder._format("which table has access records?", is_query=True) == (
        "Instruct: Find the passage\nQuery:which table has access records?"
    )


def test_default_model_is_the_qwen_encoder() -> None:
    assert baseline_faiss.DEFAULT_MODEL == "Qwen/Qwen3-Embedding-0.6B"
    assert baseline_faiss.LEGACY_MINILM_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
