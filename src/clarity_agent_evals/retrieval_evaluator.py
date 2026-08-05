from __future__ import annotations

import json
import math
import re
import sqlite3
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any

from retrieval.search import search_ranked_chunks

DOCUMENT_NAMESPACE = "epic_clarity"


def make_document_key(object_type: str, object_name: str) -> str:
    return f"{DOCUMENT_NAMESPACE}:{object_type.strip().lower()}:{object_name.strip().upper()}"


class JudgmentScope(StrEnum):
    positive_only = "positive_only"
    top_k_complete = "top_k_complete"
    corpus_complete = "corpus_complete"


FAILURE_BUCKETS = frozenset(
    {
        "named_table_lookup",
        "named_column_schema",
        "business_concept_discovery",
        "cross_table_synthesis",
        "negative_unsupported",
    }
)


@dataclass(frozen=True)
class RetrievalQuery:
    query_id: str
    query: str
    intent: str
    failure_bucket: str
    answerable: bool
    expected_answer: Mapping[str, Any]
    answerability_rationale: str
    difficulty: str
    tags: tuple[str, ...]
    split: str
    judgment_scope: JudgmentScope

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetrievalQuery:
        expected_answer = data["expected_answer"]
        if not isinstance(expected_answer, Mapping):
            raise TypeError("expected_answer must be an object")
        tags = data.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise TypeError("tags must be a list of strings")
        failure_bucket = str(data["failure_bucket"])
        if failure_bucket not in FAILURE_BUCKETS:
            raise ValueError(f"failure_bucket must be one of {', '.join(sorted(FAILURE_BUCKETS))}")
        return cls(
            query_id=str(data["query_id"]),
            query=str(data["query"]),
            intent=str(data["intent"]),
            failure_bucket=failure_bucket,
            answerable=bool(data["answerable"]),
            expected_answer=dict(expected_answer),
            answerability_rationale=str(data["answerability_rationale"]),
            difficulty=str(data["difficulty"]),
            tags=tuple(tags),
            split=str(data["split"]),
            judgment_scope=JudgmentScope(str(data["judgment_scope"])),
        )


@dataclass(frozen=True)
class RetrievalDocument:
    document_key: str
    source_system: str
    object_type: str
    object_name: str
    source_path: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetrievalDocument:
        source_system = str(data["source_system"]).strip().lower()
        object_type = str(data["object_type"]).strip().lower()
        object_name = str(data["object_name"]).strip().upper()
        document_key = str(data["document_key"]).strip()
        expected_key = make_document_key(object_type, object_name)
        if document_key != expected_key:
            raise ValueError(f"document_key {document_key!r} must equal {expected_key!r}")
        if source_system != DOCUMENT_NAMESPACE:
            raise ValueError(f"source_system must be {DOCUMENT_NAMESPACE}")
        return cls(
            document_key=document_key,
            source_system=source_system,
            object_type=object_type,
            object_name=object_name,
            source_path=_canonical_relative_path(str(data["source_path"])),
        )


@dataclass(frozen=True)
class RetrievalTarget:
    target_id: str
    query_id: str
    document_key: str
    relevance: int
    relevance_rationale: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetrievalTarget:
        relevance = int(data["relevance"])
        if relevance not in {0, 1, 2, 3}:
            raise ValueError("relevance must be one of 0, 1, 2, or 3")
        document_key = str(data["document_key"]).strip()
        if not document_key:
            raise ValueError("document_key must not be empty")
        return cls(
            target_id=str(data["target_id"]),
            query_id=str(data["query_id"]),
            document_key=document_key,
            relevance=relevance,
            relevance_rationale=str(data["relevance_rationale"]),
        )


@dataclass(frozen=True)
class RetrievalChunkTarget:
    target_id: str
    query_id: str
    document_key: str
    heading_path: str
    chunk_category: str
    required_terms: tuple[str, ...]
    match_policy: str
    relevance: int
    relevance_rationale: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetrievalChunkTarget:
        relevance = int(data["relevance"])
        if relevance not in {0, 1, 2, 3}:
            raise ValueError("relevance must be one of 0, 1, 2, or 3")
        required_terms = data["required_terms"]
        if (
            not isinstance(required_terms, list)
            or not required_terms
            or not all(isinstance(term, str) and term.strip() for term in required_terms)
        ):
            raise TypeError("required_terms must be a non-empty list of strings")
        heading_path = str(data["heading_path"]).strip()
        chunk_category = str(data["chunk_category"]).strip().lower()
        if not heading_path or not chunk_category:
            raise ValueError("heading_path and chunk_category must not be empty")
        match_policy = str(data["match_policy"])
        if match_policy not in {"exactly_one", "all"}:
            raise ValueError("match_policy must be exactly_one or all")
        return cls(
            target_id=str(data["target_id"]),
            query_id=str(data["query_id"]),
            document_key=str(data["document_key"]).strip(),
            heading_path=heading_path,
            chunk_category=chunk_category,
            required_terms=tuple(required_terms),
            match_policy=match_policy,
            relevance=relevance,
            relevance_rationale=str(data["relevance_rationale"]),
        )


class TargetResolution(StrEnum):
    resolved = "resolved"
    unavailable_document = "unavailable_document"
    ambiguous_document = "ambiguous_document"
    unavailable_evidence = "unavailable_evidence"
    ambiguous_chunk = "ambiguous_chunk"


@dataclass(frozen=True)
class ResolvedTarget:
    target: RetrievalTarget
    document: RetrievalDocument
    resolution_state: TargetResolution
    doc_id: str | None
    diagnostic: str | None


@dataclass(frozen=True)
class ResolvedChunkTarget:
    target: RetrievalChunkTarget
    document: RetrievalDocument
    resolution_state: TargetResolution
    doc_id: str | None
    chunk_ids: tuple[str, ...]
    diagnostic: str | None


@dataclass(frozen=True)
class ResolvableChunk:
    chunk_id: str
    heading_path: str
    category: str
    text: str


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: str
    doc_id: str
    source_path: str
    heading_path: str
    score: float | None


def _load_resolvable_chunks(conn: sqlite3.Connection, doc_id: str) -> list[ResolvableChunk]:
    return [
        ResolvableChunk(
            chunk_id=str(row["chunk_id"]),
            heading_path=_normalize_text(str(row["heading_path"])),
            category=str(row["category"]).casefold(),
            text=_normalize_text(str(row["text"])),
        )
        for row in conn.execute(
            "SELECT chunk_id, heading_path, category, text FROM chunks WHERE doc_id = ?",
            (doc_id,),
        )
    ]


def _load_resolvable_chunks_by_document(
    conn: sqlite3.Connection,
    document_matches_by_key: Mapping[str, Sequence[sqlite3.Row]],
) -> dict[str, list[ResolvableChunk]]:
    key_by_doc_id = {
        str(matches[0]["doc_id"]): key
        for key, matches in document_matches_by_key.items()
        if len(matches) == 1
    }
    chunks_by_key: dict[str, list[ResolvableChunk]] = {key: [] for key in document_matches_by_key}
    for row in conn.execute("SELECT chunk_id, doc_id, heading_path, category, text FROM chunks"):
        key = key_by_doc_id.get(str(row["doc_id"]))
        if key is None:
            continue
        chunks_by_key[key].append(
            ResolvableChunk(
                chunk_id=str(row["chunk_id"]),
                heading_path=_normalize_text(str(row["heading_path"])),
                category=str(row["category"]).casefold(),
                text=_normalize_text(str(row["text"])),
            )
        )
    return chunks_by_key


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows found")
    return rows


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[_-]", " ", text)
    text = re.sub(r"[\\/]+", "/", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def _canonical_relative_path(value: str) -> str:
    raw = value.strip().replace("\\", "/")
    if not raw:
        raise ValueError("source_path must not be empty")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
        raise ValueError("source_path must be relative to the corpus root")
    parts = [part for part in PurePosixPath(raw).parts if part not in {"", "."}]
    if not parts or ".." in parts:
        raise ValueError("source_path must not escape the corpus root")
    return PurePosixPath(*parts).as_posix()


def load_retrieval_dataset(
    queries_path: Path,
    qrels_path: Path,
    chunk_qrels_path: Path,
    catalog_path: Path,
) -> tuple[
    list[RetrievalQuery],
    list[RetrievalTarget],
    list[RetrievalChunkTarget],
    dict[str, RetrievalDocument],
]:
    query_rows = _load_jsonl(queries_path)
    target_rows = _load_jsonl(qrels_path)
    chunk_target_rows = _load_jsonl(chunk_qrels_path)
    catalog_rows = _load_jsonl(catalog_path)
    forbidden_fields = {"chunk_id", "doc_id", "index_version"}
    for row in [*query_rows, *target_rows, *chunk_target_rows, *catalog_rows]:
        found = forbidden_fields & set(row)
        if found:
            raise ValueError(f"dataset contains index-specific fields: {sorted(found)}")

    try:
        queries = [RetrievalQuery.from_dict(row) for row in query_rows]
        targets = [RetrievalTarget.from_dict(row) for row in target_rows]
        chunk_targets = [RetrievalChunkTarget.from_dict(row) for row in chunk_target_rows]
        documents = [RetrievalDocument.from_dict(row) for row in catalog_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid retrieval evaluation dataset: {exc}") from exc

    query_ids = [query.query_id for query in queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("retrieval query IDs must be unique")
    known_query_ids = set(query_ids)
    document_by_key: dict[str, RetrievalDocument] = {}
    path_to_key: dict[str, str] = {}
    for document in documents:
        folded_key = document.document_key.casefold()
        if folded_key in document_by_key:
            raise ValueError(f"duplicate document_key {document.document_key}")
        folded_path = document.source_path.casefold()
        if folded_path in path_to_key:
            raise ValueError(f"source_path {document.source_path} maps to multiple document keys")
        document_by_key[folded_key] = document
        path_to_key[folded_path] = document.document_key

    seen_target_ids: set[str] = set()
    seen_query_documents: set[tuple[str, str]] = set()
    positive_counts: dict[str, int] = defaultdict(int)

    for target in [*targets, *chunk_targets]:
        if target.query_id not in known_query_ids:
            raise ValueError(f"target references unknown query {target.query_id}")
        if target.document_key.casefold() not in document_by_key:
            raise ValueError(f"target references unknown document_key {target.document_key}")
        if target.target_id in seen_target_ids:
            raise ValueError(f"duplicate target_id {target.target_id}")
        seen_target_ids.add(target.target_id)

        if isinstance(target, RetrievalTarget):
            query_document = (target.query_id, target.document_key.casefold())
            if query_document in seen_query_documents:
                raise ValueError(
                    f"duplicate judgment for {target.query_id} and {target.document_key}"
                )
            seen_query_documents.add(query_document)
            if target.relevance > 0:
                positive_counts[target.query_id] += 1

    positive_document_pairs = {
        (target.query_id, target.document_key.casefold())
        for target in targets
        if target.relevance > 0
    }
    for target in chunk_targets:
        pair = (target.query_id, target.document_key.casefold())
        if target.relevance > 0 and pair not in positive_document_pairs:
            raise ValueError(
                f"positive chunk target {target.target_id} lacks a positive document qrel"
            )

    for query in queries:
        if query.answerable != (positive_counts[query.query_id] > 0):
            raise ValueError(f"{query.query_id} answerability disagrees with its positive targets")
    return queries, targets, chunk_targets, document_by_key


def _docs_support_document_key(conn: sqlite3.Connection) -> bool:
    return any(row["name"] == "document_key" for row in conn.execute("PRAGMA table_info(docs)"))


def _matching_documents(conn: sqlite3.Connection, document: RetrievalDocument) -> list[sqlite3.Row]:
    matches: list[sqlite3.Row] = []
    if _docs_support_document_key(conn):
        matches = conn.execute(
            "SELECT doc_id, source_path FROM docs WHERE lower(document_key) = lower(?)",
            (document.document_key,),
        ).fetchall()

    if not matches:
        target_path = document.source_path.casefold()
        for row in conn.execute("SELECT doc_id, source_path FROM docs"):
            try:
                indexed_path = _canonical_relative_path(str(row["source_path"])).casefold()
            except ValueError:
                continue
            if indexed_path == target_path:
                matches.append(row)
    return matches


def _resolve_target(
    conn: sqlite3.Connection,
    target: RetrievalTarget,
    document: RetrievalDocument,
    document_matches: Sequence[sqlite3.Row] | None = None,
) -> ResolvedTarget:
    matches = (
        list(document_matches)
        if document_matches is not None
        else _matching_documents(conn, document)
    )

    if not matches:
        return ResolvedTarget(
            target=target,
            document=document,
            resolution_state=TargetResolution.unavailable_document,
            doc_id=None,
            diagnostic=f"document {document.source_path} not found in index",
        )
    if len(matches) > 1:
        return ResolvedTarget(
            target=target,
            document=document,
            resolution_state=TargetResolution.ambiguous_document,
            doc_id=None,
            diagnostic=f"{len(matches)} indexed documents match {document.source_path}",
        )
    return ResolvedTarget(
        target=target,
        document=document,
        resolution_state=TargetResolution.resolved,
        doc_id=str(matches[0]["doc_id"]),
        diagnostic=None,
    )


def _resolve_chunk_target(
    conn: sqlite3.Connection,
    target: RetrievalChunkTarget,
    document: RetrievalDocument,
    document_matches: Sequence[sqlite3.Row] | None = None,
    document_chunks: Sequence[ResolvableChunk] | None = None,
) -> ResolvedChunkTarget:
    documents = (
        list(document_matches)
        if document_matches is not None
        else _matching_documents(conn, document)
    )
    if not documents:
        return ResolvedChunkTarget(
            target=target,
            document=document,
            resolution_state=TargetResolution.unavailable_document,
            doc_id=None,
            chunk_ids=(),
            diagnostic=f"document {document.source_path} not found in index",
        )
    if len(documents) > 1:
        return ResolvedChunkTarget(
            target=target,
            document=document,
            resolution_state=TargetResolution.ambiguous_document,
            doc_id=None,
            chunk_ids=(),
            diagnostic=f"{len(documents)} indexed documents match {document.source_path}",
        )

    doc_id = str(documents[0]["doc_id"])
    heading = _normalize_text(target.heading_path)
    required_terms = tuple(_normalize_text(term) for term in target.required_terms)
    matches: list[str] = []
    chunks = _load_resolvable_chunks(conn, doc_id) if document_chunks is None else document_chunks
    for chunk in chunks:
        if chunk.heading_path != heading:
            continue
        if chunk.category != target.chunk_category.casefold():
            continue
        if not all(term in chunk.text for term in required_terms):
            continue
        matches.append(chunk.chunk_id)

    if not matches:
        return ResolvedChunkTarget(
            target=target,
            document=document,
            resolution_state=TargetResolution.unavailable_evidence,
            doc_id=doc_id,
            chunk_ids=(),
            diagnostic=(
                f"no chunk in {document.source_path} matches heading, category, and required terms"
            ),
        )
    if len(matches) > 1 and target.match_policy == "exactly_one":
        return ResolvedChunkTarget(
            target=target,
            document=document,
            resolution_state=TargetResolution.ambiguous_chunk,
            doc_id=doc_id,
            chunk_ids=(),
            diagnostic=f"{len(matches)} chunks match target {target.target_id}",
        )
    return ResolvedChunkTarget(
        target=target,
        document=document,
        resolution_state=TargetResolution.resolved,
        doc_id=doc_id,
        chunk_ids=tuple(matches),
        diagnostic=None,
    )


def precision_at_k(grades: Sequence[int], k: int) -> float:
    _validate_k(k)
    return sum(grade > 0 for grade in grades[:k]) / k


def recall_at_k(grades: Sequence[int], relevant_total: int, k: int) -> float | None:
    _validate_k(k)
    if relevant_total == 0:
        return None
    return sum(grade > 0 for grade in grades[:k]) / relevant_total


def ndcg_at_k(grades: Sequence[int], ideal_grades: Sequence[int], k: int) -> float | None:
    _validate_k(k)

    def dcg(values: Sequence[int]) -> float:
        return sum(
            ((2**grade) - 1) / math.log2(rank + 1) for rank, grade in enumerate(values[:k], start=1)
        )

    ideal = dcg(sorted(ideal_grades, reverse=True))
    if ideal == 0:
        return None
    return dcg(grades) / ideal


def reciprocal_rank(grades: Sequence[int]) -> float:
    for rank, grade in enumerate(grades, start=1):
        if grade > 0:
            return 1 / rank
    return 0.0


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("K must be at least 1")


def ranked_metrics(
    ranked_ids: Sequence[str],
    relevance_by_id: Mapping[str, int],
    k_values: Sequence[int],
    *,
    positives_only: bool = False,
) -> dict[str, float | bool | None]:
    """Calculate standard ranking metrics for one ranked result list.

    ``positives_only`` marks judgments that name the relevant items but have not
    swept the rest of the corpus. Recall, hit, and MRR stay valid there — a known
    positive either appears in the ranking or it does not. Precision and nDCG do
    not, because both must assume every unjudged result is irrelevant, and at
    40,000 documents that assumption is unearned. They are reported as ``None``
    rather than as a number that would read as measured.
    """
    grades = [relevance_by_id.get(item_id, 0) for item_id in ranked_ids]
    ideal_grades = list(relevance_by_id.values())
    relevant_total = sum(grade > 0 for grade in ideal_grades)
    metrics: dict[str, float | bool | None] = {
        "mrr": reciprocal_rank(grades),
        "relevant_total": float(relevant_total),
    }
    for k in k_values:
        metrics[f"precision@{k}"] = None if positives_only else precision_at_k(grades, k)
        metrics[f"recall@{k}"] = recall_at_k(grades, relevant_total, k)
        metrics[f"ndcg@{k}"] = None if positives_only else ndcg_at_k(grades, ideal_grades, k)
        metrics[f"hit@{k}"] = any(grade > 0 for grade in grades[:k])
    return metrics


def _empty_metrics(k_values: Sequence[int]) -> dict[str, None]:
    names = ["mrr"]
    for k in k_values:
        names.extend((f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"hit@{k}"))
    return dict.fromkeys(names)


def _macro_average(
    results: Sequence[Mapping[str, Any]], level: str, k_values: Sequence[int]
) -> dict[str, float | None]:
    names = list(_empty_metrics(k_values))
    summary: dict[str, float | None] = {}
    for name in names:
        values: list[float] = []
        for result in results:
            if not result["answerable"] or result["metrics"] is None:
                continue
            value = result["metrics"][level][name]
            if value is not None:
                values.append(float(value))
        summary[name] = sum(values) / len(values) if values else None
    return summary


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _get_index_version(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT value FROM index_metadata WHERE key = 'index_version'").fetchone()
    if row is None:
        raise ValueError("RAG index is missing index_version")
    return str(row[0])


def _get_chunker_version(conn: sqlite3.Connection) -> str:
    """Read the chunker that built this index.

    Benchmark numbers are only comparable within one chunker version, so the
    report records the version the index was actually built with rather than
    whatever the current source declares.
    """
    row = conn.execute("SELECT value FROM index_metadata WHERE key = 'chunker_version'").fetchone()
    if row is None:
        raise ValueError("RAG index is missing chunker_version")
    return str(row[0])


def _fts_retrieve(
    conn: sqlite3.Connection, query: str, top_k: int
) -> tuple[list[RankedChunk], float]:
    started = perf_counter()
    raw_results = search_ranked_chunks(conn.cursor(), query=query, top_k=top_k)
    ranked: list[RankedChunk] = []
    for result in raw_results:
        chunk_id = str(result["chunk_id"])
        row = conn.execute(
            """
            SELECT c.doc_id, c.heading_path, d.source_path
            FROM chunks AS c
            JOIN docs AS d ON d.doc_id = c.doc_id
            WHERE c.chunk_id = ?
            """,
            (chunk_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"retriever returned missing chunk {chunk_id}")
        score = result.get("score")
        ranked.append(
            RankedChunk(
                chunk_id=chunk_id,
                doc_id=str(row["doc_id"]),
                source_path=str(row["source_path"]),
                heading_path=str(row["heading_path"]),
                score=float(score) if isinstance(score, (int, float)) else None,
            )
        )
    return ranked, (perf_counter() - started) * 1000


def _query_metadata(query: RetrievalQuery) -> dict[str, Any]:
    return {
        "query_id": query.query_id,
        "query": query.query,
        "intent": query.intent,
        "category": query.intent,
        "failure_bucket": query.failure_bucket,
        "answerable": query.answerable,
        "expected_answer": query.expected_answer,
        "answerability_rationale": query.answerability_rationale,
        "difficulty": query.difficulty,
        "tags": list(query.tags),
        "split": query.split,
        "judgment_scope": query.judgment_scope.value,
    }


def run_retrieval_evaluation(
    *,
    db_path: Path,
    queries_path: Path,
    qrels_path: Path,
    chunk_qrels_path: Path,
    catalog_path: Path,
    retriever: str = "fts",
    k_values: Sequence[int] = (5, 10),
) -> dict[str, Any]:
    if retriever != "fts":
        raise ValueError(f"unsupported retriever: {retriever}")
    normalized_k = tuple(sorted(set(k_values)))
    if not normalized_k:
        raise ValueError("at least one K value is required")
    for k in normalized_k:
        _validate_k(k)

    queries, targets, chunk_targets, document_by_key = load_retrieval_dataset(
        queries_path, qrels_path, chunk_qrels_path, catalog_path
    )
    targets_by_query: dict[str, list[RetrievalTarget]] = defaultdict(list)
    for target in targets:
        targets_by_query[target.query_id].append(target)
    chunk_targets_by_query: dict[str, list[RetrievalChunkTarget]] = defaultdict(list)
    for target in chunk_targets:
        chunk_targets_by_query[target.query_id].append(target)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        index_version = _get_index_version(conn)
        chunker_version = _get_chunker_version(conn)
        results: list[dict[str, Any]] = []
        resolution_summary: dict[str, int] = defaultdict(int)
        max_k = max(normalized_k)
        document_matches_by_key = {
            key: _matching_documents(conn, document) for key, document in document_by_key.items()
        }
        document_chunks_by_key = _load_resolvable_chunks_by_document(conn, document_matches_by_key)

        for query in queries:
            resolved_targets = [
                _resolve_target(
                    conn,
                    target,
                    document_by_key[target.document_key.casefold()],
                    document_matches_by_key[target.document_key.casefold()],
                )
                for target in targets_by_query[query.query_id]
            ]
            resolved_chunk_targets = [
                _resolve_chunk_target(
                    conn,
                    target,
                    document_by_key[target.document_key.casefold()],
                    document_matches_by_key[target.document_key.casefold()],
                    document_chunks_by_key[target.document_key.casefold()],
                )
                for target in chunk_targets_by_query[query.query_id]
            ]
            unresolved_positive_documents = [
                resolved
                for resolved in resolved_targets
                if resolved.target.relevance > 0
                and resolved.resolution_state != TargetResolution.resolved
            ]
            unresolved_positive_chunks = [
                resolved
                for resolved in resolved_chunk_targets
                if resolved.target.relevance > 0
                and resolved.resolution_state != TargetResolution.resolved
            ]
            if unresolved_positive_documents or unresolved_positive_chunks:
                results.append(
                    {
                        **_query_metadata(query),
                        "resolution_state": "unresolved",
                        "unresolved_targets": [
                            {
                                "level": "document",
                                "target_id": resolved.target.target_id,
                                "document_key": resolved.target.document_key,
                                "source_path": resolved.document.source_path,
                                "state": resolved.resolution_state.value,
                                "diagnostic": resolved.diagnostic,
                            }
                            for resolved in unresolved_positive_documents
                        ]
                        + [
                            {
                                "level": "chunk",
                                "target_id": resolved.target.target_id,
                                "document_key": resolved.target.document_key,
                                "source_path": resolved.document.source_path,
                                "state": resolved.resolution_state.value,
                                "diagnostic": resolved.diagnostic,
                            }
                            for resolved in unresolved_positive_chunks
                        ],
                        "metrics_status": None,
                        "metrics": None,
                        "results": [],
                    }
                )
                for resolved in [
                    *unresolved_positive_documents,
                    *unresolved_positive_chunks,
                ]:
                    resolution_summary[resolved.resolution_state.value] += 1
                continue

            relevance_by_doc_id: dict[str, int] = {}
            judged_doc_ids: set[str] = set()
            for resolved in resolved_targets:
                if resolved.resolution_state != TargetResolution.resolved:
                    continue
                assert resolved.doc_id is not None
                judged_doc_ids.add(resolved.doc_id)
                relevance_by_doc_id[resolved.doc_id] = resolved.target.relevance

            relevance_by_chunk_id: dict[str, int] = {}
            judged_chunk_ids: set[str] = set()
            for resolved in resolved_chunk_targets:
                if resolved.resolution_state != TargetResolution.resolved:
                    continue
                for chunk_id in resolved.chunk_ids:
                    previous = relevance_by_chunk_id.get(chunk_id)
                    if previous is not None and previous != resolved.target.relevance:
                        raise ValueError(
                            f"conflicting relevance grades resolve to chunk {chunk_id}"
                        )
                    judged_chunk_ids.add(chunk_id)
                    relevance_by_chunk_id[chunk_id] = resolved.target.relevance

            ranked, latency_ms = _fts_retrieve(conn, query.query, max_k)
            ranked_doc_ids = list(dict.fromkeys(item.doc_id for item in ranked))
            ranked_chunk_ids = [item.chunk_id for item in ranked]
            unjudged_doc_ids = [doc_id for doc_id in ranked_doc_ids if doc_id not in judged_doc_ids]
            unjudged_chunk_ids = [
                chunk_id for chunk_id in ranked_chunk_ids if chunk_id not in judged_chunk_ids
            ]

            metrics: dict[str, Any] | None = None
            metrics_status: str | None = None
            if query.judgment_scope == JudgmentScope.corpus_complete:
                metrics = {
                    "chunk": ranked_metrics(ranked_chunk_ids, relevance_by_chunk_id, normalized_k),
                    "document": ranked_metrics(ranked_doc_ids, relevance_by_doc_id, normalized_k),
                }
                metrics_status = "complete"
                unjudged_doc_ids = []
                unjudged_chunk_ids = []
            elif not unjudged_doc_ids and not unjudged_chunk_ids:
                metrics = {
                    "chunk": ranked_metrics(ranked_chunk_ids, relevance_by_chunk_id, normalized_k),
                    "document": ranked_metrics(ranked_doc_ids, relevance_by_doc_id, normalized_k),
                }
                metrics_status = "complete"
            elif query.judgment_scope == JudgmentScope.positive_only:
                # Recall and hit survive incomplete judgments; precision and nDCG
                # do not. Reporting nothing at all made these queries invisible
                # while they still counted toward per-bucket query totals.
                metrics = {
                    "chunk": ranked_metrics(
                        ranked_chunk_ids, relevance_by_chunk_id, normalized_k, positives_only=True
                    ),
                    "document": ranked_metrics(
                        ranked_doc_ids, relevance_by_doc_id, normalized_k, positives_only=True
                    ),
                }
                metrics_status = "recall_only"

            retrieved_doc_ids = set(ranked_doc_ids)
            results.append(
                {
                    **_query_metadata(query),
                    "resolution_state": "resolved",
                    "latency_ms": latency_ms,
                    "judgment_complete": not unjudged_doc_ids and not unjudged_chunk_ids,
                    "unjudged_doc_ids": unjudged_doc_ids,
                    "unjudged_chunk_ids": unjudged_chunk_ids,
                    "metrics_status": metrics_status,
                    "metrics": metrics,
                    "results": [
                        {
                            "rank": rank,
                            "chunk_id": item.chunk_id,
                            "doc_id": item.doc_id,
                            "source_path": item.source_path,
                            "heading_path": item.heading_path,
                            "score": item.score,
                            "document_relevance": relevance_by_doc_id.get(item.doc_id),
                            "chunk_relevance": relevance_by_chunk_id.get(item.chunk_id),
                        }
                        for rank, item in enumerate(ranked, start=1)
                    ],
                    "missed_relevant_doc_ids": sorted(
                        doc_id
                        for doc_id, relevance in relevance_by_doc_id.items()
                        if relevance > 0 and doc_id not in retrieved_doc_ids
                    ),
                    "missed_relevant_chunk_ids": sorted(
                        chunk_id
                        for chunk_id, relevance in relevance_by_chunk_id.items()
                        if relevance > 0 and chunk_id not in set(ranked_chunk_ids)
                    ),
                    "no_answer_diagnostic": (
                        {
                            "retrieved_count": len(ranked),
                            "top_score": ranked[0].score if ranked else None,
                            "note": (
                                "Retrieval has no abstention threshold; this records "
                                "false-match scores but does not claim no-answer accuracy."
                            ),
                        }
                        if not query.answerable
                        else None
                    ),
                }
            )
    finally:
        conn.close()

    resolved_results = [r for r in results if r["resolution_state"] == "resolved"]
    latency_values = [float(result["latency_ms"]) for result in resolved_results]
    unjudged_document_total = sum(len(result["unjudged_doc_ids"]) for result in resolved_results)
    unjudged_chunk_total = sum(len(result["unjudged_chunk_ids"]) for result in resolved_results)
    intents = sorted({str(result["intent"]) for result in results})
    failure_buckets = sorted({str(result["failure_bucket"]) for result in results})

    return {
        "suite": "retrieval",
        "evaluation_levels": ["document", "chunk"],
        "generated_at": datetime.now(UTC).isoformat(),
        "retriever": retriever,
        "index_version": index_version,
        "chunker_version": chunker_version,
        "k_values": list(normalized_k),
        "query_count": len(results),
        "resolved_query_count": len(resolved_results),
        "unresolved_query_count": len(results) - len(resolved_results),
        "resolution_summary": dict(resolution_summary),
        "answerable_query_count": sum(bool(r["answerable"]) for r in resolved_results),
        "unanswerable_query_count": sum(not bool(r["answerable"]) for r in resolved_results),
        "judgment_complete": (unjudged_document_total == 0 and unjudged_chunk_total == 0),
        "unjudged_total": unjudged_document_total + unjudged_chunk_total,
        "unjudged_document_total": unjudged_document_total,
        "unjudged_chunk_total": unjudged_chunk_total,
        "metrics_note": (
            "Document and chunk metrics use portable semantic qrels resolved to "
            "the active index's runtime IDs."
        ),
        "metrics": {
            "chunk": _macro_average(resolved_results, "chunk", normalized_k),
            "document": _macro_average(resolved_results, "document", normalized_k),
            "latency_ms": {
                "median": _nearest_rank(latency_values, 0.5),
                "p95": _nearest_rank(latency_values, 0.95),
                "max": max(latency_values) if latency_values else None,
            },
            "by_category": {
                intent: {
                    "query_count": sum(r["intent"] == intent for r in resolved_results),
                    "chunk": _macro_average(
                        [r for r in resolved_results if r["intent"] == intent],
                        "chunk",
                        normalized_k,
                    ),
                    "document": _macro_average(
                        [r for r in resolved_results if r["intent"] == intent],
                        "document",
                        normalized_k,
                    ),
                }
                for intent in intents
            },
            "by_failure_bucket": {
                bucket: {
                    "query_count": sum(r["failure_bucket"] == bucket for r in resolved_results),
                    # A query with no metrics contributes to no average. Report it
                    # separately so a bucket total is never read as a sample size.
                    "scored_query_count": sum(
                        r["failure_bucket"] == bucket
                        and r["answerable"]
                        and r["metrics"] is not None
                        for r in resolved_results
                    ),
                    "chunk": _macro_average(
                        [r for r in resolved_results if r["failure_bucket"] == bucket],
                        "chunk",
                        normalized_k,
                    ),
                    "document": _macro_average(
                        [r for r in resolved_results if r["failure_bucket"] == bucket],
                        "document",
                        normalized_k,
                    ),
                }
                for bucket in failure_buckets
            },
        },
        "results": results,
    }
