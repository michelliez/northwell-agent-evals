"""Build and evaluate an exact FAISS baseline without model training."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from retrieval.chunk_models import (
    BaselineEvaluationReport,
    BaselineIndexMetadata,
    BaselineQueryResult,
    ChunkRecord,
    FaissMappingRecord,
    FilteredQueryRecord,
    RankedChunkHit,
    SplitChunkRecord,
)
from retrieval.metadata_extractor import DescriptionStatus, MetadataEmbeddingRecord

from clarity_agent_evals.retrieval_evaluator import (
    JudgmentScope,
    RetrievalTarget,
    load_retrieval_dataset,
    ranked_metrics,
)

LOGGER = logging.getLogger(__name__)
INDEX_VERSION = "pretrained-flatip-v1"
EVALUATION_VERSION = "known-positive-exact-rank-v1"
TABLE_INDEX_VERSION = "pretrained-table-flatip-v1"
TABLE_EVALUATION_VERSION = "gold-document-exact-rank-v1"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LEGACY_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LIMIT = 200
DEFAULT_BATCH_SIZE = 32
DEFAULT_TOP_K = 10
DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
GOLD_K_VALUES = (1, 5, 10)


class TableFaissMappingRecord(BaseModel):
    """Stable mapping from a FAISS position to one extracted table record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vector_position: int = Field(ge=0)
    embedding_id: str
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path: str
    table_name: str
    metadata_chunk_id: str
    embedding_text_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    description_status: DescriptionStatus


class TableBaselineIndexMetadata(BaseModel):
    """Reproducibility contract for the table-level exact FAISS index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_version: str
    source_index_version: str
    extractor_version: str
    model_name: str
    device: str
    index_type: Literal["IndexFlatIP"]
    normalized_embeddings: Literal[True]
    embedding_dimension: int = Field(ge=1)
    source_record_count: int = Field(ge=1)
    indexed_record_count: int = Field(ge=1)
    requested_limit: int | None
    source_records_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    mapping_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class TableBaselineEvaluationReport(BaseModel):
    """Document-level dense results over the reviewed gold benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suite: Literal["retrieval"]
    evaluation_version: str
    retriever: Literal["dense_table_metadata"]
    model_name: str
    device: str
    artifact_version: str
    source_index_version: str
    extractor_version: str
    source_records_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_document_count: int = Field(ge=1)
    query_count: int = Field(ge=1)
    answerable_query_count: int = Field(ge=0)
    unanswerable_query_count: int = Field(ge=0)
    k_values: list[int]
    metrics: dict[str, Any]
    results: list[dict[str, Any]]


class TextEncoder(Protocol):
    """Minimal encoder interface used by production code and lightweight tests."""

    model_name: str
    device_name: str

    def encode(
        self, texts: Sequence[str], *, batch_size: int, is_query: bool = False
    ) -> np.ndarray:
        """Return one finite embedding row per input text.

        ``is_query`` lets asymmetric encoders wrap queries in an instruction while
        embedding documents bare. Symmetric encoders ignore it.
        """
        ...


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for a bounded pretrained-model FAISS evaluation."""

    chunks_path: Path
    queries_path: Path
    output_dir: Path
    model_name: str = DEFAULT_MODEL
    device: str = "auto"
    limit: int | None = DEFAULT_LIMIT
    batch_size: int = DEFAULT_BATCH_SIZE
    top_k: int = DEFAULT_TOP_K
    trust_remote_code: bool = False
    load_in_4bit: bool = False

    def validate(self) -> None:
        """Reject unsafe output placement and nonsensical runtime settings."""
        if self.output_dir in {self.chunks_path, self.queries_path}:
            raise ValueError("output_dir must differ from both input paths")
        if not self.model_name.strip():
            raise ValueError("model_name must not be blank")
        if self.device not in DEVICE_CHOICES:
            raise ValueError(f"device must be one of {DEVICE_CHOICES}")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be at least 1 or None")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")


@dataclass(frozen=True)
class TableBaselineConfig:
    """Configuration for gold evaluation over extracted table metadata."""

    records_path: Path
    benchmark_dir: Path
    output_dir: Path
    model_name: str = DEFAULT_MODEL
    device: str = "auto"
    limit: int | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    top_k: int = DEFAULT_TOP_K
    trust_remote_code: bool = False
    load_in_4bit: bool = False

    def validate(self) -> None:
        resolved_output = self.output_dir.resolve()
        if resolved_output in {self.records_path.resolve(), self.benchmark_dir.resolve()}:
            raise ValueError("output_dir must differ from records_path and benchmark_dir")
        if not self.model_name.strip():
            raise ValueError("model_name must not be blank")
        if self.device not in DEVICE_CHOICES:
            raise ValueError(f"device must be one of {DEVICE_CHOICES}")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be at least 1 or None")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")


class SentenceTransformerEncoder:
    """Load a mean-pooled SentenceTransformer checkpoint without optional utilities.

    Importing the top-level ``sentence_transformers`` package also imports
    scikit-learn and SciPy. For the MiniLM baseline we only need its Transformer
    weights plus the standard attention-mask mean pooling used by the model.
    """

    def __init__(self, model_name: str, device: str, *, trust_remote_code: bool = False) -> None:
        try:
            import torch  # pyright: ignore[reportMissingImports]
            from transformers import (  # pyright: ignore[reportMissingImports]
                AutoModel,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Baseline dependencies are missing. Run `uv sync --all-groups`."
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.device_name = self._resolve_device(torch, device)
        LOGGER.info("Loading %s on %s", model_name, self.device_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self._model.eval()
        self._model.to(self.device_name)

    @staticmethod
    def _resolve_device(torch: object, requested: str) -> str:
        cuda_available = bool(torch.cuda.is_available())  # type: ignore[attr-defined]
        mps_available = bool(torch.backends.mps.is_available())  # type: ignore[attr-defined]
        if requested == "cuda" and not cuda_available:
            raise ValueError("CUDA was requested but is not available")
        if requested == "mps" and not mps_available:
            raise ValueError("MPS was requested but is not available")
        if requested != "auto":
            return requested
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    def encode(
        self, texts: Sequence[str], *, batch_size: int, is_query: bool = False
    ) -> np.ndarray:
        """Embed with attention-mask mean pooling.

        ``is_query`` is accepted and ignored: this model is symmetric, so queries
        and documents share one embedding space with no instruction prefix.
        """
        del is_query
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            LOGGER.info(
                "Embedding %d/%d texts",
                min(start + len(batch), len(texts)),
                len(texts),
            )
            tokenized = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device_name)
            with self._torch.inference_mode():
                hidden = self._model(**tokenized).last_hidden_state
                attention_mask = tokenized["attention_mask"].unsqueeze(-1)
                masked_hidden = hidden * attention_mask
                pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            rows.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(rows, axis=0)


class QwenEmbeddingEncoder:
    """Load a Qwen3-Embedding checkpoint with its required pooling and prompting.

    This cannot reuse ``SentenceTransformerEncoder``. Qwen3-Embedding is a causal
    model trained so that the final token's hidden state carries the sequence
    embedding; mean pooling over all positions would average that summary with
    every partial-context token and measurably degrade retrieval. It is also
    asymmetric: queries are wrapped in an instruction while documents are
    embedded bare, which is why ``encode`` takes ``is_query``.
    """

    DEFAULT_TASK = (
        "Given a question about an Epic Clarity database table or column, "
        "retrieve the data-dictionary passage that answers it"
    )

    def __init__(
        self,
        model_name: str,
        device: str,
        *,
        trust_remote_code: bool = False,
        task_description: str | None = None,
        max_length: int = 512,
        load_in_4bit: bool = False,
    ) -> None:
        try:
            import torch  # pyright: ignore[reportMissingImports]
            from transformers import (  # pyright: ignore[reportMissingImports]
                AutoModel,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Baseline dependencies are missing. Run `uv sync --group genq`."
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.device_name = SentenceTransformerEncoder._resolve_device(torch, device)
        self._task = task_description or self.DEFAULT_TASK
        self._max_length = max_length
        quantization = self._quantization_config(load_in_4bit)
        LOGGER.info(
            "Loading %s on %s (%s)",
            model_name,
            self.device_name,
            "4-bit NF4" if quantization else "fp16",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, padding_side="left"
        )
        self._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            dtype=torch.float16 if self.device_name == "cuda" else torch.float32,
            device_map=self.device_name if quantization is not None else None,
            quantization_config=quantization,
        )
        self._model.eval()
        if quantization is None:
            self._model.to(self.device_name)

    def _quantization_config(self, load_in_4bit: bool) -> Any | None:
        """Build the NF4 config, or None when 4-bit is unwanted or unavailable.

        A 4B encoder is ~8 GB in fp16 -- on a 10 GB card that starves the desktop
        compositor, which shares the GPU. NF4 brings it to ~2.5 GB. Quantizing an
        encoder costs some embedding fidelity, so this is opt-in rather than the
        default: correctness of the vectors matters more here than footprint.
        """
        if not load_in_4bit or self.device_name != "cuda":
            return None
        try:
            import bitsandbytes  # noqa: F401  # pyright: ignore[reportMissingImports]
            from transformers import BitsAndBytesConfig  # pyright: ignore[reportMissingImports]
        except ImportError:
            LOGGER.warning("bitsandbytes is unavailable; loading in fp16 instead of 4-bit")
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self._torch.float16,
        )

    def _format(self, text: str, is_query: bool) -> str:
        if not is_query:
            return text
        return f"Instruct: {self._task}\nQuery:{text}"

    def _pool(self, hidden: Any, attention_mask: Any) -> Any:
        """Take the last non-padding token's hidden state for each row.

        The tokenizer is configured for left padding, so the final column is
        always a real token and the fast path applies. The masked branch remains
        because a caller supplying its own right-padded tokenizer would otherwise
        silently embed padding.
        """
        torch = self._torch
        if bool(attention_mask[:, -1].min() == 1):  # type: ignore[index]
            return hidden[:, -1]  # type: ignore[index]
        lengths = attention_mask.sum(dim=1) - 1  # type: ignore[attr-defined]
        indexes = torch.arange(hidden.shape[0], device=hidden.device)  # type: ignore[attr-defined]
        return hidden[indexes, lengths]  # type: ignore[index]

    def encode(
        self, texts: Sequence[str], *, batch_size: int, is_query: bool = False
    ) -> np.ndarray:
        """Return one last-token-pooled embedding row per input text."""
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [self._format(text, is_query) for text in texts[start : start + batch_size]]
            LOGGER.info(
                "Embedding %d/%d texts%s",
                min(start + len(batch), len(texts)),
                len(texts),
                " (queries)" if is_query else "",
            )
            tokenized = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self.device_name)
            with self._torch.inference_mode():
                hidden = self._model(**tokenized).last_hidden_state
                pooled = self._pool(hidden, tokenized["attention_mask"])
            rows.append(pooled.detach().cpu().float().numpy().astype(np.float32))
        return np.concatenate(rows, axis=0)


def build_encoder(
    model_name: str,
    device: str,
    *,
    trust_remote_code: bool = False,
    load_in_4bit: bool = False,
) -> TextEncoder:
    """Pick the encoder whose pooling matches the checkpoint.

    Selection is by model name because the pooling strategy is a property of how
    the checkpoint was trained, not something the caller should have to restate.
    Loading a Qwen3-Embedding model through the mean-pooling path produces
    embeddings that look valid and retrieve poorly, which is the worst failure
    mode available here.
    """
    if "qwen3-embedding" in model_name.casefold():
        return QwenEmbeddingEncoder(
            model_name,
            device,
            trust_remote_code=trust_remote_code,
            load_in_4bit=load_in_4bit,
        )
    if load_in_4bit:
        LOGGER.warning("load_in_4bit is ignored for %s; it is not a Qwen3 encoder", model_name)
    return SentenceTransformerEncoder(model_name, device, trust_remote_code=trust_remote_code)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _load_jsonl[T: BaseModel](path: Path, model: type[T], label: str) -> tuple[list[T], str]:
    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} JSONL does not exist: {path}") from exc
    if not raw_bytes.strip():
        raise ValueError(f"{label} JSONL is empty: {path}")
    records: list[T] = []
    for line_number, line in enumerate(raw_bytes.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        try:
            records.append(model.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{path}:{line_number}: invalid {label} record: {exc}") from exc
    return records, _sha256_bytes(raw_bytes)


def _load_chunk_jsonl(path: Path) -> tuple[list[ChunkRecord], str]:
    """Load either Stage 1 chunks or Stage 2 leakage-safe split chunks.

    The split assignment is dataset metadata rather than encoder input, so the
    baseline normalizes both formats to the shared ``ChunkRecord`` contract.
    """
    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"chunk JSONL does not exist: {path}") from exc
    if not raw_bytes.strip():
        raise ValueError(f"chunk JSONL is empty: {path}")
    records: list[ChunkRecord] = []
    for line_number, line in enumerate(raw_bytes.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        try:
            records.append(ChunkRecord.model_validate_json(line))
            continue
        except ValidationError:
            pass
        try:
            split_record = SplitChunkRecord.model_validate_json(line)
            records.append(
                ChunkRecord.model_validate(
                    split_record.model_dump(exclude={"split", "split_version"})
                )
            )
        except ValidationError as exc:
            raise ValueError(f"{path}:{line_number}: invalid chunk record: {exc}") from exc
    return records, _sha256_bytes(raw_bytes)


def _normalize_embeddings(embeddings: np.ndarray, *, expected_rows: int) -> np.ndarray:
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Encoder returned shape {embeddings.shape}; expected ({expected_rows}, dimension)"
        )
    if embeddings.shape[1] < 1:
        raise ValueError("Encoder returned zero-dimensional embeddings")
    if not np.isfinite(embeddings).all():
        raise ValueError("Encoder returned non-finite embedding values")
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Encoder returned a zero-length embedding")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


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


def _write_faiss_atomic(path: Path, index: Any) -> None:
    import faiss  # pyright: ignore[reportMissingImports]

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".faiss"
    )
    os.close(descriptor)
    try:
        faiss.write_index(index, temporary_name)
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _validate_inputs(
    chunks: list[ChunkRecord],
    queries: list[FilteredQueryRecord],
) -> dict[str, ChunkRecord]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunk_by_id) != len(chunks):
        raise ValueError("Duplicate chunk IDs found")
    query_ids = Counter(query.query_id for query in queries)
    duplicates = [query_id for query_id, count in query_ids.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate query IDs found: {', '.join(sorted(duplicates)[:5])}")
    for query in queries:
        chunk = chunk_by_id.get(query.relevant_chunk_id)
        if chunk is None:
            raise ValueError(
                f"Query {query.query_id} references missing chunk {query.relevant_chunk_id}"
            )
        if (
            query.relevant_text_hash != chunk.text_hash
            or query.source_file != chunk.source_file
            or query.chunk_type != chunk.chunk_type
        ):
            raise ValueError(f"Query {query.query_id} has inconsistent chunk provenance")
    return chunk_by_id


def _required_metric(
    metrics: dict[str, float | bool | None],
    name: str,
) -> float:
    """Return a numeric metric that must exist for known-positive evaluation."""
    value = metrics[name]
    if value is None:
        raise RuntimeError(f"Shared ranking calculator returned no value for {name}")
    return float(value)


def run_baseline(
    config: BaselineConfig,
    *,
    encoder: TextEncoder | None = None,
) -> BaselineEvaluationReport:
    """Embed chunks, build exact FAISS, and evaluate known-positive query ranks."""
    config.validate()
    all_chunks, chunks_hash = _load_chunk_jsonl(config.chunks_path)
    queries, queries_hash = _load_jsonl(config.queries_path, FilteredQueryRecord, "retained query")
    _validate_inputs(all_chunks, queries)
    selected = all_chunks[: config.limit] if config.limit is not None else all_chunks
    selected_ids = {chunk.chunk_id for chunk in selected}
    missing_positives = sorted({query.relevant_chunk_id for query in queries} - selected_ids)
    if missing_positives:
        raise ValueError(
            "The bounded index excludes positive chunks required by evaluation: "
            + ", ".join(missing_positives[:5])
        )

    # Load FAISS before PyTorch initializes MPS. On macOS, loading FAISS's
    # native runtime after MPS inference can terminate the process outside
    # Python's exception machinery even though each library works separately.
    LOGGER.info("Loading FAISS before initializing the embedding device")
    import faiss  # pyright: ignore[reportMissingImports]

    active_encoder = encoder or build_encoder(
        config.model_name,
        config.device,
        trust_remote_code=config.trust_remote_code,
        load_in_4bit=config.load_in_4bit,
    )
    chunk_vectors = _normalize_embeddings(
        active_encoder.encode([chunk.text for chunk in selected], batch_size=config.batch_size),
        expected_rows=len(selected),
    )
    query_vectors = _normalize_embeddings(
        active_encoder.encode(
            [query.query for query in queries], batch_size=config.batch_size, is_query=True
        ),
        expected_rows=len(queries),
    )

    dimension = int(chunk_vectors.shape[1])
    if query_vectors.shape[1] != dimension:
        raise ValueError("Query and chunk embedding dimensions do not match")
    index = faiss.IndexFlatIP(dimension)
    index.add(chunk_vectors)
    if index.ntotal != len(selected):
        raise RuntimeError("FAISS index size does not match selected chunk count")

    mappings = [
        FaissMappingRecord(
            vector_position=position,
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
            table_name=chunk.table_name,
            column_name=chunk.column_name,
            chunk_type=chunk.chunk_type,
            text_hash=chunk.text_hash,
        )
        for position, chunk in enumerate(selected)
    ]
    mapping_lines = [
        json.dumps(mapping.model_dump(), sort_keys=True, ensure_ascii=False) for mapping in mappings
    ]
    mapping_hash = _sha256_text("\n".join(mapping_lines) + "\n")
    metadata = BaselineIndexMetadata(
        index_version=INDEX_VERSION,
        model_name=active_encoder.model_name,
        device=active_encoder.device_name,
        index_type="IndexFlatIP",
        normalized_embeddings=True,
        embedding_dimension=dimension,
        source_chunk_count=len(all_chunks),
        indexed_chunk_count=len(selected),
        requested_limit=config.limit,
        source_chunks_hash=chunks_hash,
        mapping_hash=mapping_hash,
    )

    scores, positions = index.search(query_vectors, len(selected))
    mapping_by_position = {mapping.vector_position: mapping for mapping in mappings}
    query_results: list[BaselineQueryResult] = []
    inspection_k = min(config.top_k, len(selected))
    for query_index, query in enumerate(queries):
        ranked_positions = [int(value) for value in positions[query_index]]
        positive_position = next(
            mapping.vector_position
            for mapping in mappings
            if mapping.chunk_id == query.relevant_chunk_id
        )
        positive_rank = ranked_positions.index(positive_position) + 1
        ranked_chunk_ids = [mapping_by_position[position].chunk_id for position in ranked_positions]
        metrics = ranked_metrics(
            ranked_chunk_ids,
            {query.relevant_chunk_id: 1},
            (1, 5, 10),
        )
        top_hits: list[RankedChunkHit] = []
        for rank, vector_position in enumerate(ranked_positions[:inspection_k], start=1):
            mapping = mapping_by_position[vector_position]
            top_hits.append(
                RankedChunkHit(
                    rank=rank,
                    score=float(scores[query_index, rank - 1]),
                    chunk_id=mapping.chunk_id,
                    source_file=mapping.source_file,
                    table_name=mapping.table_name,
                    column_name=mapping.column_name,
                    chunk_type=mapping.chunk_type,
                )
            )
        query_results.append(
            BaselineQueryResult(
                query_id=query.query_id,
                query=query.query,
                relevant_chunk_id=query.relevant_chunk_id,
                positive_rank=positive_rank,
                reciprocal_rank=_required_metric(metrics, "mrr"),
                precision_at_1=_required_metric(metrics, "precision@1"),
                precision_at_5=_required_metric(metrics, "precision@5"),
                precision_at_10=_required_metric(metrics, "precision@10"),
                recall_at_1=_required_metric(metrics, "recall@1"),
                recall_at_5=_required_metric(metrics, "recall@5"),
                recall_at_10=_required_metric(metrics, "recall@10"),
                hit_at_1=_required_metric(metrics, "hit@1"),
                hit_at_5=_required_metric(metrics, "hit@5"),
                hit_at_10=_required_metric(metrics, "hit@10"),
                ndcg_at_10=_required_metric(metrics, "ndcg@10"),
                top_hits=top_hits,
            )
        )

    count = len(query_results)
    report = BaselineEvaluationReport(
        evaluation_version=EVALUATION_VERSION,
        model_name=active_encoder.model_name,
        device=active_encoder.device_name,
        index_version=INDEX_VERSION,
        candidate_chunk_count=len(selected),
        evaluated_query_count=count,
        precision_at_1=sum(result.precision_at_1 for result in query_results) / count,
        precision_at_5=sum(result.precision_at_5 for result in query_results) / count,
        precision_at_10=sum(result.precision_at_10 for result in query_results) / count,
        recall_at_1=sum(result.recall_at_1 for result in query_results) / count,
        recall_at_5=sum(result.recall_at_5 for result in query_results) / count,
        recall_at_10=sum(result.recall_at_10 for result in query_results) / count,
        hit_at_1=sum(result.hit_at_1 for result in query_results) / count,
        hit_at_5=sum(result.hit_at_5 for result in query_results) / count,
        hit_at_10=sum(result.hit_at_10 for result in query_results) / count,
        mrr=sum(result.reciprocal_rank for result in query_results) / count,
        ndcg_at_10=sum(result.ndcg_at_10 for result in query_results) / count,
        retained_queries_hash=queries_hash,
        query_results=query_results,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_faiss_atomic(config.output_dir / "corpus.faiss", index)
    _write_atomic(config.output_dir / "chunk_mapping.jsonl", mapping_lines)
    _write_atomic(
        config.output_dir / "index_metadata.json",
        [json.dumps(metadata.model_dump(), indent=2, sort_keys=True)],
    )
    _write_atomic(
        config.output_dir / "smoke_evaluation.json",
        [json.dumps(report.model_dump(), indent=2, sort_keys=True)],
    )
    return report


def _canonical_source_path(value: str) -> str:
    return value.replace("\\", "/").strip("/").casefold()


def _benchmark_paths(benchmark_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        benchmark_dir / "retrieval_queries.jsonl",
        benchmark_dir / "retrieval_qrels.jsonl",
        benchmark_dir / "retrieval_chunk_qrels.jsonl",
        benchmark_dir / "retrieval_catalog.jsonl",
    )


def _benchmark_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        try:
            content = path.read_bytes()
        except FileNotFoundError as exc:
            raise ValueError(f"Benchmark file does not exist: {path}") from exc
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_table_records(
    records: list[MetadataEmbeddingRecord],
) -> tuple[str, str, dict[str, MetadataEmbeddingRecord]]:
    checks = {
        "embedding IDs": [record.embedding_id for record in records],
        "document IDs": [record.document_id for record in records],
        "source paths": [_canonical_source_path(record.source_path) for record in records],
        "table names": [record.table_name.casefold() for record in records],
    }
    for label, values in checks.items():
        duplicates = [value for value, count in Counter(values).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate table metadata {label}: {', '.join(duplicates[:5])}")
    for record in records:
        if _sha256_text(record.embedding_text) != record.embedding_text_hash:
            raise ValueError(f"Embedding text hash disagrees for {record.embedding_id}")

    index_versions = {record.index_version for record in records}
    extractor_versions = {record.extractor_version for record in records}
    if len(index_versions) != 1:
        raise ValueError("Table metadata records contain multiple source index versions")
    if len(extractor_versions) != 1:
        raise ValueError("Table metadata records contain multiple extractor versions")
    by_path = {_canonical_source_path(record.source_path): record for record in records}
    return next(iter(index_versions)), next(iter(extractor_versions)), by_path


def _macro_document_metrics(
    results: Sequence[dict[str, Any]],
    *,
    failure_bucket: str | None = None,
) -> dict[str, float | None]:
    names = ["mrr"] + [
        f"{metric}@{k}" for k in GOLD_K_VALUES for metric in ("precision", "recall", "ndcg", "hit")
    ]
    selected = [
        result
        for result in results
        if result["answerable"]
        and result["metrics"] is not None
        and (failure_bucket is None or result["failure_bucket"] == failure_bucket)
    ]
    summary: dict[str, float | None] = {}
    for name in names:
        values = [
            float(result["metrics"][name])
            for result in selected
            if result["metrics"][name] is not None
        ]
        summary[name] = sum(values) / len(values) if values else None
    return summary


def run_table_baseline(
    config: TableBaselineConfig,
    *,
    encoder: TextEncoder | None = None,
) -> TableBaselineEvaluationReport:
    """Embed extracted table metadata and evaluate the reviewed document qrels."""
    config.validate()
    all_records, records_hash = _load_jsonl(
        config.records_path,
        MetadataEmbeddingRecord,
        "table metadata",
    )
    source_index_version, extractor_version, _all_records_by_path = _validate_table_records(
        all_records
    )
    selected = all_records[: config.limit] if config.limit is not None else all_records
    selected_paths = {_canonical_source_path(record.source_path): record for record in selected}

    queries_path, qrels_path, chunk_qrels_path, catalog_path = _benchmark_paths(
        config.benchmark_dir
    )
    queries, targets, _chunk_targets, documents_by_key = load_retrieval_dataset(
        queries_path,
        qrels_path,
        chunk_qrels_path,
        catalog_path,
    )
    benchmark_hash = _benchmark_hash((queries_path, qrels_path, chunk_qrels_path, catalog_path))
    targets_by_query: dict[str, list[RetrievalTarget]] = defaultdict(list)
    for target in targets:
        targets_by_query[target.query_id].append(target)

    missing_positive_paths: set[str] = set()
    for target in targets:
        if target.relevance <= 0:
            continue
        document = documents_by_key[target.document_key.casefold()]
        if _canonical_source_path(document.source_path) not in selected_paths:
            missing_positive_paths.add(document.source_path)
    if missing_positive_paths:
        raise ValueError(
            "The bounded table index excludes positive gold documents: "
            + ", ".join(sorted(missing_positive_paths)[:5])
        )

    active_encoder = encoder or build_encoder(
        config.model_name,
        config.device,
        trust_remote_code=config.trust_remote_code,
        load_in_4bit=config.load_in_4bit,
    )
    table_vectors = _normalize_embeddings(
        active_encoder.encode(
            [record.embedding_text for record in selected],
            batch_size=config.batch_size,
        ),
        expected_rows=len(selected),
    )
    query_vectors = _normalize_embeddings(
        active_encoder.encode(
            [query.query for query in queries], batch_size=config.batch_size, is_query=True
        ),
        expected_rows=len(queries),
    )
    if query_vectors.shape[1] != table_vectors.shape[1]:
        raise ValueError("Query and table embedding dimensions do not match")

    import faiss  # pyright: ignore[reportMissingImports]

    dimension = int(table_vectors.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(table_vectors)
    if index.ntotal != len(selected):
        raise RuntimeError("FAISS index size does not match selected table count")

    mappings = [
        TableFaissMappingRecord(
            vector_position=position,
            embedding_id=record.embedding_id,
            document_id=record.document_id,
            source_path=record.source_path,
            table_name=record.table_name,
            metadata_chunk_id=record.metadata_chunk_id,
            embedding_text_hash=record.embedding_text_hash,
            description_status=record.description_status,
        )
        for position, record in enumerate(selected)
    ]
    mapping_lines = [
        json.dumps(mapping.model_dump(), sort_keys=True, ensure_ascii=False) for mapping in mappings
    ]
    mapping_hash = _sha256_text("\n".join(mapping_lines) + "\n")
    index_metadata = TableBaselineIndexMetadata(
        artifact_version=TABLE_INDEX_VERSION,
        source_index_version=source_index_version,
        extractor_version=extractor_version,
        model_name=active_encoder.model_name,
        device=active_encoder.device_name,
        index_type="IndexFlatIP",
        normalized_embeddings=True,
        embedding_dimension=dimension,
        source_record_count=len(all_records),
        indexed_record_count=len(selected),
        requested_limit=config.limit,
        source_records_hash=records_hash,
        mapping_hash=mapping_hash,
    )

    scores, positions = index.search(query_vectors, len(selected))
    mapping_by_position = {mapping.vector_position: mapping for mapping in mappings}
    inspection_k = min(config.top_k, len(selected))
    results: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        ranked_positions = [int(value) for value in positions[query_index]]
        ranked_document_ids = [
            mapping_by_position[position].document_id for position in ranked_positions
        ]
        relevance_by_document_id: dict[str, int] = {}
        positive_source_paths: list[str] = []
        for target in targets_by_query[query.query_id]:
            document = documents_by_key[target.document_key.casefold()]
            record = selected_paths.get(_canonical_source_path(document.source_path))
            if record is not None:
                relevance_by_document_id[record.document_id] = target.relevance
                if target.relevance > 0:
                    positive_source_paths.append(record.source_path)

        metrics: dict[str, float | bool | None] | None = None
        metrics_status: str | None = None
        if query.answerable:
            positives_only = query.judgment_scope == JudgmentScope.positive_only
            metrics = ranked_metrics(
                ranked_document_ids,
                relevance_by_document_id,
                GOLD_K_VALUES,
                positives_only=positives_only,
            )
            metrics_status = "recall_only" if positives_only else "complete"

        top_hits = []
        for rank, vector_position in enumerate(ranked_positions[:inspection_k], start=1):
            mapping = mapping_by_position[vector_position]
            top_hits.append(
                {
                    "rank": rank,
                    "score": float(scores[query_index, rank - 1]),
                    "embedding_id": mapping.embedding_id,
                    "document_id": mapping.document_id,
                    "source_path": mapping.source_path,
                    "table_name": mapping.table_name,
                    "description_status": mapping.description_status,
                    "relevance": relevance_by_document_id.get(mapping.document_id),
                }
            )
        positive_ranks = [
            rank
            for rank, document_id in enumerate(ranked_document_ids, start=1)
            if relevance_by_document_id.get(document_id, 0) > 0
        ]
        results.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "intent": query.intent,
                "failure_bucket": query.failure_bucket,
                "answerable": query.answerable,
                "judgment_scope": query.judgment_scope.value,
                "metrics_status": metrics_status,
                "metrics": metrics,
                "positive_source_paths": sorted(positive_source_paths),
                "positive_ranks": positive_ranks,
                "top_hits": top_hits,
                "no_answer_diagnostic": (
                    {
                        "top_score": top_hits[0]["score"] if top_hits else None,
                        "note": "Dense retrieval has no abstention threshold.",
                    }
                    if not query.answerable
                    else None
                ),
            }
        )

    failure_buckets = sorted({query.failure_bucket for query in queries})
    report = TableBaselineEvaluationReport(
        suite="retrieval",
        evaluation_version=TABLE_EVALUATION_VERSION,
        retriever="dense_table_metadata",
        model_name=active_encoder.model_name,
        device=active_encoder.device_name,
        artifact_version=TABLE_INDEX_VERSION,
        source_index_version=source_index_version,
        extractor_version=extractor_version,
        source_records_hash=records_hash,
        benchmark_hash=benchmark_hash,
        candidate_document_count=len(selected),
        query_count=len(queries),
        answerable_query_count=sum(query.answerable for query in queries),
        unanswerable_query_count=sum(not query.answerable for query in queries),
        k_values=list(GOLD_K_VALUES),
        metrics={
            "document": _macro_document_metrics(results),
            "by_failure_bucket": {
                bucket: {
                    "query_count": sum(result["failure_bucket"] == bucket for result in results),
                    "scored_query_count": sum(
                        result["failure_bucket"] == bucket
                        and result["answerable"]
                        and result["metrics"] is not None
                        for result in results
                    ),
                    "document": _macro_document_metrics(results, failure_bucket=bucket),
                }
                for bucket in failure_buckets
            },
        },
        results=results,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_faiss_atomic(config.output_dir / "tables.faiss", index)
    _write_atomic(config.output_dir / "table_mapping.jsonl", mapping_lines)
    _write_atomic(
        config.output_dir / "index_metadata.json",
        [json.dumps(index_metadata.model_dump(), indent=2, sort_keys=True)],
    )
    _write_atomic(
        config.output_dir / "gold_evaluation.json",
        [json.dumps(report.model_dump(), indent=2, sort_keys=True)],
    )
    return report


def main() -> None:
    """Build either the synthetic-chunk or reviewed table-metadata baseline."""
    parser = argparse.ArgumentParser(
        description="Build exact pretrained FAISS and evaluate synthetic or gold queries."
    )
    parser.add_argument(
        "corpus_path",
        type=Path,
        help="Synthetic chunk JSONL or metadata-extractor table record JSONL.",
    )
    evaluation = parser.add_mutually_exclusive_group(required=True)
    evaluation.add_argument(
        "--queries",
        type=Path,
        help="Reviewed synthetic-query JSONL (synthetic chunk mode).",
    )
    evaluation.add_argument(
        "--benchmark-dir",
        type=Path,
        help="Directory containing the reviewed retrieval gold JSONL files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional corpus bound (synthetic mode defaults to 200; gold mode to all tables).",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help=(
            "Load a Qwen3 encoder in 4-bit NF4. A 4B model is ~8 GB in fp16, which "
            "starves the desktop compositor on a 10 GB card; NF4 brings it to ~2.5 GB."
        ),
    )
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
        if args.benchmark_dir is not None:
            table_report = run_table_baseline(
                TableBaselineConfig(
                    records_path=args.corpus_path,
                    benchmark_dir=args.benchmark_dir,
                    output_dir=args.output_dir,
                    model_name=args.model,
                    device=args.device,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    top_k=args.top_k,
                    trust_remote_code=args.trust_remote_code,
                    load_in_4bit=args.load_in_4bit,
                )
            )
        else:
            report = run_baseline(
                BaselineConfig(
                    chunks_path=args.corpus_path,
                    queries_path=args.queries,
                    output_dir=args.output_dir,
                    model_name=args.model,
                    device=args.device,
                    limit=args.limit if args.limit is not None else DEFAULT_LIMIT,
                    batch_size=args.batch_size,
                    top_k=args.top_k,
                    trust_remote_code=args.trust_remote_code,
                    load_in_4bit=args.load_in_4bit,
                )
            )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    if args.benchmark_dir is not None:
        document_metrics = table_report.metrics["document"]
        print(
            f"Evaluated {table_report.query_count} gold queries against "
            f"{table_report.candidate_document_count} exact FAISS table candidates: "
            f"Hit@1={document_metrics['hit@1']:.3f} "
            f"Hit@5={document_metrics['hit@5']:.3f} "
            f"Hit@10={document_metrics['hit@10']:.3f} "
            f"MRR={document_metrics['mrr']:.3f}"
        )
    else:
        print(
            f"Evaluated {report.evaluated_query_count} query against "
            f"{report.candidate_chunk_count} exact FAISS candidates: "
            f"Hit@1={report.hit_at_1:.3f} Hit@5={report.hit_at_5:.3f} "
            f"Hit@10={report.hit_at_10:.3f} MRR={report.mrr:.3f}"
        )


if __name__ == "__main__":
    main()
