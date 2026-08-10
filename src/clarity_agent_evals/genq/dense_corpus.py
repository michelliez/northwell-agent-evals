"""Build the dense-arm FAISS artifact directly from the runtime SQLite index.

The `agent-harness-genq-baseline` chunk mode is welded to a synthetic-query
evaluation whose corpus uses genq pipeline chunk IDs; those IDs do not exist
in the runtime index, so its artifact can never hydrate at eval or request
time. This builder sources every chunk from the index itself, so
``chunk_mapping.jsonl`` is keyed by runtime chunk IDs by construction, and it
reuses the shared encoder, normalization, and atomic writers so the artifact
format matches what the eval runner's ``--retriever dense``/``hybrid`` arms
and the application's `DENSE_INDEX_DIR` both validate.

Embeds chunk TEXT only — doc2query expansion columns stay out, so the dense
arm measures the encoder rather than the FTS experiment.

The artifact is coupled to the index it was built from: rebuild it after any
index rebuild, and keep it beside the other shared inputs. Expect roughly
2.5 h on a consumer CUDA GPU or a long overnight run on Apple MPS.

Usage (eval repo root, genq group synced):
    uv run agent-harness-genq-dense-corpus \
        --db ../fixtures/rag/index-v6-genq-combined-expansion.sqlite \
        --output-dir ../fixtures/embeddings/dense-corpus-qwen06b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from retrieval.chunk_models import BaselineIndexMetadata, FaissMappingRecord

from clarity_agent_evals.genq.baseline_faiss import (
    _normalize_embeddings,
    _write_atomic,
    _write_faiss_atomic,
    build_encoder,
)

ARTIFACT_VERSION = "pretrained-flatip-v1"


def _chunk_type(category: str, heading_path: str) -> str:
    if category == "metadata":
        return "table_metadata"
    if category == "column_info":
        return "column_definition"
    if category == "table_data":
        if "Primary-Key" in heading_path:
            return "primary_key"
        if "Index-Information" in heading_path:
            return "index_information"
        if "Foreign-Key" in heading_path:
            return "foreign_key"
        return "section"
    return "section"


def load_chunks(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT c.chunk_id, c.category, c.heading_path, c.text, c.text_hash,
               d.source_path
        FROM chunks AS c JOIN docs AS d ON d.doc_id = c.doc_id
        ORDER BY d.doc_id, c.chunk_index
        """
    ).fetchall()
    conn.close()
    chunks = []
    for row in rows:
        text = (row["text"] or "").strip()
        if not text:
            continue
        heading = row["heading_path"] or ""
        segments = [part.strip() for part in heading.split(">")]
        chunks.append(
            {
                "chunk_id": row["chunk_id"],
                "source_file": row["source_path"].replace("\\", "/").rsplit("/", 1)[-1],
                "table_name": segments[0] if segments and segments[0] else "UNKNOWN",
                "column_name": segments[-1] if row["category"] == "column_info" else None,
                "chunk_type": _chunk_type(row["category"], heading),
                "text": text,
                "text_hash": row["text_hash"],
            }
        )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    started = time.monotonic()
    chunks = load_chunks(args.db)
    print(f"{len(chunks):,} non-empty chunks loaded from {args.db.name}", flush=True)

    encoder = build_encoder(args.model, args.device)
    print(f"encoder ready: {encoder.model_name} on {encoder.device_name}", flush=True)

    import faiss  # pyright: ignore[reportMissingImports]

    texts = [chunk["text"] for chunk in chunks]
    vectors = _normalize_embeddings(
        encoder.encode(texts, batch_size=args.batch_size),
        expected_rows=len(chunks),
    )
    print(f"embedded in {(time.monotonic() - started) / 60:.1f} min", flush=True)

    dimension = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    if index.ntotal != len(chunks):
        raise RuntimeError("FAISS index size does not match chunk count")

    mappings = [
        FaissMappingRecord(
            vector_position=position,
            chunk_id=chunk["chunk_id"],
            source_file=chunk["source_file"],
            table_name=chunk["table_name"],
            column_name=chunk["column_name"],
            chunk_type=chunk["chunk_type"],
            text_hash=chunk["text_hash"],
        )
        for position, chunk in enumerate(chunks)
    ]
    mapping_lines = [
        json.dumps(m.model_dump(), sort_keys=True, ensure_ascii=False) for m in mappings
    ]
    mapping_hash = hashlib.sha256(("\n".join(mapping_lines) + "\n").encode()).hexdigest()
    corpus_hash = hashlib.sha256(
        "\n".join(chunk["chunk_id"] + ":" + chunk["text_hash"] for chunk in chunks).encode()
    ).hexdigest()

    metadata = BaselineIndexMetadata(
        index_version=ARTIFACT_VERSION,
        model_name=encoder.model_name,
        device=encoder.device_name,
        index_type="IndexFlatIP",
        normalized_embeddings=True,
        embedding_dimension=dimension,
        source_chunk_count=len(chunks),
        indexed_chunk_count=len(chunks),
        requested_limit=None,
        source_chunks_hash=corpus_hash,
        mapping_hash=mapping_hash,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_faiss_atomic(args.output_dir / "corpus.faiss", index)
    _write_atomic(args.output_dir / "chunk_mapping.jsonl", mapping_lines)
    _write_atomic(
        args.output_dir / "index_metadata.json",
        [metadata.model_dump_json(indent=2)],
    )
    print(
        f"artifact complete: {len(chunks):,} vectors, dim {dimension}, "
        f"{(time.monotonic() - started) / 60:.1f} min total -> {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
