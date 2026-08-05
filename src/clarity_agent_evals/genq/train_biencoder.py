"""Fine-tune a bi-encoder on synthetic (query, passage) pairs.

Loads retained queries from Stage 4 and their positive passages from the
converted chunk JSONL, then trains with ``MultipleNegativesRankingLoss``.
Only the train split is used for training; the validation split provides
development feedback.  Test-split queries are excluded entirely.

The trained model is saved in both sentence-transformers format (for further
training) and raw HuggingFace format (for ``SentenceTransformerEncoder``
inference in the FAISS baseline).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from retrieval.chunk_models import FilteredQueryRecord, SplitChunkRecord

LOGGER = logging.getLogger(__name__)
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_EVALUATION_STEPS = 500
# Qwen3-Embedding declares max_seq_length 32768, which suits long documents and
# is catastrophic here: Clarity passages measure 31 tokens at the median, 106 at
# p95, and 460 at the longest. Attention memory grows with sequence length, so
# the model default made activations dominate and pushed a 10 GB card into
# system RAM over PCIe -- 32 seconds per iteration before it failed outright.
DEFAULT_MAX_SEQ_LENGTH = 512
DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


class TrainingReport(BaseModel):
    """Audit record for one fine-tuning run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_model: str
    output_dir: str
    train_pair_count: int = Field(ge=0)
    validation_pair_count: int = Field(ge=0)
    excluded_query_count: int = Field(ge=0)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    # What the run actually used, not what was asked for: both fall back to full
    # precision off CUDA, and a report that recorded the request would make two
    # runs look identical when their numerics differed.
    dtype: str
    optimizer: str
    max_seq_length: int = Field(ge=1)
    gradient_checkpointing: bool


@dataclass(frozen=True)
class BiEncoderTrainingConfig:
    """Configuration for one fine-tuning run."""

    base_model: str
    retained_queries_path: Path
    chunks_path: Path
    output_dir: Path
    trust_remote_code: bool = False
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    evaluation_steps: int = DEFAULT_EVALUATION_STEPS
    device: str = "auto"
    seed: int = 42
    # A 596M-parameter encoder in fp32 needs 2.2 GB of weights, 2.2 GB of
    # gradients, and 4.4 GB of AdamW moments -- 8.9 GB before a single
    # activation, against 8.9 GB free on a 10 GB card. Lowering batch_size does
    # not help, because none of that is activation memory. bf16 weights and
    # 8-bit moments bring it to roughly 3.4 GB.
    use_bfloat16: bool = True
    use_8bit_optimizer: bool = True
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    # Activations, not parameters, are what actually overflow here: 28 layers at
    # batch 32 hold roughly 17 GB, against 3.4 GB for weights, gradients, and
    # 8-bit moments combined. Recomputing them during the backward pass costs
    # about 30% throughput and removes the dominant term. Shrinking the batch
    # would also work but weakens MultipleNegativesRankingLoss, which draws its
    # negatives from within the batch.
    use_gradient_checkpointing: bool = True

    def validate(self) -> None:
        if not self.base_model.strip():
            raise ValueError("base_model must not be blank")
        if self.device not in DEVICE_CHOICES:
            raise ValueError(f"device must be one of {DEVICE_CHOICES}")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.warmup_ratio < 0 or self.warmup_ratio >= 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.evaluation_steps < 1:
            raise ValueError("evaluation_steps must be at least 1")
        if self.max_seq_length < 1:
            raise ValueError("max_seq_length must be at least 1")


def _resolve_optimizer(torch: Any, want_8bit: bool) -> tuple[Any, str]:
    """Return the optimizer class to train with, and the name to record.

    Falls back to AdamW rather than failing when bitsandbytes is unavailable,
    because a missing optional dependency should cost memory headroom, not the
    run. The chosen name is reported so a later reader can tell which happened.
    """
    if not want_8bit:
        return torch.optim.AdamW, "adamw"
    try:
        from bitsandbytes.optim import AdamW8bit  # pyright: ignore[reportMissingImports]
    except ImportError:
        LOGGER.warning("bitsandbytes is unavailable; using fp32 AdamW moments instead of 8-bit")
        return torch.optim.AdamW, "adamw"
    return AdamW8bit, "adamw8bit"


def load_training_pairs(
    queries_path: Path,
    chunks_path: Path,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], int]:
    """Load (query, passage) pairs split into train and validation sets.

    Returns ``(train_pairs, validation_pairs, excluded_count)`` where each pair
    is ``(query_text, passage_text)``.  Only retained queries from the train and
    validation splits are used.  Test-split queries are excluded entirely.
    """
    chunks_by_id: dict[str, str] = {}
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = SplitChunkRecord.model_validate_json(line)
        chunks_by_id[chunk.chunk_id] = chunk.text

    train_pairs: list[tuple[str, str]] = []
    validation_pairs: list[tuple[str, str]] = []
    excluded = 0
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        query = FilteredQueryRecord.model_validate_json(line)
        passage = chunks_by_id.get(query.relevant_chunk_id)
        if passage is None:
            raise ValueError(
                f"Query {query.query_id} references unknown chunk {query.relevant_chunk_id}"
            )
        if query.split == "train":
            train_pairs.append((query.query, passage))
        elif query.split == "validation":
            validation_pairs.append((query.query, passage))
        else:
            excluded += 1

    if not train_pairs:
        raise ValueError("No train-split queries found — nothing to train on")
    return train_pairs, validation_pairs, excluded


def train_biencoder(config: BiEncoderTrainingConfig) -> TrainingReport:
    """Fine-tune a bi-encoder with MultipleNegativesRankingLoss."""
    config.validate()

    try:
        from sentence_transformers import (  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
            InputExample,
            SentenceTransformer,
            losses,  # pyright: ignore[reportAttributeAccessIssue]
        )
        from sentence_transformers.evaluation import (  # pyright: ignore[reportMissingImports]
            InformationRetrievalEvaluator,
        )
        from torch.utils.data import DataLoader  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "Fine-tuning dependencies are missing. Run `uv sync --group genq`."
        ) from exc

    train_pairs, validation_pairs, excluded = load_training_pairs(
        config.retained_queries_path, config.chunks_path
    )
    LOGGER.info(
        "Loaded %d train pairs, %d validation pairs (%d test-split excluded)",
        len(train_pairs),
        len(validation_pairs),
        excluded,
    )

    import torch  # pyright: ignore[reportMissingImports]

    if config.device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = config.device
    # bf16 rather than fp16: the exponent range matches fp32, so training does
    # not need loss scaling to avoid underflowing gradients. Only on CUDA --
    # CPU bf16 matmul is emulated and slower than the fp32 it replaces.
    use_bfloat16 = config.use_bfloat16 and resolved_device == "cuda"
    model_kwargs: dict[str, Any] = {"dtype": torch.bfloat16} if use_bfloat16 else {}
    model = SentenceTransformer(
        config.base_model,
        device=resolved_device,
        trust_remote_code=config.trust_remote_code,
        model_kwargs=model_kwargs,
    )
    declared_max_seq_length = model.max_seq_length
    model.max_seq_length = config.max_seq_length
    if config.use_gradient_checkpointing:
        inner: Any = model[0].auto_model  # type: ignore[index]
        # use_cache keeps per-layer key/value tensors alive, which is exactly
        # what checkpointing exists to discard; transformers warns and disables
        # it anyway, so turn it off rather than leave a contradiction in place.
        inner.config.use_cache = False
        inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer_class, optimizer_name = _resolve_optimizer(
        torch, config.use_8bit_optimizer and resolved_device == "cuda"
    )
    LOGGER.info(
        "Training on %s in %s with %s, max_seq_length %d (model declares %s)",
        resolved_device,
        "bfloat16" if use_bfloat16 else "float32",
        optimizer_name,
        config.max_seq_length,
        declared_max_seq_length,
    )

    train_examples = [InputExample(texts=[q, p]) for q, p in train_pairs]
    train_dataloader = DataLoader(
        train_examples,  # pyright: ignore[reportArgumentType]
        shuffle=True,
        batch_size=config.batch_size,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)

    evaluator = None
    if validation_pairs:
        val_queries = {f"vq_{i}": q for i, (q, _) in enumerate(validation_pairs)}
        val_corpus = {f"vc_{i}": p for i, (_, p) in enumerate(validation_pairs)}
        val_relevant = {f"vq_{i}": {f"vc_{i}"} for i in range(len(validation_pairs))}
        evaluator = InformationRetrievalEvaluator(
            val_queries, val_corpus, val_relevant, name="validation"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=config.epochs,
        warmup_steps=int(len(train_dataloader) * config.epochs * config.warmup_ratio),
        evaluation_steps=config.evaluation_steps if evaluator else 0,
        output_path=str(config.output_dir),
        optimizer_class=optimizer_class,
        optimizer_params={"lr": config.learning_rate},
        show_progress_bar=True,
    )

    hf_dir = config.output_dir / "hf_model"
    hf_dir.mkdir(parents=True, exist_ok=True)
    # `model[0]` is a Transformer module at runtime; the shipped stubs type
    # SentenceTransformer.__getitem__ as returning a Tensor.
    inner_model: Any = model[0].auto_model  # type: ignore[index]
    inner_model.save_pretrained(str(hf_dir))
    model.tokenizer.save_pretrained(str(hf_dir))  # type: ignore[union-attr]
    LOGGER.info("Saved HuggingFace-format checkpoint to %s", hf_dir)

    report = TrainingReport(
        base_model=config.base_model,
        output_dir=str(config.output_dir),
        train_pair_count=len(train_pairs),
        validation_pair_count=len(validation_pairs),
        excluded_query_count=excluded,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        dtype="bfloat16" if use_bfloat16 else "float32",
        optimizer=optimizer_name,
        max_seq_length=config.max_seq_length,
        gradient_checkpointing=config.use_gradient_checkpointing,
    )
    report_path = config.output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report.model_dump(), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    """Fine-tune a bi-encoder from the command line."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a bi-encoder on synthetic (query, passage) pairs."
    )
    parser.add_argument("base_model", help="HuggingFace model name or local path")
    parser.add_argument("--queries", type=Path, required=True, help="Retained query JSONL")
    parser.add_argument("--chunks", type=Path, required=True, help="SplitChunkRecord JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--evaluation-steps", type=int, default=DEFAULT_EVALUATION_STEPS)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fp32",
        action="store_true",
        default=False,
        help="Train in float32 instead of bfloat16 (a 0.6B encoder will not fit on 10 GB)",
    )
    parser.add_argument(
        "--no-8bit-optimizer",
        action="store_true",
        default=False,
        help="Use fp32 AdamW moments instead of 8-bit, costing about 3.3 GB more",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        default=False,
        help="Keep all activations in memory; about 30%% faster and roughly 17 GB larger",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=(
            "Token cap per passage. Qwen3-Embedding declares 32768, which is 300x "
            "the median Clarity passage and makes activations dominate memory."
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
        report = train_biencoder(
            BiEncoderTrainingConfig(
                base_model=args.base_model,
                retained_queries_path=args.queries,
                chunks_path=args.chunks,
                output_dir=args.output_dir,
                trust_remote_code=args.trust_remote_code,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                warmup_ratio=args.warmup_ratio,
                evaluation_steps=args.evaluation_steps,
                device=args.device,
                seed=args.seed,
                use_bfloat16=not args.fp32,
                use_8bit_optimizer=not args.no_8bit_optimizer,
                max_seq_length=args.max_seq_length,
                use_gradient_checkpointing=not args.no_gradient_checkpointing,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Trained on {report.train_pair_count:,} pairs for {report.epochs} epochs "
        f"in {report.dtype} with {report.optimizer}. Model saved to {report.output_dir}"
    )


if __name__ == "__main__":
    main()
