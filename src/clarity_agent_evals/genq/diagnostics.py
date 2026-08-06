"""Distribution diagnostics for a generated query set.

Every number here is computed from text. None of them asks anyone to judge
whether a query reads naturally, which is the judgement that does not scale to
one reviewer and tens of thousands of queries.

The reason this exists: a generated set can be large, fluent, and still useless
for evaluation, and nothing in the pipeline noticed. A copy threshold was once
raised to 0.86, where it rejected nothing across 62,120 queries, and later
lowered to 0.35, where it rejected half of a run for the wrong reason. Both
passed review. A benchmark that a bag-of-words scorer solves cannot rank
retrievers, and the only way to know before training is to measure the set.

Four measures, each answering a different failure:

leakage
    Share of a query's terms that already occur in its positive passage. This
    is the load-bearing one. Above roughly 0.7 the set rewards term overlap and
    an inverted index will score near the ceiling for a reason unrelated to
    retrieval quality.

heaps_beta
    Vocabulary growth fitted as V(n) = K * n**beta. Human query sets grow with
    beta near 0.5-0.7. A generated set whose vocabulary saturates early is
    paraphrasing a template, and the queries past that point add nothing.

length
    Query length in tokens. Compared against a reference rather than a fixed
    expectation, because the target depends on how queries are elicited: search
    logs mode at 2-4 tokens, while analyst-phrased benchmark questions run far
    longer. A fixed threshold would mislabel one as the other.

near_duplicate
    Similarity to the closest other query in the set. Detects a set that was
    written once and re-rolled. Lexical rather than embedding-based, so it
    cannot be flattered by an encoder trained on this same data.

Interpretation needs a reference distribution from the same domain, which is
what --benchmark-dir supplies. Thresholds borrowed from another corpus are how
a set gets declared healthy for the wrong reason.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)
DIAGNOSTICS_VERSION = "genq-distribution-diagnostics-v1"

# Above this share of query terms already present in the passage, the set is
# measuring term overlap rather than retrieval. Advisory: the command reports
# the number and flags it, and never decides on the operator's behalf.
LEAKAGE_CONCERN_THRESHOLD = 0.7
# Pairwise nearest-neighbour work is quadratic, and the shape of the
# distribution is clear long before the whole set is compared.
DEFAULT_NEIGHBOUR_SAMPLE = 1500
DEFAULT_SEED = 42

_TOKEN = re.compile(r"[a-z0-9_]+")

# Words carrying no retrieval signal in this corpus. Deliberately the same list
# the filter uses, so "content terms" means one thing across the pipeline.
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "can",
        "column",
        "data",
        "define",
        "description",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "information",
        "of",
        "on",
        "or",
        "the",
        "this",
        "table",
        "to",
        "what",
        "where",
        "which",
        "with",
    }
)


def _tokens(value: str) -> list[str]:
    return _TOKEN.findall(value.casefold())


def _content_tokens(value: str) -> set[str]:
    return {
        token
        for token in _tokens(value)
        if token not in STOP_WORDS and (len(token) >= 3 or any(c.isdigit() for c in token))
    }


class LengthProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: float
    median: float
    mode: int
    p10: float
    p90: float


class DiagnosticsReport(BaseModel):
    """What a generated set looks like, beside what it is meant to resemble."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    diagnostics_version: str
    label: str
    query_count: int = Field(ge=0)
    queries_path: str
    leakage_all_terms: float
    leakage_content_terms: float
    leakage_exceeds_threshold: bool
    leakage_threshold: float
    heaps_beta: float
    vocabulary_size: int = Field(ge=0)
    length: LengthProfile
    near_duplicate_median: float
    near_duplicate_sample: int = Field(ge=0)
    reference: DiagnosticsReport | None = None


@dataclass(frozen=True)
class DiagnosticsConfig:
    queries_path: Path
    chunks_path: Path
    report_path: Path | None = None
    benchmark_dir: Path | None = None
    decision: str | None = None
    neighbour_sample: int = DEFAULT_NEIGHBOUR_SAMPLE
    seed: int = DEFAULT_SEED

    def validate(self) -> None:
        if self.neighbour_sample < 2:
            raise ValueError("neighbour_sample must be at least 2")
        if self.decision is not None and self.decision not in {"retain", "reject", "review"}:
            raise ValueError("decision must be one of retain, reject, review")


def _load_jsonl(path: Path, label: str) -> list[dict]:
    try:
        raw = path.read_text("utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"{label} JSONL does not exist: {path}") from exc
    records = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid {label} record: {exc}") from exc
    if not records:
        raise ValueError(f"{label} JSONL is empty: {path}")
    return records


def load_pairs(config: DiagnosticsConfig) -> list[tuple[str, str]]:
    """Return (query, positive passage) for every query in the set."""
    passages = {
        record["chunk_id"]: record["text"] for record in _load_jsonl(config.chunks_path, "chunks")
    }
    pairs: list[tuple[str, str]] = []
    missing = 0
    for record in _load_jsonl(config.queries_path, "queries"):
        if config.decision and record.get("decision") != config.decision:
            continue
        passage = passages.get(record.get("relevant_chunk_id", ""))
        if passage is None:
            missing += 1
            continue
        pairs.append((record["query"], passage))
    if missing:
        LOGGER.warning("%d queries referenced a chunk absent from %s", missing, config.chunks_path)
    if not pairs:
        raise ValueError("No query/passage pairs survived loading; check --chunks and --decision")
    return pairs


def load_benchmark_pairs(benchmark_dir: Path, chunks_path: Path) -> list[tuple[str, str]]:
    """Return (query, passage) for the reviewed benchmark, as a reference distribution."""
    by_file: dict[str, list[str]] = {}
    for record in _load_jsonl(chunks_path, "chunks"):
        by_file.setdefault(record["source_file"], []).append(record["text"])

    catalog = {
        record["document_key"]: record["source_path"]
        for record in _load_jsonl(benchmark_dir / "retrieval_catalog.jsonl", "catalog")
    }
    documents: dict[str, list[str]] = {}
    for record in _load_jsonl(benchmark_dir / "retrieval_qrels.jsonl", "qrels"):
        source_path = catalog.get(record["document_key"])
        if source_path:
            documents.setdefault(record["query_id"], []).extend(by_file.get(source_path, []))

    pairs = []
    for record in _load_jsonl(benchmark_dir / "retrieval_queries.jsonl", "benchmark queries"):
        texts = documents.get(record["query_id"])
        if texts:
            pairs.append((record["query"], " ".join(texts)))
    return pairs


def leakage(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    """Mean share of a query's terms occurring in its passage, all then content only."""
    all_terms: list[float] = []
    content_terms: list[float] = []
    for query, passage in pairs:
        passage_all = set(_tokens(passage))
        passage_content = _content_tokens(passage)
        query_all = _tokens(query)
        query_content = _content_tokens(query)
        if query_all:
            all_terms.append(sum(1 for t in query_all if t in passage_all) / len(query_all))
        if query_content:
            content_terms.append(len(query_content & passage_content) / len(query_content))
    return (
        statistics.mean(all_terms) if all_terms else 0.0,
        statistics.mean(content_terms) if content_terms else 0.0,
    )


def heaps_beta(queries: list[str], seed: int = DEFAULT_SEED) -> tuple[float, int]:
    """Fit V(n) = K * n**beta by least squares on the log-log vocabulary curve.

    Shuffled first, because a set written table by table would otherwise show
    vocabulary arriving in alphabetical bursts rather than at its true rate.
    """
    shuffled = list(queries)
    random.Random(seed).shuffle(shuffled)
    checkpoints = {int(round(len(shuffled) ** (i / 24))) for i in range(1, 25)}
    vocabulary: set[str] = set()
    points: list[tuple[float, float]] = []
    for index, query in enumerate(shuffled, start=1):
        vocabulary |= set(_tokens(query))
        if index in checkpoints and index >= 10:
            points.append((math.log(index), math.log(len(vocabulary))))
    if len(points) < 3:
        return float("nan"), len(vocabulary)
    mean_x = statistics.mean(x for x, _ in points)
    mean_y = statistics.mean(y for _, y in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return float("nan"), len(vocabulary)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in points)
    return numerator / denominator, len(vocabulary)


def length_profile(queries: list[str]) -> LengthProfile:
    counts = [len(_tokens(query)) for query in queries]
    if len(counts) < 10:
        low = high = float(statistics.median(counts))
    else:
        deciles = statistics.quantiles(counts, n=10)
        low, high = float(deciles[0]), float(deciles[-1])
    return LengthProfile(
        mean=statistics.mean(counts),
        median=statistics.median(counts),
        mode=Counter(counts).most_common(1)[0][0],
        p10=low,
        p90=high,
    )


def near_duplicate_median(
    queries: list[str],
    *,
    sample_size: int = DEFAULT_NEIGHBOUR_SAMPLE,
    seed: int = DEFAULT_SEED,
) -> tuple[float, int]:
    """Median Jaccard similarity to the most similar other query in the sample."""
    sample = list(queries)
    random.Random(seed).shuffle(sample)
    sample = sample[:sample_size]
    token_sets = [_content_tokens(query) for query in sample]
    # An inverted index keeps this to pairs that share a term; comparing every
    # pair is quadratic in the sample and most pairs share nothing.
    postings: dict[str, list[int]] = {}
    for index, tokens in enumerate(token_sets):
        for token in tokens:
            postings.setdefault(token, []).append(index)

    best: list[float] = []
    for index, tokens in enumerate(token_sets):
        if not tokens:
            continue
        shared: Counter[int] = Counter()
        for token in tokens:
            for other in postings[token]:
                if other != index:
                    shared[other] += 1
        top = 0.0
        for other, overlap in shared.items():
            union = len(tokens) + len(token_sets[other]) - overlap
            if union:
                top = max(top, overlap / union)
        best.append(top)
    return (statistics.median(best) if best else float("nan"), len(sample))


def diagnose(
    pairs: list[tuple[str, str]],
    *,
    label: str,
    queries_path: Path,
    sample_size: int = DEFAULT_NEIGHBOUR_SAMPLE,
    seed: int = DEFAULT_SEED,
) -> DiagnosticsReport:
    queries = [query for query, _ in pairs]
    leak_all, leak_content = leakage(pairs)
    beta, vocabulary = heaps_beta(queries, seed=seed)
    neighbour, sampled = near_duplicate_median(queries, sample_size=sample_size, seed=seed)
    return DiagnosticsReport(
        diagnostics_version=DIAGNOSTICS_VERSION,
        label=label,
        query_count=len(pairs),
        queries_path=str(queries_path),
        leakage_all_terms=leak_all,
        leakage_content_terms=leak_content,
        leakage_exceeds_threshold=leak_content >= LEAKAGE_CONCERN_THRESHOLD,
        leakage_threshold=LEAKAGE_CONCERN_THRESHOLD,
        heaps_beta=beta,
        vocabulary_size=vocabulary,
        length=length_profile(queries),
        near_duplicate_median=neighbour,
        near_duplicate_sample=sampled,
    )


def run_diagnostics(config: DiagnosticsConfig) -> DiagnosticsReport:
    config.validate()
    report = diagnose(
        load_pairs(config),
        label=config.queries_path.stem,
        queries_path=config.queries_path,
        sample_size=config.neighbour_sample,
        seed=config.seed,
    )
    if config.benchmark_dir is not None:
        reference_pairs = load_benchmark_pairs(config.benchmark_dir, config.chunks_path)
        if reference_pairs:
            report = report.model_copy(
                update={
                    "reference": diagnose(
                        reference_pairs,
                        label="reviewed benchmark",
                        queries_path=config.benchmark_dir,
                        sample_size=config.neighbour_sample,
                        seed=config.seed,
                    )
                }
            )
        else:
            LOGGER.warning(
                "Benchmark directory produced no query/passage pairs: %s", config.benchmark_dir
            )
    if config.report_path is not None:
        config.report_path.parent.mkdir(parents=True, exist_ok=True)
        config.report_path.write_text(
            json.dumps(report.model_dump(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return report


def _print(report: DiagnosticsReport) -> None:
    rows = [report] + ([report.reference] if report.reference else [])
    print(
        f"{'set':24} {'n':>8} {'leak(all)':>10} {'leak(content)':>14} "
        f"{'beta':>7} {'vocab':>8} {'len mean':>9} {'mode':>5} {'NN median':>10}"
    )
    for row in rows:
        print(
            f"{row.label[:24]:24} {row.query_count:8,} {row.leakage_all_terms:10.3f} "
            f"{row.leakage_content_terms:14.3f} {row.heaps_beta:7.3f} {row.vocabulary_size:8,} "
            f"{row.length.mean:9.1f} {row.length.mode:5} {row.near_duplicate_median:10.3f}"
        )
    if report.leakage_exceeds_threshold:
        print(
            f"\nWARNING: content-term leakage {report.leakage_content_terms:.3f} is at or above "
            f"{report.leakage_threshold}. Term overlap alone can find the positive passage, so "
            "this set cannot rank retrievers; a bag-of-words baseline will score near the ceiling."
        )
    if report.reference is not None:
        ratio = report.leakage_content_terms / max(report.reference.leakage_content_terms, 1e-9)
        print(
            f"\nLeakage is {ratio:.1f}x the reviewed benchmark "
            f"({report.leakage_content_terms:.3f} against {report.reference.leakage_content_terms:.3f})."
        )
        print(
            "Compare length against the benchmark rather than a fixed expectation: how queries "
            "were elicited sets the target."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the distribution of a generated query set against a reviewed benchmark."
    )
    parser.add_argument("queries_path", type=Path)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Reviewed benchmark to use as the reference distribution.",
    )
    parser.add_argument(
        "--decision",
        choices=("retain", "reject", "review"),
        default=None,
        help="Restrict to one decision when the input is a review ledger.",
    )
    parser.add_argument("--neighbour-sample", type=int, default=DEFAULT_NEIGHBOUR_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s"
    )

    try:
        report = run_diagnostics(
            DiagnosticsConfig(
                queries_path=args.queries_path,
                chunks_path=args.chunks,
                report_path=args.report,
                benchmark_dir=args.benchmark_dir,
                decision=args.decision,
                neighbour_sample=args.neighbour_sample,
                seed=args.seed,
            )
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    _print(report)


if __name__ == "__main__":
    main()
