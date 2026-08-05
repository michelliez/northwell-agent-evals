from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_host.config import get_config
from agent_host.graph import ask
from agent_host.nodes.intent_nodes import classify_intent
from clarity_agent_evals.assertions import EvaluationCase, evaluate_case
from clarity_agent_evals.intent_assertions import (
    IntentEvaluationCase,
    IntentResult,
    evaluate_intent_case,
    summarize_intent_results,
)
from clarity_agent_evals.retrieval_evaluator import run_retrieval_evaluation

INTENT_PROMPT_VERSION = "v6"

DEFAULT_CASE_DIR = Path("evals")
DEFAULT_RESULTS_DIR = Path(".local/evals")
DEFAULT_BENCHMARK_DIR = DEFAULT_CASE_DIR / "retrieval" / "benchmark"


def _default_index_path() -> Path:
    """Use the same RAG_DB_PATH/.env resolution as the agent."""
    return get_config().index_path


def load_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw_case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        cases.append(EvaluationCase.from_dict(raw_case))
    if not cases:
        raise ValueError(f"{path}: no evaluation cases found")
    return cases


def load_intent_cases(path: Path) -> list[IntentEvaluationCase]:
    cases: list[IntentEvaluationCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw_case = json.loads(line)
            cases.append(IntentEvaluationCase.from_dict(raw_case))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid intent evaluation case") from exc
    if not cases:
        raise ValueError(f"{path}: no intent evaluation cases found")
    return cases


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    trace_path = Path(path)
    if not trace_path.is_file():
        return []
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_cases(cases: list[EvaluationCase]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            response = ask(case.prompt)
            response_dict = response.model_dump()
            events = load_trace(response.trace_file or "")
            failures = evaluate_case(case, response_dict, events)
            results.append(
                {
                    "id": case.id,
                    "category": case.category,
                    "passed": not failures,
                    "run_id": response.run_id,
                    "failures": [asdict(failure) for failure in failures],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "id": case.id,
                    "category": case.category,
                    "passed": False,
                    "run_id": None,
                    "failures": [{"check": "runner", "message": f"{type(exc).__name__}: {exc}"}],
                }
            )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "total": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "failed": sum(1 for result in results if not result["passed"]),
        "results": results,
    }


def run_intent_cases(cases: list[IntentEvaluationCase], repetitions: int) -> dict[str, Any]:
    """Run synthetic labels through the same classifier used by the graph."""
    cfg = get_config()

    results: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for case in cases:
            expected = {
                "intent": case.expected_intent,
                "recommended_action": case.expected_recommended_action,
                "needs_clarification": case.expected_needs_clarification,
                "safety_class": case.safety_class,
            }
            try:
                result_obj = IntentResult.model_validate(classify_intent(case.prompt, cfg))
                failures = evaluate_intent_case(case, result_obj)
                results.append(
                    {
                        "id": case.id,
                        "round": case.round,
                        "repetition": repetition,
                        "category": case.category,
                        "expected": expected,
                        "observed": result_obj.model_dump(),
                        "passed": not failures,
                        "operational_failure": False,
                        "failures": [asdict(failure) for failure in failures],
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "id": case.id,
                        "round": case.round if hasattr(case, "round") else 1,
                        "repetition": repetition,
                        "category": case.category,
                        "expected": expected,
                        "observed": None,
                        "passed": False,
                        "operational_failure": True,
                        "failures": [
                            {"check": "runner", "message": f"{type(exc).__name__}: {exc}"}
                        ],
                    }
                )

    return {
        "suite": "intent",
        "generated_at": datetime.now(UTC).isoformat(),
        "intent_prompt_version": INTENT_PROMPT_VERSION,
        "repetitions": repetitions,
        "metrics": summarize_intent_results(results),
        "results": results,
    }


def _print_retrieval_report(report: dict[str, Any], report_path: Path) -> None:
    metrics = report["metrics"]
    k_values = report["k_values"]
    header = "".join(f"@{k:<14}" for k in k_values)

    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 80)
    print(f"Report: {report_path.name}")
    print(f"Chunker: {report.get('chunker_version', 'unknown')}")
    print(f"Total queries: {report['query_count']}")
    print(
        f"Answerable: {report['answerable_query_count']}, "
        f"unanswerable: {report['unanswerable_query_count']}"
    )
    print(f"Judgment complete: {report['judgment_complete']}")
    print(f"Unjudged documents: {report['unjudged_document_total']}")
    print(f"Unjudged chunks: {report['unjudged_chunk_total']}")
    print("=" * 80)

    for level in ("document", "chunk"):
        print(f"\n{level.upper()}-LEVEL METRICS:")
        print("-" * 50)
        print(f"{'METRIC':<20}{header}")
        for metric in ("precision", "recall", "ndcg", "hit"):
            cells = ""
            for k in k_values:
                value = metrics[level].get(f"{metric}@{k}")
                cells += f"{'N/A' if value is None else f'{value:.3f}':<15}"
            print(f"{metric.upper():<20}{cells}")

    first_k = k_values[0]
    print(f"\nHIT@{first_k} BY FAILURE BUCKET:")
    print("-" * 70)
    print(f"{'BUCKET':<32}{'SCORED/TOTAL':<14}{'DOCUMENT':<14}{'CHUNK':<14}")
    for bucket, bucket_metrics in metrics["by_failure_bucket"].items():
        document_hit = bucket_metrics["document"][f"hit@{first_k}"]
        chunk_hit = bucket_metrics["chunk"][f"hit@{first_k}"]
        document_cell = "N/A" if document_hit is None else f"{document_hit:.3f}"
        chunk_cell = "N/A" if chunk_hit is None else f"{chunk_hit:.3f}"
        counts = f"{bucket_metrics.get('scored_query_count', 0)}/{bucket_metrics['query_count']}"
        print(f"{bucket:<32}{counts:<14}{document_cell:<14}{chunk_cell:<14}")

    print("\nLATENCY (ms):")
    print("-" * 50)
    for label, key in (("Median", "median"), ("P95", "p95"), ("Max", "max")):
        print(f"{label:<20}{metrics['latency_ms'][key]:.1f}")
    print("\n" + "=" * 80 + "\n")


def write_report(report: dict[str, Any], results_dir: Path, prefix: str = "evaluation") -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"{prefix}-{timestamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run synthetic trace-based or direct intent evaluation cases."
    )
    parser.add_argument(
        "--suite",
        choices=("smoke", "red_team", "intent", "retrieval"),
        default="smoke",
        help="Case suite to run.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=DEFAULT_CASE_DIR,
        help="Directory containing suite JSONL files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for generated JSON reports.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repeat the intent suite to measure model variability.",
    )
    parser.add_argument(
        "--retriever",
        choices=("fts",),
        default="fts",
        help="Retriever implementation for the retrieval suite.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=_default_index_path(),
        help=(
            "SQLite RAG index for the retrieval suite "
            "(default: RAG_DB_PATH or .local/rag/index.sqlite)."
        ),
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="Directory holding the reviewed retrieval benchmark files.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=(5, 10),
        help="Ranking cutoffs for the retrieval suite.",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if any(k < 1 for k in args.k):
        parser.error("every --k value must be at least 1")

    if args.suite == "retrieval":
        report = run_retrieval_evaluation(
            db_path=args.db,
            queries_path=args.benchmark_dir / "retrieval_queries.jsonl",
            qrels_path=args.benchmark_dir / "retrieval_qrels.jsonl",
            chunk_qrels_path=args.benchmark_dir / "retrieval_chunk_qrels.jsonl",
            catalog_path=args.benchmark_dir / "retrieval_catalog.jsonl",
            retriever=args.retriever,
            k_values=args.k,
        )
        report_path = write_report(report, args.results_dir, prefix="retrieval-evaluation")
        _print_retrieval_report(report, report_path)
        print(
            f"Resolution: {report['resolved_query_count']} resolved, "
            f"{report['unresolved_query_count']} unresolved"
        )
        if report["resolution_summary"]:
            print(f"Unresolved states: {report['resolution_summary']}")
        print(f"\n{report['metrics_note']}")
        print(f"Report: {report_path}")
        return

    if args.suite == "intent":
        cases = load_intent_cases(args.case_dir / "intent.jsonl")
        report = run_intent_cases(cases, args.repetitions)
        report_path = write_report(report, args.results_dir, prefix="intent-evaluation")
        metrics = report["metrics"]
        print(
            f"{metrics['observed_total']} / {metrics['total']} observations completed; "
            f"{metrics['reported_differences']} labelled differences"
        )
        print(f"Intent accuracy: {metrics['intent_accuracy']}")
        print(f"Unsafe-to-safe routes: {metrics['unsafe_to_safe_routes']}")
        print(f"Report: {report_path}")
        if metrics["operational_failures"]:
            raise SystemExit(1)
        return

    cases = load_cases(args.case_dir / f"{args.suite}.jsonl")
    report = run_cases(cases)
    report_path = write_report(report, args.results_dir)
    print(f"{report['passed']} / {report['total']} passed")
    for result in report["results"]:
        if not result["passed"]:
            details = "; ".join(failure["message"] for failure in result["failures"])
            print(f"FAIL {result['id']}: {details}")
    print(f"Report: {report_path}")
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
