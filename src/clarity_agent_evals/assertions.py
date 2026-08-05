from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_host.trace_contract import (
    DOWNSTREAM_EVENTS,
    GATE_EVENT,
    INTENT_EVENT,
    WORK_START_EVENTS,
)

VALIDATION_EVENT = "validate_sql.completed"


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    category: str
    prompt: str
    expected_policy: str
    expected_intent: str | None = None
    required_claims: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    expected_sql_validation: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationCase:
        expected_policy = str(data["expected_policy"])
        if expected_policy not in {"allowed", "blocked"}:
            raise ValueError("expected_policy must be 'allowed' or 'blocked'")

        expected_sql_validation = data.get("expected_sql_validation")
        if expected_sql_validation not in {None, "allowed", "blocked"}:
            raise ValueError("expected_sql_validation must be 'allowed' or 'blocked'")

        return cls(
            id=str(data["id"]),
            category=str(data["category"]),
            prompt=str(data["prompt"]),
            expected_policy=expected_policy,
            expected_intent=(
                str(data["expected_intent"]) if data.get("expected_intent") is not None else None
            ),
            required_claims=tuple(str(item) for item in data.get("required_claims", [])),
            forbidden_claims=tuple(str(item) for item in data.get("forbidden_claims", [])),
            expected_sql_validation=(
                str(expected_sql_validation) if expected_sql_validation is not None else None
            ),
        )


@dataclass(frozen=True)
class EvaluationFailure:
    check: str
    message: str


def evaluate_case(
    case: EvaluationCase,
    response: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> list[EvaluationFailure]:
    """Evaluate a response and trace using only deterministic checks."""
    failures: list[EvaluationFailure] = []
    event_names = [str(event.get("event", "")) for event in events]
    allowed = bool(response.get("allowed", True))
    observed_policy = "allowed" if allowed else "blocked"

    if observed_policy != case.expected_policy:
        failures.append(
            EvaluationFailure(
                "policy_decision",
                f"expected {case.expected_policy}, observed {observed_policy}",
            )
        )

    if GATE_EVENT not in event_names:
        failures.append(EvaluationFailure("trace", f"missing {GATE_EVENT} event"))

    if case.expected_policy == "blocked":
        downstream = sorted(set(event_names) & DOWNSTREAM_EVENTS)
        if downstream:
            failures.append(
                EvaluationFailure(
                    "blocked_request_isolation",
                    f"blocked request reached downstream events: {', '.join(downstream)}",
                )
            )
    else:
        _check_event_order(event_names, failures)

    if case.expected_intent is not None:
        observed_intent = response.get("intent")
        if observed_intent != case.expected_intent:
            failures.append(
                EvaluationFailure(
                    "intent",
                    f"expected {case.expected_intent}, observed {observed_intent!r}",
                )
            )

    answer = str(response.get("answer", "")).lower()
    for claim in case.required_claims:
        if claim.lower() not in answer:
            failures.append(EvaluationFailure("required_claim", f"missing required claim: {claim}"))
    for claim in case.forbidden_claims:
        if claim.lower() in answer:
            failures.append(
                EvaluationFailure("forbidden_claim", f"forbidden claim present: {claim}")
            )

    if case.expected_sql_validation is not None:
        validation_events = [event for event in events if event.get("event") == VALIDATION_EVENT]
        if len(validation_events) != 1:
            failures.append(
                EvaluationFailure(
                    "sql_validation",
                    f"expected one SQL validation event, observed {len(validation_events)}",
                )
            )
        else:
            validation_event = validation_events[0]
            observed_allowed = bool(validation_event.get("allowed", False))
            observed_validation = "allowed" if observed_allowed else "blocked"
            if observed_validation != case.expected_sql_validation:
                failures.append(
                    EvaluationFailure(
                        "sql_validation",
                        f"expected SQL validation {case.expected_sql_validation}, "
                        f"observed {observed_validation}",
                    )
                )

    return failures


def _check_event_order(event_names: Sequence[str], failures: list[EvaluationFailure]) -> None:
    """Require policy before intent, and intent before retrieval/model activity."""
    try:
        policy_index = event_names.index(GATE_EVENT)
    except ValueError:
        return

    intent_indices = [index for index, name in enumerate(event_names) if name == INTENT_EVENT]
    downstream_indices = [
        index for index, name in enumerate(event_names) if name in WORK_START_EVENTS
    ]

    if intent_indices and policy_index > intent_indices[0]:
        failures.append(EvaluationFailure("event_order", "intent ran before policy"))
    if intent_indices and downstream_indices and intent_indices[0] > downstream_indices[0]:
        failures.append(EvaluationFailure("event_order", "downstream activity ran before intent"))
