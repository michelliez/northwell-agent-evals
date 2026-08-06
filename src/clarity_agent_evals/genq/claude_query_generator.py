"""Claude Haiku provider for offline synthetic-query generation."""

from __future__ import annotations

import json
import logging
import os
import ssl
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi

from clarity_agent_evals.genq.prompts import QUERY_GENERATION_SYSTEM_PROMPT

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_TIMEOUT_SECONDS = 60.0
LOGGER = logging.getLogger(__name__)

# Shared with every other provider so that comparing generators compares models
# rather than two prompts that drifted apart. See prompts.py for the rejected
# variants and why prompt tuning was abandoned.
SYSTEM_PROMPT = QUERY_GENERATION_SYSTEM_PROMPT


class MessagesClient(Protocol):
    """Subset of the Anthropic client used by this provider."""

    messages: Any


class _HTTPMessagesClient:
    """Minimal Anthropic Messages client for offline query generation.

    Importing the complete generated Anthropic SDK is disproportionately slow
    in the project's Python 3.14 environment. GenQ only needs one JSON endpoint,
    so this bounded client avoids loading hundreds of unrelated SDK types.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        default_headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> None:
        self.messages = self
        self._url = _messages_url(base_url)
        self._timeout_seconds = timeout_seconds
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())
        self._headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "anthropic-version": DEFAULT_ANTHROPIC_VERSION,
            "x-api-key": api_key,
            **dict(default_headers),
        }

    def create(self, **payload: Any) -> dict[str, Any]:
        request = Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        try:
            with urlopen(  # noqa: S310
                request,
                timeout=self._timeout_seconds,
                context=self._ssl_context,
            ) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise RuntimeError(f"Claude query request failed with HTTP {exc.code}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Claude query request timed out after {self._timeout_seconds:g} seconds"
            ) from exc
        except URLError as exc:
            raise RuntimeError("Claude query request could not reach the configured API") from exc

        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Claude query API returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("Claude query API returned an invalid response object")
        return decoded


class ClaudeHaikuQueryGenerator:
    """Generate structured query lists with Claude Haiku.

    ``passages_per_request`` controls how many passages share one API call. The
    default of 1 preserves byte-identical behaviour with earlier runs, so their
    recorded ``output_query_hash`` still reproduces. Raising it amortises the
    request prologue, which measurement shows is ~692 tokens of tool-schema
    scaffolding against a ~45-token passage: batching 10 cuts input from ~748 to
    ~135 tokens per passage.
    """

    device_name = "anthropic-api"

    def __init__(
        self,
        model_name: str = DEFAULT_CLAUDE_MODEL,
        *,
        client: MessagesClient | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        passages_per_request: int = 1,
    ) -> None:
        self.model_name = model_name
        if passages_per_request < 1:
            raise ValueError("passages_per_request must be at least 1")
        self.passages_per_request = passages_per_request
        if client is not None:
            self._client = client
            return

        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("AI_HUB_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "Claude query generation requires ANTHROPIC_API_KEY or AI_HUB_API_KEY"
            )
        resolved_base_url = (
            base_url or os.getenv("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC_BASE_URL
        )
        timeout_seconds = _timeout_from_env()
        self._client = _HTTPMessagesClient(
            api_key=resolved_key,
            base_url=resolved_base_url,
            default_headers=default_headers or _headers_from_env(),
            timeout_seconds=timeout_seconds,
        )

    def generate(
        self,
        passages: Sequence[str],
        *,
        queries_per_passage: int,
        max_input_tokens: int,
        max_query_tokens: int,
        top_p: float,
        seed: int,
    ) -> list[list[str]]:
        """Generate exact-size query lists; Claude does not expose a seed parameter."""
        del seed
        results: list[list[str]] = []
        max_passage_chars = max_input_tokens * 4
        stride = self.passages_per_request
        for start in range(0, len(passages), stride):
            group = [passage[:max_passage_chars] for passage in passages[start : start + stride]]
            LOGGER.info(
                "Requesting Claude queries for passages %d-%d/%d",
                start + 1,
                start + len(group),
                len(passages),
            )
            if len(group) == 1:
                response = self._client.messages.create(
                    **_single_payload(
                        self.model_name, group[0], queries_per_passage, max_query_tokens, top_p
                    )
                )
                results.append(_extract_queries(response, queries_per_passage))
            else:
                response = self._client.messages.create(
                    **_batched_payload(
                        self.model_name, group, queries_per_passage, max_query_tokens, top_p
                    )
                )
                results.extend(_extract_batched_queries(response, len(group), queries_per_passage))
            LOGGER.info(
                "Received Claude queries through passage %d/%d", start + len(group), len(passages)
            )
        return results


def _single_payload(
    model_name: str,
    passage: str,
    queries_per_passage: int,
    max_query_tokens: int,
    top_p: float,
) -> dict[str, Any]:
    """Build the one-passage request. Kept byte-identical to the original shape."""
    return {
        "model": model_name,
        "max_tokens": min(4096, max(256, queries_per_passage * max_query_tokens * 2)),
        "top_p": top_p,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"{_query_contract(queries_per_passage)}\n\n"
                    "Passage:\n"
                    f"{passage}"
                ),
            }
        ],
        "tools": [
            {
                "name": "return_queries",
                "description": "Return the generated semantic-search questions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "minItems": queries_per_passage,
                            "maxItems": queries_per_passage,
                        }
                    },
                    "required": ["queries"],
                    "additionalProperties": False,
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "return_queries"},
    }


def _batched_payload(
    model_name: str,
    passages: Sequence[str],
    queries_per_passage: int,
    max_query_tokens: int,
    top_p: float,
) -> dict[str, Any]:
    """Build a request carrying several passages, each tagged with its index."""
    listing = "\n\n".join(
        f"<passage index={index}>\n{passage}\n</passage>" for index, passage in enumerate(passages)
    )
    return {
        "model": model_name,
        "max_tokens": min(8192, 128 + len(passages) * queries_per_passage * max_query_tokens),
        "top_p": top_p,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"{_query_contract(queries_per_passage, each=True)} "
                    f"Return one result object per "
                    "passage, with passage_index matching the index attribute.\n\n"
                    f"{listing}"
                ),
            }
        ],
        "tools": [
            {
                "name": "return_queries",
                "description": "Return generated semantic-search questions for every passage.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "passage_index": {"type": "integer"},
                                    "queries": {
                                        "type": "array",
                                        "items": {"type": "string", "minLength": 1},
                                        "minItems": queries_per_passage,
                                        "maxItems": queries_per_passage,
                                    },
                                },
                                "required": ["passage_index", "queries"],
                                "additionalProperties": False,
                            },
                            "minItems": len(passages),
                            "maxItems": len(passages),
                        }
                    },
                    "required": ["results"],
                    "additionalProperties": False,
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "return_queries"},
    }


def _query_contract(queries_per_passage: int, *, each: bool = False) -> str:
    """Describe ordered query styles without changing the structured output shape."""
    target = "for EACH passage" if each else "for this passage"
    if queries_per_passage == 2:
        return (
            f"Generate exactly two distinct search questions {target}. Preserve this order: "
            "queries[0] is identifier-aware and may name a documented table or column when "
            "natural; queries[1] is an indirect semantic question that MUST NOT mention the "
            "table name, column name, source filename, or copy more than two consecutive words "
            "from the passage. The indirect question should sound like an analyst who knows the "
            "business concept but does not know where it is documented."
        )
    return f"Generate exactly {queries_per_passage} distinct search questions {target}."


def _extract_queries(response: Any, expected_count: int) -> list[str]:
    content = response.get("content") if isinstance(response, dict) else response.content
    if not isinstance(content, list):
        raise RuntimeError("Claude did not return the required return_queries tool result")
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        block_name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
        if block_type != "tool_use":
            continue
        if block_name != "return_queries":
            continue
        payload = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
        if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
            break
        queries = [q for q in payload["queries"] if isinstance(q, str)]
        if len(queries) < expected_count:
            raise RuntimeError(
                f"Claude returned {len(queries)} valid query values; expected {expected_count}"
            )
        return queries[:expected_count]
    raise RuntimeError("Claude did not return the required return_queries tool result")


def _extract_batched_queries(
    response: Any,
    expected_passages: int,
    expected_count: int,
) -> list[list[str]]:
    """Unpack a multi-passage reply, refusing anything that could mis-map queries.

    The single-passage path is positionally safe: one request, one answer. Once a
    request carries several passages, alignment depends on a model-supplied
    ``passage_index``. A wrong index would attach queries to the wrong column and
    silently poison training data in a way no downstream check inspects, so every
    index is verified to appear exactly once before any result is returned.
    """
    content = response.get("content") if isinstance(response, dict) else response.content
    if not isinstance(content, list):
        raise RuntimeError("Claude did not return the required return_queries tool result")
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        block_name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
        if block_type != "tool_use" or block_name != "return_queries":
            continue
        payload = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            break

        by_index: dict[int, list[str]] = {}
        for entry in payload["results"]:
            if not isinstance(entry, dict):
                raise RuntimeError("Claude returned a malformed batched result entry")
            index = entry.get("passage_index")
            queries = entry.get("queries")
            if not isinstance(index, int) or isinstance(index, bool):
                raise RuntimeError(f"Claude returned a non-integer passage_index: {index!r}")
            if index in by_index:
                raise RuntimeError(f"Claude returned passage_index {index} more than once")
            if not isinstance(queries, list):
                raise RuntimeError(f"Claude returned no query list for passage_index {index}")
            valid = [query for query in queries if isinstance(query, str) and query.strip()]
            if len(valid) < expected_count:
                raise RuntimeError(
                    f"Claude returned {len(valid)} valid queries for passage_index {index}; "
                    f"expected {expected_count}"
                )
            by_index[index] = valid[:expected_count]

        expected_indexes = set(range(expected_passages))
        if set(by_index) != expected_indexes:
            missing = sorted(expected_indexes - set(by_index))
            unexpected = sorted(set(by_index) - expected_indexes)
            raise RuntimeError(
                f"Claude returned passage indexes {sorted(by_index)} for {expected_passages} "
                f"passages (missing={missing}, unexpected={unexpected})"
            )
        return [by_index[index] for index in range(expected_passages)]
    raise RuntimeError("Claude did not return the required return_queries tool result")


def _headers_from_env() -> dict[str, str]:
    raw = os.getenv("ANTHROPIC_CUSTOM_HEADERS", "")
    headers: dict[str, str] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        name, separator, value = item.partition(":")
        if not separator or not name.strip() or not value.strip():
            raise RuntimeError(
                "ANTHROPIC_CUSTOM_HEADERS must contain comma-separated name:value pairs"
            )
        headers[name.strip()] = value.strip()
    return headers


def _messages_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return base + "/messages"
    return base + "/v1/messages"


def _timeout_from_env() -> float:
    raw = os.getenv("GENQ_CLAUDE_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError("GENQ_CLAUDE_TIMEOUT_SECONDS must be numeric") from exc
    if value <= 0:
        raise RuntimeError("GENQ_CLAUDE_TIMEOUT_SECONDS must be greater than zero")
    return value
