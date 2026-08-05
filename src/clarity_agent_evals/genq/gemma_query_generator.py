from __future__ import annotations

import json
import logging
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)


class GemmaMLXQueryGenerator:
    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://127.0.0.1:8080",
    ) -> None:
        self.model_name = model_name
        self.device_name = "mlx-local-api"
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(os.getenv("GENQ_GEMMA_TIMEOUT_SECONDS", "120"))
        self.max_attempts = int(os.getenv("GENQ_GEMMA_MAX_ATTEMPTS", "5"))
        if self.max_attempts < 1:
            raise ValueError("GENQ_GEMMA_MAX_ATTEMPTS must be at least 1")

    def generate(
        self,
        passages: list[str],
        *,
        queries_per_passage: int,
        max_input_tokens: int,
        max_query_tokens: int,
        top_p: float,
        seed: int,
    ) -> list[list[str]]:
        del seed

        return [
            self._generate_for_passage(
                passage[: max_input_tokens * 4],
                queries_per_passage=queries_per_passage,
                max_query_tokens=max_query_tokens,
                top_p=top_p,
            )
            for passage in passages
        ]

    def _generate_for_passage(
        self,
        passage: str,
        *,
        queries_per_passage: int,
        max_query_tokens: int,
        top_p: float,
    ) -> list[str]:
        generated: list[str] = []
        for query_number in range(1, queries_per_passage + 1):
            previous = "\n".join(f"- {query}" for query in generated) or "None"
            prompt = f"""
Write ONE realistic search question that a healthcare data analyst might type
to find the documentation passage below.

Requirements:
- Output only one question, with no label, explanation, JSON, or Markdown.
- End the question with a question mark.
- Mention the documented table name when it is useful.
- Do not copy or summarize the passage.
- Make it different from the previous questions.

Previous questions:
{previous}

Documentation passage:
{passage}
""".strip()

            last_error: RuntimeError | None = None
            for attempt in range(1, self.max_attempts + 1):
                try:
                    queries = self._request_queries(
                        prompt,
                        queries_per_passage=1,
                        max_query_tokens=max_query_tokens,
                        top_p=top_p,
                        retry=attempt > 1,
                    )
                    candidate = queries[0].strip() if queries else ""
                    if not candidate.endswith("?"):
                        raise RuntimeError("Gemma did not return a question")
                    if candidate.casefold() in {query.casefold() for query in generated}:
                        raise RuntimeError("Gemma returned a duplicate query")
                    generated.append(candidate)
                    break
                except RuntimeError as exc:
                    last_error = exc
                    if attempt < self.max_attempts:
                        LOGGER.warning(
                            "Invalid Gemma query %d/%d (attempt %d/%d): %s; retrying",
                            query_number,
                            queries_per_passage,
                            attempt,
                            self.max_attempts,
                            exc,
                        )
            else:
                assert last_error is not None
                LOGGER.error(
                    "Skipping passage because Gemma failed to generate query %d/%d "
                    "after %d attempts: %s",
                    query_number,
                    queries_per_passage,
                    self.max_attempts,
                    last_error,
                )
                return []

        return generated

    def _request_queries(
        self,
        prompt: str,
        *,
        queries_per_passage: int,
        max_query_tokens: int,
        top_p: float,
        retry: bool,
    ) -> list[str]:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate search queries for evaluating a "
                        "documentation retrieval system. Follow the requested "
                        "output format exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2 if retry else 0.7,
            "top_p": top_p,
            "max_tokens": max(32, queries_per_passage * max_query_tokens),
        }
        request = Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_data = json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(
                "Gemma query generation could not reach the local MLX server. "
                "Make sure mlx_lm.server is running."
            ) from exc

        try:
            message = response_data["choices"][0]["message"]
            text = message["content"] if isinstance(message, dict) else str(message)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("The MLX server returned an unexpected response.") from exc

        try:
            return self._parse_queries(text)
        except RuntimeError:
            # Only log malformed model output, never the source passage. The
            # bounded preview makes local-model formatting failures debuggable
            # without flooding long generation logs.
            LOGGER.warning("Gemma raw invalid response: %r", text[:500])
            raise

    @staticmethod
    def _parse_queries(text: str) -> list[str]:
        cleaned = text.strip()

        tagged_queries = [
            query.strip()
            for query in re.findall(
                r"<query>\s*(.*?)\s*</query>",
                cleaned,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if query.strip()
        ]
        if tagged_queries:
            if len(tagged_queries) != len(set(tagged_queries)):
                raise RuntimeError("Gemma returned duplicate queries.")
            return tagged_queries

        plain_questions = []
        for line in cleaned.splitlines():
            candidate = re.sub(
                r"^\s*(?:[-*]|\d+[.)]|question\s*\d*\s*:)\s*",
                "",
                line,
                flags=re.IGNORECASE,
            ).strip(" \t\"'“”‘’")
            if candidate.endswith("?") and len(candidate.split()) >= 3:
                plain_questions.append(candidate)
        if plain_questions:
            if len(plain_questions) != len(set(plain_questions)):
                raise RuntimeError("Gemma returned duplicate queries.")
            return plain_questions

        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json")
            cleaned = cleaned.removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start < 0 or end <= start:
                raise RuntimeError("Gemma did not return a recognizable query.") from None

            try:
                data = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as exc:
                raise RuntimeError("Gemma did not return a recognizable query.") from exc

        if isinstance(data, list):
            queries = data
        elif isinstance(data, dict):
            queries = data.get("queries")
        else:
            queries = None
        if not isinstance(queries, list):
            raise RuntimeError("Gemma response does not contain a query list.")

        normalized = [
            query.strip() for query in queries if isinstance(query, str) and query.strip()
        ]

        if len(normalized) != len(set(normalized)):
            raise RuntimeError("Gemma returned duplicate queries.")

        return normalized
