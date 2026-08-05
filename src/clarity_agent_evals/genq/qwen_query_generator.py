"""Local Qwen3 provider for offline synthetic-query generation.

This is the no-API-cost counterpart to ``claude_query_generator``. It implements
the same ``QueryGenerator`` protocol, so ``query_generation.generate_queries``
drives it unchanged and every provenance hash, split assignment, and report field
keeps its meaning.

Why transformers rather than vLLM: vLLM has no Windows support, and this runs on
a Windows workstation. Batched ``model.generate`` on a single GPU is enough for a
job whose passages are ~100 tokens and whose outputs are ~40.

Nothing here runs in the request path.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from clarity_agent_evals.genq.prompts import QUERY_GENERATION_SYSTEM_PROMPT

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_MAX_RETRIES = 2
LOGGER = logging.getLogger(__name__)

# Identical to the Claude provider's by construction, not by copy. A local model
# is only a fair substitute if it is given the same instruction.
SYSTEM_PROMPT = QUERY_GENERATION_SYSTEM_PROMPT

# The model is asked for a bare JSON array. Anything it emits around that array
# (a stray sentence, a markdown fence) is tolerated by decoding from each "["
# until one yields a complete value, rather than by demanding a clean response.
#
# This is a parser rather than a regular expression because matching brackets is
# not a regular language. The previous `\[.*?\]` truncated at the first "]" it
# met, so a query quoting a bracketed term -- "the Carryover Mapping [HMM]
# database", ordinary in Epic documentation -- turned valid JSON into a fragment
# and failed every retry identically. `raw_decode` tracks string state, so
# brackets inside a query are just characters.
_DECODER = json.JSONDecoder()


class QwenQueryGenerator:
    """Generate structured query lists locally with a Qwen3 instruct model.

    Unlike the batched Claude path, batching here carries no alignment risk. Each
    passage keeps its own prompt and the batch is a tensor dimension, so output
    row *i* is by construction the answer to input row *i*. There is no
    model-supplied index to mistrust.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_MODEL,
        *,
        device: str = "auto",
        load_in_4bit: bool = True,
        batch_size: int = 16,
        max_retries: int = DEFAULT_MAX_RETRIES,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        # Declared as Any because the attribute holds either a transformers model
        # or a test double, and narrowing it to one defeats both.
        self._model: Any
        self._tokenizer: Any

        if model is not None and tokenizer is not None:
            self._model = model
            self._tokenizer = tokenizer
            self._torch = None
            self.device_name = device
            return

        try:
            import torch  # pyright: ignore[reportMissingImports]
            from transformers import (  # pyright: ignore[reportMissingImports]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Local generation dependencies are missing. Run `uv sync --group genq`."
            ) from exc

        self._torch = torch
        self.device_name = self._resolve_device(torch, device)
        quantization = self._quantization_config(load_in_4bit)
        LOGGER.info(
            "Loading %s on %s (%s)",
            model_name,
            self.device_name,
            "4-bit NF4" if quantization else "bf16",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Decoder-only batched generation requires left padding: right padding
        # would place pad tokens between the prompt and the first generated
        # token, so short prompts in a batch would decode from padding.
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=self.device_name if self.device_name != "cpu" else None,
            quantization_config=quantization,
        )
        self._model.eval()
        if quantization is None and self.device_name != "cpu":
            self._model.to(self.device_name)

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> str:
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if requested != "auto":
            return requested
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _quantization_config(self, load_in_4bit: bool) -> Any | None:
        """Build the NF4 config, or None when 4-bit is unavailable or unwanted.

        4-bit matters on a 10 GB card: a 4B model is ~8 GB in bf16, leaving almost
        no KV cache and forcing tiny batches, versus ~2.5 GB at NF4.
        """
        if not load_in_4bit or self.device_name != "cuda":
            return None
        try:
            import bitsandbytes  # noqa: F401  # pyright: ignore[reportMissingImports]
            from transformers import BitsAndBytesConfig  # pyright: ignore[reportMissingImports]
        except ImportError:
            LOGGER.warning("bitsandbytes is unavailable; loading in bf16 instead of 4-bit")
            return None
        assert self._torch is not None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self._torch.bfloat16,
        )

    def _build_prompt(self, passage: str, queries_per_passage: int) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Generate exactly {queries_per_passage} distinct search questions "
                    f"for this passage.\n\n{passage}\n\n"
                    f"Respond with ONLY a JSON array of exactly {queries_per_passage} "
                    'strings, for example ["first question?", "second question?"]. '
                    "No prose, no markdown, no keys."
                ),
            },
        ]
        try:
            # Qwen3 emits <think> blocks by default. For a one-sentence paraphrase
            # that reasoning is pure cost, and at corpus scale it dominates runtime.
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            # Instruct-only checkpoints reject the flag because they never think.
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _decode_batch(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        top_p: float,
        temperature: float,
    ) -> list[str]:
        assert self._torch is not None
        inputs = self._tokenizer(
            list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(self._model.device)
        with self._torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)

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
        """Return exactly ``queries_per_passage`` queries for every passage."""
        if self._torch is not None:
            self._torch.manual_seed(seed)
        max_passage_chars = max_input_tokens * 4
        # `generate` runs until every sequence in the batch stops, so one rambling
        # row makes the whole batch pay this ceiling. Measured emissions are
        # 33-65 tokens for two queries, so the budget is the queries' own token
        # allowance plus room for JSON punctuation -- not a multiple of it.
        # Overrunning costs a retry; over-provisioning costs every batch.
        max_new_tokens = min(1024, 24 + queries_per_passage * max_query_tokens)

        results: list[list[str]] = []
        for start in range(0, len(passages), self.batch_size):
            group = [
                passage[:max_passage_chars] for passage in passages[start : start + self.batch_size]
            ]
            LOGGER.info(
                "Generating passages %d-%d/%d", start + 1, start + len(group), len(passages)
            )
            pending = list(enumerate(group))
            decoded: dict[int, list[str]] = {}
            for attempt in range(self.max_retries + 1):
                if not pending:
                    break
                prompts = [self._build_prompt(text, queries_per_passage) for _, text in pending]
                raw = self._decode_batch(
                    prompts,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    # A repeat of a failed parse needs different sampling, or it
                    # reproduces the same malformed output.
                    temperature=0.7 + 0.2 * attempt,
                )
                still_pending: list[tuple[int, str]] = []
                for (offset, text), completion in zip(pending, raw, strict=True):
                    parsed = _parse_queries(completion, queries_per_passage)
                    if parsed is None:
                        still_pending.append((offset, text))
                        continue
                    decoded[offset] = parsed
                pending = still_pending
                if pending:
                    LOGGER.warning(
                        "Retrying %d passage(s) after unparsable output (attempt %d/%d)",
                        len(pending),
                        attempt + 1,
                        self.max_retries + 1,
                    )
            if pending:
                raise RuntimeError(
                    f"{len(pending)} passage(s) produced no valid {queries_per_passage}-query "
                    f"JSON array after {self.max_retries + 1} attempts"
                )
            results.extend(decoded[offset] for offset in range(len(group)))
        return results


def _parse_queries(completion: str, expected_count: int) -> list[str] | None:
    """Return exactly ``expected_count`` queries, or None when the reply is unusable.

    Returning None rather than raising lets the caller retry that one passage
    instead of aborting a multi-hour run over a single malformed reply.
    """
    text = completion.strip()
    for start, character in enumerate(text):
        if character != "[":
            continue
        try:
            candidate, _ = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, list):
            continue
        queries = [
            " ".join(item.replace("\t", " ").split()).strip()
            for item in candidate
            if isinstance(item, str) and item.strip()
        ]
        if len(queries) >= expected_count:
            return queries[:expected_count]
    return None
