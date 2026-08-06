from __future__ import annotations

from clarity_agent_evals.genq.claude_query_generator import SYSTEM_PROMPT as CLAUDE_PROMPT
from clarity_agent_evals.genq.prompts import QUERY_GENERATION_SYSTEM_PROMPT
from clarity_agent_evals.genq.qwen_query_generator import SYSTEM_PROMPT as QWEN_PROMPT


def test_every_provider_uses_the_same_prompt_object() -> None:
    """A model comparison is only valid if the instruction is held fixed.

    Identity rather than equality: two equal copies can drift on the next edit,
    which is the failure this consolidation exists to prevent.
    """
    assert CLAUDE_PROMPT is QUERY_GENERATION_SYSTEM_PROMPT
    assert QWEN_PROMPT is QUERY_GENERATION_SYSTEM_PROMPT


def test_prompt_states_the_analyst_framing_and_anti_copying_rule() -> None:
    """The two properties the measured variants were trying to buy.

    Both rewrites lowered passage copying only by collapsing question variety,
    so this prompt keeps the softer instruction deliberately.
    """
    prompt = QUERY_GENERATION_SYSTEM_PROMPT
    assert "healthcare data analyst" in prompt
    assert "avoid copying long phrases" in prompt
    assert "Vary wording and intent" in prompt
    assert "Do not answer the questions" in prompt
