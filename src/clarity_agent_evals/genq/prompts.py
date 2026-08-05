"""The shared instruction given to every synthetic-query generator.

One constant, imported by both providers, so that comparing Claude against a
local model compares *models* rather than two prompts that drifted apart. It
previously existed as byte-identical copies in each provider module, which is a
difference waiting to happen.

Three variants were measured against this one on 469 identical table passages,
scored on how much of a query is lifted from its passage (lower is better) and
how many distinct question openings appear (higher is better):

    variant                       copy%   openers
    shipped (this prompt)         55.1%       425
    "you are searching for it"    47.3%       173
    "vary the intent: a/b/c/d"    41.2%       117
    Claude Haiku, same prompt     46.5%       344

Both rewrites bought a lower copy rate by collapsing question variety: naming a
viewpoint produced one kind of question, and enumerating intents produced one
templated question per intent (with 14 exact duplicates). Under prompt pressure
these two properties move in opposite directions, because any instruction
specific enough to stop restating is specific enough to dictate form.

Haiku holds both on this same prompt, so the gap is model capability rather than
instruction quality, and further prompt tuning is more likely to game the proxy
than to close it. Keep this prompt; if evaluation-set quality matters more than
API cost, change the model rather than the words.
"""

from __future__ import annotations

QUERY_GENERATION_SYSTEM_PROMPT = """\
You create realistic semantic-search queries for Epic Clarity data-dictionary passages.
Write questions that a healthcare data analyst, report developer, or SQL developer might ask.
Use only facts supported by the supplied passage. Do not answer the questions.
Vary wording and intent, avoid copying long phrases, and preserve important table or column names
only when a real user would plausibly include them.
"""

__all__ = ["QUERY_GENERATION_SYSTEM_PROMPT"]
