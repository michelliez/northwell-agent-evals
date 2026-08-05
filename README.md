# Clarity Agent Evaluations

This repository owns evaluation datasets, evaluation runners, retrieval benchmarks,
synthetic-query generation, filtering, baseline embedding experiments, and bi-encoder
training. It intentionally depends on the sibling application repository; the
application does not depend on this repository.

Expected sibling layout:

```text
workspace/
├── clarity_agent_application/
└── clarity_agent_evals/
```

Install and run the lightweight evaluation suite:

```bash
uv sync
uv run agent-harness-eval --suite smoke
uv run agent-harness-eval --suite red_team
uv run agent-harness-eval --suite intent --repetitions 3
```

Install the optional ML dependencies for query generation, FAISS baselines, and
training:

```bash
uv sync --group genq
```

Versioned cases live in `evals/`. Generated reports, models, indexes, and private
corpus derivatives belong under `.local/` and are not committed.

The application repository retains runtime retrieval, API, UI, policy, SQL, and
LangGraph code. Its metadata extraction commands can produce input artifacts for
the workflows in this repository.
