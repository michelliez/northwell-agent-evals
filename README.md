# Clarity Agent Evaluations

This repository owns evaluation datasets, evaluation runners, retrieval benchmarks,
synthetic-query generation, filtering, baseline embedding experiments, and bi-encoder
training. It intentionally depends on the sibling application repository; the
application does not depend on this repository.

Expected sibling layout. The two directory names are the repository names, so a
default `git clone` of each produces this without renaming anything. The parent
may be called anything and may be moved, because the dependency path is relative
-- only the sibling relationship matters:

```text
<any parent>/
├── fixtures/                       # shared inputs, committed to neither repo
│   ├── ClarityDictionaryHTML-full/ # the Clarity dictionary; everything derives from it
│   ├── rag/                        # built SQLite indexes
│   └── models/                     # downloaded encoder weights
├── dsi_clarity_agent/
└── dsi_clarity_agent_eval/
```

`fixtures/` sits outside both checkouts because the corpus and the index are
inputs to both repositories and are far too large to duplicate per clone. Point
each repository at it with a directory junction, which needs no administrator
rights on Windows:

```powershell
cmd /c mklink /J "var\rag" "..\fixtures\rag"
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
