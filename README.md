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
│   ├── embeddings/                 # dense FAISS artifacts (corpus.faiss + mapping + metadata)
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

## Retrieval benchmark

The reviewed 85-query benchmark lives in `evals/retrieval/benchmark/`. One
runner measures four retriever arms against any compatible index; the dense
and hybrid arms need the genq group and a FAISS artifact directory:

```bash
uv run python -m clarity_agent_evals.runner --suite retrieval \
    --retriever fts --db ../fixtures/rag/<index>.sqlite --k 5 10 20
# other arms: fts+relationships | dense | hybrid
#   dense/hybrid add: --dense-index-dir ../fixtures/embeddings/dense-corpus-qwen06b
```

The frozen baseline (2026-08-10) is recorded in the application repository at
`src/retrieval/README.md`: hybrid RRF wins overall hit@5 0.622 with zero
identifier-bucket regressions. Rerun all arms before changing any retrieval
configuration.

Reconstructing the shared inputs without a GPU:

- **Index**: `evals/retrieval/synthetic/qwen-combined-expansion-v1.jsonl`
  (LFS) is the frozen doc2query corpus — 83,738 generated texts over 31,060
  table chunks, questions + analyst summaries merged. Rebuild the production
  index from a finished v5 index in minutes, CPU-only, with the application's
  `agent-harness-rag-expansion-migrate <v5.sqlite> <corpus.jsonl> <v6.sqlite>`.
- **Dense artifact**: `agent-harness-genq-dense-corpus` re-embeds all chunks
  (~2.5 h CUDA, overnight on Apple MPS); with no GPU, obtain the artifact
  directory out of band. It must be rebuilt whenever the index it was built
  from changes.

Versioned cases live in `evals/`. Generated reports, models, indexes, and private
corpus derivatives belong under `.local/` and are not committed.

The application repository retains runtime retrieval, API, UI, policy, SQL, and
LangGraph code. Its metadata extraction commands can produce input artifacts for
the workflows in this repository.
