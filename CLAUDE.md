# Repository Context

This repository owns evaluation and offline dataset work for the Clarity agent:
evaluation suites and runners, retrieval benchmarks, synthetic-query generation
and filtering, encoder baselines, and bi-encoder fine-tuning.

Read this file first, then `README.md` for commands and `src/clarity_agent_evals/genq/README.md`
for the generation pipeline.

Never add Co-Authored-By or any AI attribution to commits, PRs, or comments.

## The boundary with the application repository

This repository depends on `dsi_clarity_agent`; the application never depends on
this one. The dependency is a `[tool.uv.sources]` path to a sibling checkout, so
both must be cloned side by side under the same parent:

```text
<any parent>/
├── fixtures/                       # shared inputs, committed to neither repo
│   ├── ClarityDictionaryHTML-full/ # the Clarity dictionary; everything derives from it
│   ├── rag/                        # built SQLite indexes
│   └── models/                     # downloaded encoder weights
├── dsi_clarity_agent/
└── dsi_clarity_agent_eval/
```

The two directory names are the repository names, so a default `git clone` of
each lands in the right place. The parent may be renamed or moved freely, because
the dependency path is relative: only the sibling relationship is load-bearing.
`fixtures/` is shared through a directory junction rather
than copied, because the corpus and index are inputs to both repositories and
too large to duplicate per clone.

Keep the path source in `pyproject.toml`. `agent-harness` is also a real package
on public PyPI owned by someone else, so removing the source entry turns an
unresolvable name into a silently installable stranger's package.

Import public names from the application. `retrieval.chunk_models`,
`retrieval.search`, and `retrieval.metadata_extractor` are the intended surface;
reaching for an underscore-prefixed symbol across the boundary couples this
repository to a private detail the application is free to rename.

## Pipeline ownership

The generation pipeline spans both repositories. Stage 1 lives in the
application because the production indexer parses the same HTML.

| Stage | Command | Repository |
|---|---|---|
| 1. Parse HTML into chunks | `agent-harness-genq-parse` | application |
| 1b. Extract table metadata | `agent-harness-metadata-extract` | application |
| 1c. Metadata into chunks | `agent-harness-genq-metadata-convert` | this one |
| 2. Assign leakage-safe splits | `agent-harness-genq-split` | this one |
| 3. Generate synthetic queries | `agent-harness-genq-generate` | this one |
| 4. Filter and review | `agent-harness-genq-filter` | this one |
| 5. Baseline and training | `agent-harness-genq-baseline`, `agent-harness-genq-train` | this one |

```text
clarity_agent_evals/       evaluation execution, assertions, dataset splits, query filters
clarity_agent_evals/genq/  offline synthetic-query generation; never in the request path
evals/                     versioned benchmark cases and committed datasets
```

## Invariants

- Nothing here runs in the agent request path. This code is offline tooling.
- Generation, filtering, and splitting are reproducible: a run is identified by
  `input_corpus_hash`, `output_query_hash`, and `retained_queries_hash`. The
  filter fails closed when a hash does not match its input.
- Splits are assigned by source hash so that a table's chunks cannot straddle
  train and test. Leakage is prevented structurally, not by convention.
- Chunk identity comes from `INDEX_CHUNKER_VERSION` in the application's
  `retrieval/index_contract.py`. While it is unchanged, a rebuilt index
  reproduces byte-identical chunk IDs and existing query datasets stay valid.
  `INDEX_SCHEMA_VERSION` is separate and governs only the SQLite storage layout.
- Retrieval scores are only comparable across runs built from the same chunker
  version. Record it beside any metric.
- Generated artifacts live under ignored `.local/`. Committed datasets live
  under `evals/` and are tracked with Git LFS.
- Never commit credentials, PHI, proprietary schemas or HTML, SQLite indexes,
  sensitive traces, or real query results.

## Evaluation quality

A benchmark that every method solves cannot rank methods. Before trusting a
retrieval number, check that a trivial baseline does not match it: BM25 scores
0.954 MRR on the synthetic queries against a dense encoder's 0.955, because 79%
of those queries name their target table and a table name is unique in the
corpus. The gold benchmark separates methods; the synthetic set is a regression
canary only.

Prefer a small human-verified benchmark over a large generated one. Report the
gold set alongside any synthetic number, never instead of it.

## Working rules

- Keep the application's public API the only coupling; do not vendor its code.
- Update focused tests and the one document that owns a changed contract.
- Record what a run actually did, not what was requested. A report that stores
  the request makes two runs look identical when their numerics differed.
- Do not commit generated artifacts alongside source changes.

## Verification

```powershell
uv run ruff format --check src tests
uv run ruff check src tests
uv run pyright
uv run pytest
```
