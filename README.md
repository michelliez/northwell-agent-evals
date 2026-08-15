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

## Generate a new synthetic evaluation set

The `clarity-evals-generate` command is the portable entry point for both
evaluation-set types. It creates deterministic, leakage-safe manifests before
calling Claude, and writes every generated artifact beneath the selected output
directory. From a fresh clone with the application repository beside this one:

```bash
uv sync
cp .env.example .env
```

Add either `ANTHROPIC_API_KEY` or `AI_HUB_API_KEY` to `.env`. For an
organization-hosted Anthropic-compatible gateway, also set
`ANTHROPIC_BASE_URL` and, when required, `ANTHROPIC_CUSTOM_HEADERS`.

Generate ordinary single-chunk questions from a Stage 1 chunk corpus. By
default, only validation and test chunks are sent to Claude; complete source
documents, rather than individual chunks, are assigned to the 80/10/10 splits:

```bash
uv run clarity-evals-generate non-relationship \
  --chunks /path/to/chunks.jsonl \
  --output-dir .local/custom-evals/non-relationship \
  --splits validation test \
  --queries-per-chunk 2 \
  --items-per-request 10
```

This writes the split and selected corpora under `corpus/`, raw questions under
`queries/`, retained questions and the review ledger under `reviewed/`, and a
top-level `run_report.json`. Add `train` to `--splits` when building training
data. Use `--limit 20` for an inexpensive smoke run, or `--skip-filter` only
when raw, unreviewed questions are explicitly wanted.

Generate relationship questions from an application index containing the
`table_relationships` graph:

```bash
uv run clarity-evals-generate relationship \
  --db ../dsi_clarity_agent/.local/rag/index-relationships-v5.sqlite \
  --output-dir .local/custom-evals/relationship \
  --train-size 8000 \
  --validation-size 1000 \
  --test-size 1000 \
  --splits validation test \
  --items-per-request 10
```

Relationship pairs are kept intact and assigned to exactly one split. Each
successful pair produces one deterministic identifier-aware question and one
Claude-generated identifier-free question. Add `--manifest-only` to create and
inspect the three leakage-safe manifests without making API calls. Add `train`
to `--splits` only when training questions are needed. Keep test output sealed
until retrieval settings have been selected using validation data.

Both modes accept `--model`, `--batch-size`, `--limit`, token bounds, and a
stable generation seed. Run either subcommand with `--help` for the complete
contract. Private source corpora, SQLite indexes, API keys, and generated
outputs remain ignored by Git.

## Relationship evaluation datasets

Build bounded validation and test samples from the application's resolved
relationship graph. Exact column edges are deduplicated, all edges for an
unordered table pair stay together, and the pair is assigned to an 80/10/10
split by a seeded SHA-256 threshold. This prevents a relationship pair from
appearing in more than one ML split while leaving the documentation corpus
available to RAG as intended.

```bash
uv run agent-harness-relationship-dataset \
  --db ../dsi_clarity_agent/.local/rag/index-relationships-v5.sqlite \
  --output-dir .local/relationships \
  --seed clarity-relationships-v1 \
  --train-size 8000 \
  --validation-size 1000 \
  --test-size 1000
```

The command writes `train_relationships.jsonl`,
`validation_relationships.jsonl`, `test_relationships.jsonl`, and
`split_report.json`. Sizes count table-pair
groups; a group can contain multiple column edges for a composite relationship.
The test sample should remain sealed until validation-time retrieval settings
are frozen.

Generate the two fixed evaluation question types for validation first:

```bash
uv run agent-harness-relationship-genq \
  .local/relationships/validation_relationships.jsonl \
  --output .local/relationships/queries/validation.generated.jsonl \
  --report .local/relationships/queries/validation.generation-report.json \
  --model claude-haiku-4-5-20251001 \
  --relationships-per-request 10 \
  --batch-size 20 \
  --max-input-tokens 400 \
  --max-query-tokens 64
```

Every pair produces one deterministic `identifier_aware` question that names
only its anchor table and one Claude-generated `identifier_free` business
question that contains none of the documented table or column identifiers.
The generator preserves the complete gold edge list and table contexts on both
output records. Invalid structured responses and identifier leakage are
retried up to three times. A batch that
remains invalid is retried one pair at a time; only pairs that still fail are
skipped and recorded under `failed_relationship_group_ids` in the report.
Transport and API failures still stop the run rather than silently dropping a
large range. Generate test questions only after validation-time retrieval
settings are frozen.

The application repository retains runtime retrieval, API, UI, policy, SQL, and
LangGraph code. Its metadata extraction commands can produce input artifacts for
the workflows in this repository.
