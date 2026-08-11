# Retrieval Evaluation Data

This directory separates the reviewed retrieval benchmark from generated query
data.

- `benchmark/` is tracked. It holds the small, hand-authored query, document
  qrel, chunk qrel, and document catalog files used by
  `agent-harness-eval --suite retrieval`.
- `generated/` is git-ignored. It holds local query-generation artifacts whose
  records reproduce descriptions from the proprietary Clarity corpus and must
  never be committed.

## Running the benchmark

```powershell
uv run agent-harness-eval --suite retrieval --k 5 10
```

The index defaults to `RAG_DB_PATH` when set, then `.local/rag/index.sqlite`.
Override either value with `--db`, and point at another benchmark directory
with `--benchmark-dir`.

Validate the qrels without running retrieval:

```powershell
uv run python -m clarity_agent_evals.retrieval_gold
```

Reports are written to `.local/evals/retrieval-evaluation-<timestamp>.json`.

## Reviewed failure buckets

The tracked benchmark uses analyst-phrased questions rather than phrases copied
from HTML descriptions. Its 260 queries cover five retrieval failure buckets:

- `named_table_lookup`: a known table must yield its purpose or grain;
- `named_column_schema`: known fields must yield types, meanings, or key roles;
- `business_concept_discovery`: business language must find the right tables;
- `cross_table_synthesis`: evidence must be compared across documents;
- `negative_unsupported`: the corpus does not support the requested answer.

The query file records `failure_bucket`, and reports include document- and
chunk-level metrics for each bucket. Of the 435 chunk qrels, those for Q01–Q85
are manually reviewed against the column-level index; those for Q86–Q260 are
model-authored and mechanically verified to resolve (heading, category, and
required terms matched against the live index). Do not regenerate qrels from
source descriptions or mechanically patch old heading paths.

## Coverage

The first 24 queries covered 17 documents that were all the head or tail of an
alphabetical directory listing — `A0H_*`, `AAG_*`, `ABF_*`, `ABN_DOCUMENT_ID`,
and `ZC_XPR_*`. Those are Hyperspace-access maintenance feeds and code lookups
that no analyst queries, so the benchmark measured retrieval on the least
representative 0.04% of the corpus.

Queries Q25–Q50 add twelve tables analysts actually use: `PAT_ENC`,
`PAT_ENC_HSP`, `CLARITY_ADT`, `ORDER_PROC`, `ORDER_MED`, `PATIENT`,
`CLARITY_SER`, `CLARITY_DEP`, `HSP_ACCOUNT`, `CLARITY_EAP`, `PAT_ENC_DX`, and
`PROBLEM_LIST`. Q51–Q85 extend the same high-traffic axis (reviewed 2026-08-10).

Q86–Q260 are the stratified extension (2026-08-11): a coverage audit showed the
first 85 queries touched 14 of 2,498 table-name families while oversampling the
alphabetical head of the corpus, and a power analysis showed ~200 scored
queries were needed to certify retrieval-arm differences of ~8 points of hit@5.
The 175 added queries target 23 previously uncovered families (surgical,
pharmacy, lab, MyChart, home health, billing, and others), no family exceeding
15% of the additions. Provenance: model-generated from dictionary content,
spot-checked rather than per-query reviewed. Representativeness of the tables
still matters more than query count — the open gap is validating the query
distribution against real analyst demand.

## Judgment scope and what each metric means

`judgment_scope` records how far judging actually went, and it changes which
metrics are reported:

| Scope | Reported | Suppressed |
|---|---|---|
| `corpus_complete` | everything | — |
| `positive_only` | `recall@k`, `hit@k`, `mrr` | `precision@k`, `ndcg@k` |

Recall and hit survive incomplete judgments: a known positive either appears in
the ranking or it does not. Precision and nDCG cannot, because both must treat
every unjudged result as irrelevant, and across 40,551 documents that assumption
is unearned.

`business_concept_discovery` queries are judged `positive_only` by necessity —
asserting that no other table could answer "which table holds the people who
deliver care" would mean judging the whole corpus. Claiming `corpus_complete`
there would silently inflate precision.

Per-bucket rows print `SCORED/TOTAL`. When those differ, some queries in the
bucket contributed to no average — read the total as coverage, never as sample
size.

## Comparability

The qrels are **portable**: they name documents semantically
(`epic_clarity:<object_type>:<OBJECT_NAME>`) and are resolved to the active
index's runtime chunk and document IDs at evaluation time. The same benchmark
therefore runs against any index built from the same corpus.

Every report records `index_version` and `chunker_version`, read from the index
being evaluated rather than from the current source. **Numbers are only
comparable within one `chunker_version`** — changing chunk boundaries changes
what counts as a relevant chunk. `CHUNKER_VERSION` is currently
`section-table-genq-columns-v5` (`src/retrieval/index_contract.py`) and must be
incremented whenever chunk boundaries change.

Reports also carry `judgment_complete`, `unjudged_document_total`, and
`unjudged_chunk_total`. Treat recall as a lower bound whenever judgment is
incomplete.

## Synthetic query generation

Synthetic queries come from the GenQ pipeline: parse the HTML corpus, split it,
generate queries with Claude, filter them, then score retrieval with a FAISS
baseline. It spans three packages by concern — parsing in `retrieval/`,
splitting and filtering in `evals/`, generation and the baseline in
`clarity_agent_evals/genq/`. See [that README](../../src/clarity_agent_evals/genq/README.md).

```powershell
uv run agent-harness-genq-parse <HTML_PATH>
uv run agent-harness-genq-split
uv run agent-harness-genq-generate
uv run agent-harness-genq-filter
uv run --group genq agent-harness-genq-baseline
```

Only the FAISS baseline needs the optional `genq` dependency group:

```powershell
uv sync --group genq
```

GenQ splits the corpus before generating, so held-out evaluation chunks cannot
also become training pairs. It is an offline tool and must never enter the
request path.

### The committed table-level dataset

`synthetic/` holds one complete generation run over the table corpus, so
fine-tuning does not require re-running the pipeline. The `.jsonl` files are Git
LFS; the `*_report.json` files beside them are plain git, because their hashes
are the point.

| file | rows | what it is |
|---|---|---|
| `table_chunks.jsonl` | 39,392 | Stage 2 passages, one per table, deduplicated on description |
| `table_queries_raw.jsonl` | 62,120 | Qwen3-4B output before filtering |
| `table_queries_retained.jsonl` | 62,048 | passed Stage 4; **this is the training input** |
| `table_queries_reviews.jsonl` | 62,120 | every decision with reason codes, including the 30 rejects and 42 flagged |

Retained by split: 49,706 train, 6,140 validation, 6,202 test.

Training needs **both** files, because a query record carries
`relevant_chunk_id` rather than passage text:

```powershell
uv run --group genq agent-harness-genq-train Qwen/Qwen3-Embedding-0.6B `
  --queries evals/retrieval/synthetic/table_queries_retained.jsonl `
  --chunks evals/retrieval/synthetic/table_chunks.jsonl `
  --output-dir .local/models/qwen-emb-tuned
```

Provenance chains through the reports: generation records
`input_corpus_hash 41391515…` over `table_chunks.jsonl` and
`output_query_hash 2ed77433…`; the filter consumes that same
`raw_queries_hash 2ed77433…` and produces
`retained_queries_hash e66cfc30…`. A mismatch means the files no longer belong
to each other and Stage 4 fails closed rather than filtering the wrong pair.

Two things about this run differ from the defaults and matter when comparing
against another:

- **`--min-passage-chars 40`, not the default 100.** The length distribution is
  bimodal, and 100 sits in the middle of it: 8,332 passages are `Table: NAME`
  with no description at all, while 5,666 in the 40–99 band carry real ones
  ("The status history of an ABN form"). The default silently discarded that
  second group. 31,060 of 39,392 passages are eligible at 40.
- **The queries cannot be regenerated.** Chunk corpora are deterministic, since
  `index_contract.py` pins `INDEX_CHUNKER_VERSION`, but these are LLM samples.
  The seed makes them repeatable on one GPU; CUDA kernel nondeterminism and a
  different accelerator do not reproduce them. That is why they are committed
  rather than left to a re-run.

Retention was 99.88%, which measures the absence of verbatim copying and
per-chunk duplication — not diversity or grounding. Notably, 32 of the 42
flagged queries invented specifics absent from the passage (dates, department
names, status values); the gate caught them only because they had *no* lexical
overlap, so partial fabrication is still present and unmeasured. Treat the
validation and test slices with that in mind: a hallucinated query makes its own
gold chunk unfindable and depresses measured retrieval unfairly.

The reviewed benchmark above and the GenQ pipeline measure different things at
different granularity: the benchmark is a small hand-authored document-level
qrel set, GenQ is a large synthetic chunk-level set. Do not compare their
numbers to each other.
