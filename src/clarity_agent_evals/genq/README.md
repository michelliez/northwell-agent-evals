# Synthetic Query Generation

Offline tooling that prepares Epic Clarity documentation for a future
domain-adapted semantic retriever. **None of it runs in the request path.**

The pipeline is deliberately split across three packages by what each stage is
for, not by when it was written:

| Stage | Module | Command |
|---|---|---|
| 1. Parse HTML into column chunks | `retrieval/column_parser.py` | `agent-harness-genq-parse` |
| 2. Assign leakage-safe splits | `evals/dataset_split.py` | `agent-harness-genq-split` |
| 3. Generate synthetic queries (Claude) | `clarity_agent_evals/genq/query_generation.py` | `agent-harness-genq-generate` |
| 3b. Generate synthetic queries (local) | `clarity_agent_evals/genq/qwen_query_generator.py` | `agent-harness-genq-generate --provider qwen` |
| 4. Filter and review them | `evals/query_filter.py` | `agent-harness-genq-filter` |
| 5. Encoder/FAISS baseline | `clarity_agent_evals/genq/baseline_faiss.py` | `agent-harness-genq-baseline` |

Two extraction paths feed stage 3, one per granularity:

| Granularity | Module | Command | Units |
|---|---|---|---|
| Table (one per document) | `retrieval/metadata_extractor.py` + `genq/metadata_to_chunks.py` | `agent-harness-metadata-extract`, `agent-harness-genq-metadata-convert` | 40,551 |
| Column | `clarity_agent_evals/genq/column_extractor.py` | `agent-harness-genq-columns` | 195,350 after dedup |

Stage 1 is **also production code**: `retrieval/indexer.py` calls
`parse_column_records` to build the column chunks in the live SQLite index.
Changing it changes the production index — bump `CHUNKER_VERSION` in
`retrieval/index_contract.py` and rebuild.

## Table-level embedding input

Before embedding individual columns, the first dense baseline operates at table
level. `retrieval/metadata_extractor.py` reads the production SQLite index in
read-only mode and writes one deterministic record for every table with a
metadata chunk:

```bash
uv run agent-harness-metadata-extract var/rag/index-genq-columns-v5.sqlite \
  --output .local/embeddings/table-metadata/records.jsonl \
  --report .local/embeddings/table-metadata/extraction-report.json
```

Each encoder input is `Table: <name>\nDescription: <description>` when a
description exists, or `Table: <name>` as an explicit fallback. Records retain
the description status, document ID, metadata chunk ID, source and text hashes,
index version, and extractor version. Missing or duplicate metadata chunks fail
the run. This stage performs no model loading and creates no vector index.

### Collapsing shared descriptions

Epic reuses boilerplate across families of tables: 200 tables carry one identical
deprecation notice, 127 carry another, and 33 `*_DELETE_CT` tables share a single
sentence. Because the description is the only text that distinguishes one table
passage from another, those tables are not separable by any encoder — the
generator would pay for 200 requests to learn one fact, and contrastive training
would be asked to push apart passages that differ only in the table name.

`agent-harness-genq-metadata-convert --dedup-descriptions` keeps the first record
for each distinct description and drops the rest. Tables with no description are
never collapsed, because the `Table: <name>` fallback shares no text. The
extractor emits rows ordered by source path, so the surviving representative is
deterministic. Deduplication runs before `--limit`, so `--limit N` yields N
distinct passages rather than N raw rows.

Over the full 40,551-table corpus this drops 1,159 records across 425 duplicate
groups, leaving 25,394 chunks eligible for generation instead of 26,337. The
conversion report records `dedup_description_count` and `dedup_group_count`.

## Column-level embedding input

Table metadata answers "which table holds X". It cannot answer "which column
stores the admission timestamp", because a table's metadata chunk never names
its columns. `clarity_agent_evals/genq/column_extractor.py` reads the `column_info`
chunks the indexer already wrote and prepares them the same way:

```bash
uv run agent-harness-genq-columns var/rag/index-genq-columns-v5.sqlite \
  --output .local/genq/column-corpus/columns_dedup.jsonl \
  --report .local/genq/column-corpus/extraction_report.json
```

Two reductions apply, for the same reason they apply at table level.

**Scaffolding is stripped.** The stored chunk is
`Table A0H_MAP. Column CID. INI: A0H. Item: 11. Type: NUMERIC (18,0).
Deprecated?: No. Discontinued?: No. Preserved?: No. Character Replacement?: No.
EHI Status: Not Exported. Description: ...`. Everything between the type and the
description is near-identical across all 356,056 columns, so embedding it drags
the whole corpus toward one dense region. The encoder input becomes
`Table: <t>\nColumn: <c>\nType: <type>\nDescription: <description>`.

**Shared descriptions collapse.** 41.6% of column descriptions are exact
duplicates of another column's: one sentence about Community IDs appears on
14,638 distinct columns. Exact *chunk* hashing catches almost none of this
(0.5%) because the table and column names differ; the key must be the
description alone.

Over the full index this turns 356,056 column chunks into **195,350**, after
also skipping 15,456 descriptions shorter than `--min-description-chars` (40)
and 349 with none. `--no-dedup` keeps every column. As at table level, dedup
runs before `--limit`.

`stratified_sample` draws a sample spread across Chronicles table families
rather than uniformly, so a small Claude sample is representative of the schema
instead of its largest table families.

The first dense retrieval run consumes that artifact directly and evaluates it
against the same reviewed 50-query document benchmark as FTS:

```bash
uv run --group genq agent-harness-genq-baseline \
  .local/embeddings/table-metadata/records.jsonl \
  --benchmark-dir evals/retrieval/benchmark \
  --output-dir .local/embeddings/table-metadata/minilm-gold \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 128 --top-k 10
```

This mode indexes all records by default and writes `tables.faiss`,
`table_mapping.jsonl`, `index_metadata.json`, and `gold_evaluation.json`. The
evaluation resolves portable catalog paths to extracted document IDs, reports
document MRR/Hit/Recall/nDCG at 1, 5, and 10 plus failure-bucket breakdowns, and
retains the ranked hits for inspection. Positive-only qrels do not produce
precision or nDCG; unsupported queries are diagnostic because this baseline has
no abstention threshold.

Stages 2 and 4 live in `evals/` because leakage-safe splits and query quality
gates are evaluation concerns whether or not a model is ever trained.

Generated artifacts are written below `.local/`, which is git-ignored because
its contents derive from proprietary Epic HTML.

```text
.local/
├── genq/corpus/     parsed chunks and leakage-safe splits
├── queries/         raw and reviewed synthetic queries
├── embeddings/      experimental FAISS indexes and evaluations
└── rag/index.sqlite production SQLite retrieval index
```

## Dependencies

Stages 1–4 need nothing beyond the core dependencies. Only stage 5 needs the
optional group:

```bash
uv sync --group genq
```

## Stage 1: Parse HTML into column chunks

One chunk per column definition, rather than several columns per passage. Other
chunk types are table metadata, primary keys, foreign keys, index information,
and relationships. Each chunk carries a readable logical `chunk_id`, a
source-file hash, and a text hash.

```bash
uv run agent-harness-genq-parse ../ClarityDictionaryHTML-full \
  --limit 100 \
  --output .local/genq/corpus/chunks.jsonl \
  --report .local/genq/corpus/parse_report.json
```

The 100-file proof of concept produced 1,613 chunks with no parser warnings.

## Stage 2: Leakage-safe splits

Whole source files are assigned to `train`, `validation`, or `test` by seeded
SHA-256 threshold — so runs are reproducible and adding files never moves an
existing file between splits. Ratios are approximately 80/10/10.

```bash
uv run agent-harness-genq-split .local/genq/corpus/chunks.jsonl \
  --output .local/genq/corpus/chunks_with_splits.jsonl \
  --report .local/genq/corpus/split_report.json \
  --seed epic-genq-v1
```

## Stage 3: Generate synthetic queries

Claude Haiku generates questions from stage 2 passages, one request per passage,
returning an exact count through a forced tool result. The system prompt asks
for questions a healthcare data analyst, report developer, or SQL developer
would ask — the domain fit is the whole point, since a generic web-search query
generator reintroduces exactly the distribution gap this pipeline exists to
close.

```bash
uv run agent-harness-genq-generate .local/genq/corpus/chunks_with_splits.jsonl \
  --provider claude \
  --model claude-haiku-4-5-20251001 \
  --output .local/queries/generated_queries.jsonl \
  --report .local/queries/generation_report.json \
  --queries-per-chunk 5 \
  --batch-size 8
```

The same pipeline can use a locally hosted Gemma model through MLX-LM's
OpenAI-compatible HTTP server. Select it with `--provider gemma`, identify the
loaded model with `--gemma-model`, and set its endpoint with
`--gemma-base-url` (or `GEMMA_BASE_URL`). The default endpoint is
`http://127.0.0.1:8080`. MLX remains isolated from the main project environment;
this process only sends HTTP requests to the already-running local server.
Gemma retries each requested question up to `GENQ_GEMMA_MAX_ATTEMPTS` (default
`5`). If every attempt fails, that chunk is skipped and counted in
`failed_generation_chunk_count` in the generation report.

Uses `ANTHROPIC_API_KEY` or `AI_HUB_API_KEY`, and honors `ANTHROPIC_BASE_URL`
and `ANTHROPIC_CUSTOM_HEADERS`. `--max-input-tokens` bounds the passage using
the four-characters-per-token approximation.

The API exposes no seeded sampling, so `--seed` records dataset provenance but
does not make output byte-reproducible. The generated JSONL is the artifact you
keep and version — it is written atomically with content hashes and stamps the
generator model onto every query, so the *dataset* is reproducible even though
the *generator* is not.

`QueryGenerator` is a Protocol, so a local generator can be added later as a
drop-in without touching the pipeline.

### Cost, throughput, and credentials

`--batch-size` controls log granularity and nothing else. The flag that changes
cost is `--passages-per-request`.

Measured against `/v1/messages/count_tokens` on the real corpus, a one-passage
request costs **~748 input tokens of which 692 is fixed** — the tool-use schema
scaffolding the API injects — against a mean passage of only ~45 tokens. That
prologue is re-sent once per passage, so 94% of input spend buys nothing.
Grouping passages into one request amortizes it:

| `--passages-per-request` | Input tokens/passage | 40,551 tables @ 2 queries |
|---|---|---|
| 1 (default, reproduces earlier runs) | 748 | $39.11 |
| 5 | 228 | $18.89 |
| 10 | 135 | $15.01 |
| 20 | 97 | $13.19 |

Returns flatten past 10 because output tokens, which batching cannot touch,
start to dominate. Larger batches also carry more risk: alignment then depends
on a model-supplied `passage_index`, and a wrong index would attach queries to
the wrong table. `_extract_batched_queries` therefore verifies that every
expected index appears exactly once and fails the batch otherwise, because that
corruption is silent and nothing downstream inspects it.

Wall-clock is bounded by total output tokens rather than by request count, so
batching cuts cost far more than it cuts time; only concurrency addresses the
latter. Nothing is written until the final atomic write — an interrupted run has
to start over.

The Message Batches API would halve both cost and wall-clock, but the AI-hub
gateway in `ANTHROPIC_BASE_URL` does not proxy `/v1/messages/batches` (it
returns 404 while `/v1/messages` and `/v1/models` succeed). Until that endpoint
is proxied, or a direct Anthropic key is used, the sequential path is the only
one available.

`main` loads `.env` through `python-dotenv`, the same way `agent_host.config`
does, so credentials can live in one place; already-exported environment
variables still win. The gateway advertises `@`-versioned model IDs
(`claude-haiku-4-5@20251001`), so pass no `--model` flag and let
`DEFAULT_CLAUDE_MODEL` (`claude-haiku-4-5-20251001`) apply — that is the string
the completed 500-table run used.

## Stage 3b: Local generation with Qwen3

Claude is worth its cost on table descriptions, which are the passages real
users search for and where paraphrase quality matters. It is poor value on
195,350 column chunks, whose descriptions are single declarative sentences —
near the floor of what an instruct model has to do. A local model covers those
for no API spend.

```bash
uv run agent-harness-genq-generate \
  .local/genq/column-corpus/columns_dedup.jsonl \
  --provider qwen \
  --output .local/queries/columns/generated_queries.jsonl \
  --report .local/queries/columns/generation_report.json \
  --queries-per-chunk 2 --batch-size 16
```

`QwenQueryGenerator` implements the same `QueryGenerator` Protocol, so
`generate_queries` drives it unchanged: identity, duplicate accounting, split
provenance, hashes, and the report all keep their meaning, and
`GeneratedQueryRecord.generator_model` records which model wrote each query.

Three properties differ from the API path:

- **Batching is alignment-safe.** Each passage keeps its own prompt and the
  batch is a tensor dimension, so output row *i* answers input row *i* by
  construction. There is no model-supplied index to mistrust.
- **Output is parsed, not schema-forced.** There is no `tool_choice` locally, so
  the reply is prompted as a bare JSON array and recovered from surrounding
  prose or markdown fences. A passage whose reply will not parse is re-sampled
  at a higher temperature up to `--max-retries`; only that passage is resent,
  not the batch. Persistent failure raises rather than emitting a short list.
- **Thinking mode must stay off.** Qwen3 emits `<think>` blocks by default; for
  a one-sentence paraphrase repeated 195,350 times that reasoning dominates
  runtime. `_build_prompt` passes `enable_thinking=False`, falling back for
  instruct-only checkpoints that reject the flag.

Loading is 4-bit NF4 via bitsandbytes by default. On a 10 GB card a 4B model is
~8 GB in bf16, which leaves almost no KV cache and forces tiny batches, versus
~2.5 GB quantized. `--no-4bit` opts out.

### Batch size has a cliff, not a curve

Throughput improves with batch size up to a point and then collapses. Measured
on 192 real column chunks, RTX 3080 (10 GB), 4-bit NF4:

| `--batch-size` | Wall | Per chunk | Peak VRAM | |
|---|---|---|---|---|
| 8 | 158 s | 0.82 s | — | |
| 16 | 88 s | 0.46 s | — | |
| **32** | **66 s** | **0.34 s** | **6.9 GB** | best |
| 48 | far worse | — | 9.9 GB | saturates |
| 96 | — | — | — | produced no output |

The collapse tracks VRAM, not sequence length. Batch 32 peaks at 6.9 GB with
headroom; batch 48 climbs to 9.9 GB of 10 GB as the KV cache grows during
generation, and Windows' driver degrades rather than failing cleanly, so the run
still finishes but far slower. Batch 96 does not finish at all.

**Use `--batch-size 32` on a 10 GB card**, and re-measure rather than
extrapolating on a different card — the knee is wherever the KV cache stops
fitting, which depends on VRAM, quantization, and `--max-query-tokens`.

Keep `--max-query-tokens` honest for the same reason: it sets `max_new_tokens`,
which bounds KV-cache growth. Overrunning it costs one passage a retry;
over-provisioning it consumes VRAM that batch size could have used.

Mixing generators introduces a confound: if every table query is Claude's and
every column query is Qwen's, a retrieval difference cannot be attributed to
either granularity or generator. Generating a small Claude sample of column
chunks as well — held out from training — gives a paired comparison that
separates the two.

## Stage 4: Filter and review

Joins every raw query back to its stage 2 chunk and applies deterministic
quality gates, recording one decision per query: `retain`, `reject`, or
`review`.

Rejected automatically: too few or too many words, excessive length,
source-filename mentions, model control tokens, long copies of the passage,
exact and near duplicates for the same chunk.

Sent to review: no meaningful lexical overlap with the source passage, or one
query pointing at multiple positives within a split. Low overlap is not treated
as proof of hallucination — a valid paraphrase may share few words.

The run fails if a query's chunk ID, source file, text hash, chunk type, split,
or split version disagrees with stage 2.

```bash
uv run agent-harness-genq-filter .local/queries/generated_queries.jsonl \
  --chunks .local/genq/corpus/chunks_with_splits.jsonl \
  --retained-output .local/queries/reviewed/retained_queries.jsonl \
  --review-output .local/queries/reviewed/query_reviews.jsonl \
  --report .local/queries/reviewed/filter_report.json
```

Deterministic filters remove mechanical defects. They do not establish
naturalness or factual support — inspect samples from every split and chunk type
before training on them.

## Stage 5: Pretrained encoder and exact FAISS baseline

Measures semantic retrieval without training anything. Embeds chunks and
retained queries with `sentence-transformers/all-MiniLM-L6-v2`, normalizes, and
stores passage vectors in `faiss.IndexFlatIP` — with normalized vectors, inner
product equals cosine similarity, and `IndexFlatIP` is exact, so
approximate-nearest-neighbor error cannot distort the baseline.

FAISS returns integer positions rather than application metadata, so the stage
writes a separate JSONL mapping each vector position to its stable table or
chunk identity. That mapping is required for correct retrieval.

The command supports two explicit evaluation modes: `--benchmark-dir` for the
table-level metadata artifact and reviewed gold queries shown above, or
`--queries` for the older synthetic chunk/query smoke evaluation below.

```bash
uv run --group genq agent-harness-genq-baseline .local/genq/corpus/chunks.jsonl \
  --queries .local/queries/reviewed/retained_queries.jsonl \
  --output-dir .local/embeddings/genq-baseline-smoke \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --limit 200 --batch-size 8 --top-k 10
```

### Encoder selection

`--model` defaults to `Qwen/Qwen3-Embedding-0.6B`; MiniLM stays reachable by
passing `--model sentence-transformers/all-MiniLM-L6-v2`.

`build_encoder` picks the implementation from the model name, because **pooling
is a property of how the checkpoint was trained**, not something a caller should
have to restate:

- `SentenceTransformerEncoder` loads Transformer weights directly and applies
  attention-mask mean pooling, avoiding SentenceTransformers' optional
  scikit-learn and SciPy imports at inference time.
- `QwenEmbeddingEncoder` uses **last-token pooling** with left padding, and
  wraps queries — but not documents — in an `Instruct: …\nQuery:…` prefix.
  Qwen3-Embedding is causal and trained so the final token's hidden state
  carries the sequence embedding; mean-pooling it would average that summary
  with every partial-context token.

Running a Qwen checkpoint through the mean-pooling path produces embeddings that
look valid and retrieve badly, which is the worst failure mode available here —
hence selection by name and a test asserting it. The asymmetry is why
`TextEncoder.encode` takes `is_query`; symmetric encoders accept and ignore it.

Scoring uses the same ranking calculator as the main retrieval evaluator. Each
synthetic query currently has one labeled positive, so Recall@K equals Hit@K and
Precision@K can penalize semantically useful but unlabeled sibling chunks.

## Status

Stages 1–5 are implemented. All table-level baselines run over the same 40,551
metadata records and the same 50 reviewed queries (41 answerable):

| Document metric | MiniLM-L6-v2 | Qwen3-Emb-0.6B | 0.6B 4-bit | Qwen3-Emb-4B 4-bit |
|---|---|---|---|---|
| Hit@1 | 0.220 | **0.390** | 0.390 | 0.268 |
| Hit@5 | 0.268 | **0.488** | 0.488 | 0.561 |
| Hit@10 | 0.366 | **0.512** | 0.537 | 0.610 |
| MRR | 0.261 | **0.446** | 0.445 | 0.370 |
| nDCG@10 | 0.268 | **0.502** | 0.498 | 0.490 |

Two conclusions, with different confidence.

**4-bit NF4 quantization is free.** Holding the model at 0.6B and changing only
precision moves Hit@1 by 0.000 and MRR by −0.002. A quantized encoder is a third
of the VRAM at no measurable cost, which is what makes larger encoders runnable
on a 10 GB card.

**0.6B versus 4B is not established.** With 41 answerable queries, Hit@1 of
0.390 versus 0.268 is sixteen queries against eleven — a difference of five, well
inside the ±0.073 standard error at this sample size. The same is true of 4B's
apparent Hit@10 advantage. Prefer 0.6B: no measured disadvantage, a sixth of the
request-path cost, and the burden of evidence is on the larger model.

The MiniLM → Qwen3 jump is a different matter and is real: it moves every metric
and every failure bucket, by margins several times larger.

Hit@5 by answerable failure bucket:

| Bucket | MiniLM | Qwen3-Embedding |
|---|---|---|
| named table lookup | 0.500 | **0.900** |
| named column schema | 0.300 | **0.500** |
| cross-table synthesis | 0.300 | **0.400** |
| business concept discovery | 0.000 | **0.182** |

The encoder swap alone roughly doubles retrieval with no training and no change
to the corpus, which reframes what fine-tuning has to beat: the MiniLM figures
were the floor a trained model was being compared against, and that floor moved.
Business concept discovery goes from total failure to merely poor, and remains
the weakest bucket — it is the one asking for a table by what it means rather
than by what it is called.

These are still pretrained dense floors, not production retrieval targets. The
nine unsupported queries remain diagnostics until an abstention threshold is
designed.

The earlier one-query, 200-chunk synthetic smoke test placed its positive at
rank 3 (Hit@5 1.000, MRR 0.333). That run validated the synthetic machinery but
established nothing about retrieval quality.

Synthetic-query evaluation does not replace the hand-authored analyst benchmark
in `evals/retrieval/benchmark/` — the two measure different things at different
granularity and their numbers are not comparable. A future training experiment
still needs a representative generation run across all chunk types and a manual
review pass.

## Tests

```bash
uv run pytest tests/retrieval/genq tests/test_dataset_split.py \
  tests/test_query_filter.py tests/retrieval/test_column_parser.py -q
```
