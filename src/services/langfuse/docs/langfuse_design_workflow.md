# Langfuse Integration: High-Level Design

How the RAG API traces each user question with self-hosted Langfuse v2, what that lets us monitor in the Langfuse UI, and how Langfuse stores the traces in PostgreSQL.

## 1. Components

| Component | Container | Role |
|---|---|---|
| RAG API (FastAPI) | `rag-api` | Serves `/api/v1/ask` and `/api/v1/stream`; emits trace data via the Langfuse Python SDK (v2) |
| `LangfuseTracer` | in `rag-api` | Thin wrapper over the SDK client. Disabled automatically when `LANGFUSE__ENABLED=false` or keys are missing ([client.py](../client.py)) |
| `RAGTracer` | in `rag-api` | RAG-specific helpers: one context manager per pipeline step ([tracer.py](../tracer.py)) |
| Langfuse server (v2) | `rag-langfuse` | Receives events over HTTP, writes them to Postgres, serves the UI on http://localhost:3000 |
| Langfuse Postgres | `rag-langfuse-postgres` | The only datastore for Langfuse v2: traces, spans, users, projects, API keys |
| App Postgres | `rag-postgres` | The RAG app's own data (papers). **Not used by Langfuse**; the two databases are fully separate |

```mermaid
flowchart LR
    U[User / frontend] -->|POST /ask, /stream| API[rag-api<br/>FastAPI]
    API --> R[(Redis cache)]
    API --> OS[(OpenSearch)]
    API --> J[Jina embeddings]
    API --> B[AWS Bedrock<br/>Llama 3.3 70B]
    API -. "SDK batches events<br/>POST /api/public/ingestion" .-> LF[rag-langfuse<br/>Langfuse v2]
    LF --> LPG[(rag-langfuse-postgres)]
    Dev[Engineer] -->|browser :3000| LF
```

Tracing is off the critical path where possible: if Langfuse is down or disabled, every tracer call returns `None`, and the request continues untraced.

## 2. How a user prompt is traced

### 2.1 Request flow

Each call to `/ask` or `/stream` opens one trace and adds a span for each pipeline step:

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as /ask router
    participant T as RAGTracer / SDK
    participant Ca as Redis cache
    participant E as Jina
    participant S as OpenSearch
    participant L as Bedrock
    participant LF as Langfuse server

    C->>A: AskRequest(query, top_k, use_hybrid, model, categories)
    A->>T: trace_request("api_user", query) → trace "rag_request"
    A->>Ca: find_cached_response(request)
    alt cache hit
        Ca-->>A: AskResponse
        A-->>C: cached answer (trace has input only, no spans)
    else cache miss
        A->>T: span "query_embedding"
        A->>E: embed_query(query)
        A->>T: span "search_retrieval"
        A->>S: search_unified(...)
        A->>T: span "prompt_construction"
        A->>T: span "llm_generation"
        A->>L: generate_rag_answer(...)
        A->>T: end_request(answer, total duration)
        A->>Ca: store_response(request, response)
        A-->>C: AskResponse
    end
    T->>LF: flush() at end of trace — batched HTTP POST
```

The SDK queues events in memory and a background thread sends them in batches (`LANGFUSE__FLUSH_AT=15` events or every `LANGFUSE__FLUSH_INTERVAL=1.0` s). `RAGTracer.trace_request` also calls `flush()` when each request finishes. On API shutdown, `main.py` calls `shutdown()` to send anything still queued.

### 2.2 What each trace contains

```
rag_request                          (trace)
│  input:    { query }
│  output:   { answer, total_duration_seconds, response_length }
│  user_id:  "api_user"   session_id: "session_api_user"
│
├── query_embedding                  (span; only when use_hybrid=true)
│     input:  { query, query_length }
│     output: { embedding_duration_ms, success }   ← or { success: false, error } on fallback to BM25
│
├── search_retrieval                 (span)
│     input:  { query, top_k }
│     output: { chunks_returned, unique_papers, total_hits, arxiv_ids }
│
├── prompt_construction              (span)
│     input:  { chunk_count }
│     output: { prompt_length, prompt_preview (first 200 chars) }
│
└── llm_generation                   (GENERATION)
      model:  us.meta.llama3-3-70b-instruct-v1:0
      input:  { prompt (full) }
      output: { response, response_length, model_used }
      usage:  input / output / total tokens (from Bedrock) + cost in USD

trace tags:  [ask | stream, cache_hit | cache_miss]
trace scores (§3.1): retrieval_hit, unique_papers, hybrid_fallback,
                     judge_refused, judge_answer_relevance, judge_faithfulness, user_feedback
```

On a **cache hit** the trace has no spans: it is tagged `cache_hit`, its output is the cached answer, and the response carries this request's own `trace_id` (not that of the request that filled the cache), so feedback lands on the right trace.

Every span records a start and end time, so Langfuse shows how long each step took.

- `/ask` records the structured prompt (`create_bedrock_structured_prompt`), which is what `generate_rag_answer` sends.
- `/stream` records the plain prompt (`create_rag_prompt`), which is what `generate_rag_answer_stream` sends.
- For `/stream`, the `llm_generation` span covers the whole token stream, from first token to last.

### 2.3 Where it lives in code

| Concern | File |
|---|---|
| Tracer created once at startup, shut down on exit | [src/main.py](../../../main.py) (`lifespan`) |
| Injected into routes as `LangfuseDep` | [src/dependencies.py](../../../dependencies.py) |
| Spans around each pipeline step | [src/routers/ask.py](../../../routers/ask.py) |
| Settings (`LANGFUSE__*` env vars) | [src/config.py](../../../config.py) (`LangfuseSettings`) |

## 3. What we monitor in the Langfuse UI

| UI view | What it shows for this app | Questions it answers |
|---|---|---|
| **Traces** list | One row per question: query, answer, timestamp, total latency | What are people asking? Which requests were slow or failed? |
| **Trace detail** (timeline) | The span tree from §2.2, shown as a waterfall | Is the slow part Jina, OpenSearch or Bedrock? |
| `search_retrieval` output | `chunks_returned`, `unique_papers`, `arxiv_ids` | Did retrieval find anything? Is it all from one paper? Which papers back the answer? |
| `query_embedding` output | `success: false` + error | How often does hybrid search silently fall back to BM25? |
| `llm_generation` input/output | The exact prompt sent and the raw answer | Debugging hallucinations or poor answers against the actual context given |
| **Generations** view, trace cost/token columns | Input/output tokens and USD cost per LLM call | What does a question cost? Which prompts are largest? |
| Tags filter `cache_hit` / `cache_miss`, `ask` / `stream` | Which answers came from Redis, and through which endpoint | Cache hit rate; `/ask` vs `/stream` traffic |
| **Scores** (per trace, and in the Scores view and dashboard) | The quality scores in §3.1 | Is answer quality drifting? Which questions get thumbs down? |
| **Dashboard** | Trace volume, latency, cost and scores over time | Is load, latency, spend or quality trending? |

### 3.1 Quality scores

| Score | Type | Source | When |
|---|---|---|---|
| `retrieval_hit` | BOOLEAN | Heuristic: retrieval returned ≥ 1 chunk | Every uncached request |
| `unique_papers` | NUMERIC | Heuristic: distinct papers among retrieved chunks | Uncached, with chunks |
| `hybrid_fallback` | BOOLEAN | Heuristic: hybrid requested, but embeddings failed so BM25 ran | Uncached, `use_hybrid=true` |
| `judge_refused` | BOOLEAN | LLM judge, Phase 4 `refusal` rubric | Uncached, sampled |
| `judge_answer_relevance` | NUMERIC 0–3 | LLM judge, reference-free rubric (`prompts/answer_relevance_v1.txt`) | Attempted answers only |
| `judge_faithfulness` | NUMERIC 0–1 | LLM judge, Phase 4 `claims` + `faithfulness` rubrics: supported / all claims (counts in the comment) | Attempted answers with claims |
| `user_feedback` | BOOLEAN | 👍 = 1 / 👎 = 0 from the frontend via `POST /api/v1/feedback` | When the user votes; re-voting overwrites |

**Heuristic scores** are computed in the request from data it already has, with no extra calls ([tracer.py](../tracer.py) `score_retrieval`).

**Judge scores** come from [judge.py](../judge.py) (`OnlineJudge`). It reuses the offline evaluation's `Judge` class and rubrics, so live scores mean the same as the Phase 4 numbers. It runs as a background task after the response is sent, so it adds no user latency. Its scores go on the `rag_request` trace. Its 4 calls (prompt, reply, tokens, cost) go in a **separate `rag_judge` trace** with id `<rag_request trace id>-judge`, tagged `judge`, with `parent_trace_id` in its metadata. Keeping them out of `rag_request` leaves that trace's latency at what the user waited for. It also lets the dashboard split generator cost from judge cost by trace name.
- A judged answer costs up to 4 DeepSeek V3.2 calls (refusal, relevance, claims, faithfulness). Refusals stop after the first call.
- `LANGFUSE__JUDGE_SAMPLE_RATE` controls the fraction of answers judged.
- On shutdown, the API waits up to 30 s for pending judgments.
- Offline relevance compares against a golden answer. The online rubric can't, so it judges only whether the answer addresses the question.

**User feedback:** the trace id reaches the frontend in `AskResponse.trace_id` (`/ask`) or the stream's first metadata frame (`/stream`). The 👍/👎 buttons appear only when a trace id is present. The score id is `{trace_id}-user_feedback`, so a changed vote overwrites the earlier one.

### Not yet observable

- **Per-user views.** Every trace uses `user_id="api_user"`, so the Users and Sessions views show a single user. Passing a real user or session id from the frontend would split them.

## 4. PostgreSQL's role in Langfuse v2

### 4.1 Postgres is Langfuse v2's only datastore

Langfuse v2 needs exactly one database: PostgreSQL. It holds both the application state (accounts, projects, keys) and all observability data (traces, spans, scores). There is no separate analytics store, queue or object store.

Langfuse v3 moved the trace data to ClickHouse and added Redis and S3/MinIO. That's why the v2.9x startup log prints `CLICKHOUSE_URL not configured, skipping migration`. It is a no-op on v2.

This project runs a **dedicated** Postgres for Langfuse (`rag-langfuse-postgres`, database `langfuse`, volume `langfuse_postgres_data`), separate from the app's `rag-postgres`. Trace volume therefore can't affect paper storage or ingestion, and either one can be backed up, reset or upgraded on its own.

### 4.2 Startup: schema migrations

On every container start, Langfuse runs its Prisma migrations against `DATABASE_URL`, about 270 of them, creating any missing tables, indexes and views. Migrations that are already applied are skipped, and progress is recorded in the `_prisma_migrations` table.

### 4.3 Write path: from SDK event to rows

```mermaid
flowchart LR
    SDK["Langfuse Python SDK<br/>(in rag-api)"] -->|"batch of events<br/>(trace-create, span-create,<br/>span-update, ...)"| ING["POST /api/public/ingestion<br/>HTTP Basic auth: public_key:secret_key"]
    ING --> AUTH{"Validate key against<br/>api_keys table<br/>→ resolve project_id"}
    AUTH -->|valid| UP["Upsert by event id"]
    AUTH -->|invalid| R401[401: events dropped]
    UP --> TR[(traces)]
    UP --> OB[(observations)]
    UP --> SC[(scores)]
    UP --> SE[(trace_sessions)]
```

1. **Authentication.** The SDK sends `LANGFUSE__PUBLIC_KEY` and `LANGFUSE__SECRET_KEY` as HTTP Basic auth. Langfuse looks up the key in `api_keys` (it stores a hash of the secret, never the plain secret) and resolves which **project** the events belong to. Every data row carries that `project_id`.
2. **Upsert.** Each event has a client-generated id. A create inserts a row. A later update to the same id (e.g. `span.update(output=...)` or `span.end()`) is merged into that row. That's how a span gets its output and end time after it started.
3. **Mapping to tables:**

| Our concept | Postgres table | Key columns |
|---|---|---|
| `rag_request` trace | `traces` | `id`, `project_id`, `name`, `timestamp`, `input`, `output`, `user_id`, `session_id`, `metadata`, `tags` |
| Each span (`query_embedding`, `search_retrieval`, ...) | `observations` | `id`, `trace_id`, `parent_observation_id`, `type` (SPAN / GENERATION / EVENT), `name`, `start_time`, `end_time`, `input`, `output`, `model`, token and cost columns |
| `session_api_user` | `trace_sessions` | `id`, `project_id` |
| Quality scores (§3.1) | `scores` | `trace_id`, `observation_id`, `name`, `value`, `source` |

`input`, `output` and `metadata` are JSONB columns, so the whole prompt and answer are stored as sent.

### 4.4 Read path: how the UI queries it

The UI doesn't read the raw tables directly for its lists. Langfuse defines two Postgres **views**:

- **`traces_view`** joins a trace with its observations to derive trace-level latency and totals.
- **`observations_view`** adds per-observation latency, time to first token, and cost. Cost normally comes from matching `model` against the `models` and `prices` tables. Our generation sends its cost explicitly in `usage` (priced from `BEDROCK__*_PRICE_PER_1M_TOKENS`), and Langfuse uses that as-is, so no model definition is needed in the Langfuse UI.

Filtering, search and the dashboard charts are SQL queries over these tables and views, using indexes on `project_id`, `trace_id` and `start_time`. This is what limits v2 at very high volume (millions of traces). For this project's volume, a few hundred traces a day, Postgres is comfortably sufficient.

### 4.5 Other Langfuse state in the same database

| Table(s) | Purpose |
|---|---|
| `users`, `Account`, `Session`, `verification_tokens` | UI sign-up and login (NextAuth) |
| `organizations`, `projects`, `organization_memberships`, `project_memberships` | Tenancy and access |
| `api_keys` | SDK credentials per project |
| `prompts` | Langfuse prompt management (unused here; our prompts live in `src/services/LLM/prompts`) |
| `datasets`, `dataset_items`, `dataset_runs`, `dataset_run_items` | Langfuse datasets (unused; our golden set is `src/evaluation/datasets`) |
| `models`, `prices` | Model-name → token-price mapping for cost |

### 4.6 Retention and backup

Langfuse v2 doesn't delete old traces automatically. Data grows in the `langfuse_postgres_data` Docker volume until traces are deleted in the UI or the volume is pruned. `docker compose down -v` deletes the volume, and with it all traces, the account, projects and API keys. After that, the `LANGFUSE__*_KEY` values in `.env` must be regenerated.

## 5. Configuration

| Env var (`.env`) | Default | Effect |
|---|---|---|
| `LANGFUSE__ENABLED` | `true` | Master switch |
| `LANGFUSE__PUBLIC_KEY` / `LANGFUSE__SECRET_KEY` | empty | Project keys from the Langfuse UI (Settings → API Keys). Empty → tracing disabled |
| `LANGFUSE__HOST` | `http://localhost:3000` | Overridden to `http://langfuse:3000` inside Docker by `compose.yml` |
| `LANGFUSE__FLUSH_AT` / `LANGFUSE__FLUSH_INTERVAL` | `15` / `1.0` | SDK batch size / interval |
| `LANGFUSE__DEBUG` | `false` | Verbose SDK logging |
| `LANGFUSE__JUDGE_ENABLED` | `true` | Online LLM-as-judge on/off. **Set `false` during evaluation runs**: the harness already judges offline, and leaving it on doubles judge spend |
| `LANGFUSE__JUDGE_MODEL` | `deepseek.v3.2` | Judge model (same as Phase 4) |
| `LANGFUSE__JUDGE_SAMPLE_RATE` | `1.0` | Fraction of uncached answers judged |
| `LANGFUSE__JUDGE_MAX_TOKENS` | `4000` | Judge reply budget |
| `LANGFUSE__JUDGE_INPUT_PRICE_PER_1M_TOKENS` / `LANGFUSE__JUDGE_OUTPUT_PRICE_PER_1M_TOKENS` | `0.62` / `1.85` | USD price of the judge model (DeepSeek V3.2, Bedrock US), used for the cost on `rag_judge` traces |
| `BEDROCK__INPUT_PRICE_PER_1M_TOKENS` / `BEDROCK__OUTPUT_PRICE_PER_1M_TOKENS` | `0.72` / `0.72` | USD price of the generator model, used for the cost sent to Langfuse. Update if the model or AWS price changes |

First-time setup: open http://localhost:3000, sign up, create an organization and project, create API keys, put them in `.env`, then restart `rag-api`.

## 6. Failure behavior and known limits

| Situation | Behavior |
|---|---|
| Keys missing or `ENABLED=false` | The tracer logs "disabled" at startup. All calls are no-ops |
| Langfuse server down | The SDK can't send events, so traces are lost. Requests still succeed |
| Wrong keys (e.g. after a volume reset) | Ingestion returns 401 and events are dropped. Requests still succeed. Check `rag-api` logs |
| Exception inside a request | `/ask` catches it inside the trace and re-raises after the trace closes. The SDK's context manager can't have exceptions thrown through it |

Known limits:
- **Per-request flush latency.** `RAGTracer.trace_request` calls the SDK's `flush()` at the end of every request, and `flush()` blocks until the queue is sent. This adds a Langfuse round-trip to each `/ask` response. The background batching already sends events within `FLUSH_INTERVAL`, so removing the per-request flush would keep tracing without the added latency. The shutdown flush still guarantees delivery on exit.
- **No sampling.** Every request is traced, which is fine at this scale.
