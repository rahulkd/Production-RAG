# TODO: remove the ~1s Langfuse flush from the request path

Status: **not done**. Measured on 2026-10-31; no code changed yet.

## Problem

Every `/ask` and `/stream` request pays an extra ~0.2-1.0 s (about 10-30 %) that the Langfuse UI does not show.

Cause: [tracer.py](../tracer.py) `RAGTracer.trace_request` calls `self.tracer.flush()` in its `finally` block, so each request blocks until the SDK has sent its queued events to the Langfuse server.

```python
# src/services/langfuse/tracer.py  (trace_request)
try:
    yield trace
finally:
    if trace:
        self.tracer.flush()      # <- blocks ~1 s
```

`flush()` is slow because the Langfuse v2 SDK batches events on a background thread and holds them for up to `LANGFUSE__FLUSH_INTERVAL` (default `1.0` s). A blocking flush waits out that window and then the HTTP POST to `/api/public/ingestion`.

## Measurements (2026-10-31, `rag-api` container, Langfuse v2.95.11)

Gap between the client-seen total and the pipeline time recorded in the trace:

| Question | Client total | Traced pipeline | Gap |
|---|---|---|---|
| speculative decoding | 3.48 s | 2.62 s | 0.86 s |
| LoRA | 3.03 s | 2.67 s | 0.36 s |
| mixture of experts | 3.04 s | 2.81 s | 0.23 s |

Direct probe of `client.flush()` with 5 queued events (1 trace + 4 spans, the same as one request): **1.04-1.07 s, three runs in a row**. `flush()` on an empty queue: 0.0 ms.

The trace's own duration (e.g. 2.63 s in the UI) does not include this. It is measured from trace open to answer ready; the flush happens after.

## Effect per endpoint

- `/ask`: the user waits for the flush before getting the response. Real added latency.
- `/stream`: the user already has the full answer, but the connection stays open until the flush ends, so the frontend's "total" time (message latency line) is inflated by it. First-token time is unaffected.

## Proposed fix

Remove the per-request flush in `trace_request`:

```python
try:
    yield trace
finally:
    pass   # SDK's background thread sends events; shutdown() flushes on exit
```

Why this is safe:
- The SDK already sends events in the background every `LANGFUSE__FLUSH_INTERVAL` s, or when `LANGFUSE__FLUSH_AT` (15) events are queued. Creating a span only appends to an in-memory queue.
- `src/main.py` calls `app.state.langfuse_tracer.shutdown()` after `yield` in `lifespan`. It flushes before closing, so queued traces are delivered on a normal stop.

Optional tuning once the blocking flush is gone:
- Lower `LANGFUSE__FLUSH_INTERVAL` (e.g. `0.5`) so traces show up in the UI sooner. This has no request-latency cost because sending is in the background.
- Keep `LANGFUSE__FLUSH_AT` as is. One request produces 5 events, so the batch is sent by the interval, not by count.

## Trade-off / things to check

- If the process is killed hard (SIGKILL, OOM), up to `FLUSH_INTERVAL` seconds of traces can be lost. Acceptable for monitoring, and a normal `docker stop` (SIGTERM) still flushes.
- After the change, re-measure client total vs traced pipeline time. The gap should drop to a few ms.
- Traces appear in the UI up to `FLUSH_INTERVAL` s after the request, instead of immediately.
- `trace_request` has no other users of the flush. `LangfuseTracer.flush()` itself stays (shutdown uses it).

## How to re-measure

1. Rebuild the API: `docker compose up -d --build api` (the `src/` code is baked into the image).
2. Ask 3 new questions through `/api/v1/ask` (cached repeats skip the pipeline), timing with `curl -w '%{time_total}'`.
3. Compare each to `output.total_duration_seconds` of its trace in `rag-langfuse-postgres`:
   ```sql
   select output->>'total_duration_seconds' from traces
   where input->>'query' = '<question>' order by timestamp desc limit 1;
   ```
4. Or use the frontend's `/ask` toggle: the latency line under the answer shows the browser-measured total.

## Related, not part of this task

- The LLM call is recorded as a plain span, so token and cost columns in Langfuse are empty. See "Not yet observable" in [langfuse_design_workflow.md](langfuse_design_workflow.md).
