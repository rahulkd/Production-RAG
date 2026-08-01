# RAG Frontend

[assistant-ui](https://github.com/assistant-ui/assistant-ui) chat UI for the arXiv
Paper Curator RAG pipeline.

## How it connects

```
browser ──POST /api/chat──▶ Next route handler ──POST /api/v1/stream──▶ FastAPI
        ◀──── SSE ─────────                     ◀──── SSE ─────────────
```

The browser only ever talks to the Next origin. The route handler in
[app/api/chat/route.ts](app/api/chat/route.ts) forwards the request to the RAG
API and pipes the response body straight back, so:

- the API needs no CORS configuration,
- the API needs no published port in production,
- `RAG_API_URL` stays server-side and is read at **runtime**, not baked into the
  build — the same image runs in every environment.

## Threads

`useLocalRuntime` on its own is single-thread — its thread-list core reports an
empty list and throws on `switchToNewThread`. Multi-thread support comes from
wrapping it in `useRemoteThreadListRuntime`, which is the composition
assistant-ui documents for a backend that owns no thread state.

[lib/rag-runtime.ts](lib/rag-runtime.ts) supplies both halves: a per-thread
runtime hook, and a `RemoteThreadListAdapter` holding the thread registry.
Thread titles are derived from the first question rather than a second LLM call,
which would double the Bedrock cost of every new thread.

**Threads and their messages live in memory for the life of the tab.** A reload
starts clean. Persisting them needs a thread/message store behind the FastAPI
service, plus a `ThreadHistoryAdapter` passed to `useLocalRuntime`.

## How answers are rendered

[lib/rag-adapter.ts](lib/rag-adapter.ts) is a `ChatModelAdapter` driving
assistant-ui's `useLocalRuntime`. It parses the backend's SSE frames
(`{sources}`, `{chunk}`, `{answer, done}`, `{error}`) and yields the accumulated
text plus the retrieved papers as `source` message parts, which
[components/rag/message-sources.tsx](components/rag/message-sources.tsx) renders
as citation chips under each answer.

## Configuration

| Variable      | Default                 | Purpose                         |
| ------------- | ----------------------- | ------------------------------- |
| `RAG_API_URL` | `http://localhost:8000` | Base URL of the FastAPI service |

Models are listed in [lib/rag.ts](lib/rag.ts). Adding one there puts it in the
composer's model picker; it must also be enabled in Bedrock for the account the
API authenticates as.

## Local development

```bash
npm install
cp .env.example .env.local   # optional; the default already points at localhost:8000
npm run dev                  # http://localhost:3000
```

The API must be reachable — `docker compose up api` or `make run` at the repo root.

## Checks

```bash
npm run lint       # oxlint + oxfmt --check
npm run typecheck  # tsc --noEmit
npm run build
npm run test:e2e   # drives a real browser against a running app
```

[e2e/ui-test.mjs](e2e/ui-test.mjs) loads the app in headless Chromium, sends a
query, and asserts the answer streams, sources render, threads are created and
switched between, and the console stays clean. It defaults to
`http://localhost:3111`; set `BASE=http://localhost:3000` to test the container.

It exists because a build passing proves nothing about render behaviour — an
earlier `useShallow` selector that allocated fresh objects looked fine to `tsc`
and to curl, and crashed the browser tab on every answer. First run needs
`npx playwright install chromium` (and `install-deps` on a bare host).

## Docker

Built and run by the root `compose.yml` as the `frontend` service:

```bash
docker compose up -d --build frontend
```
