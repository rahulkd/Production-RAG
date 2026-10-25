# Evaluation harness

Measures the RAG pipeline over a frozen cs.AI + cs.LG corpus. See
[docs/evaluation_plan.md](docs/evaluation_plan.md) for the full analysis plan.

**Status: Phases 1-5 implemented (corpus, golden set, retrieval, generation, end-to-end).**

---

## Phase 1 — build the corpus

Two commands. Run them from the repo root.

```bash
# 1. Select 50 papers and write the manifest (~1 min, arXiv is rate-limited to 3s/query)
python -m src.evaluation.corpus.discover

# 2. Fetch, parse, chunk, embed, and index them (~45-90 min, docling parses serially)
OPENSEARCH__HOST=http://localhost:9200 python -m src.evaluation.corpus.build
```

> **The host override matters.** `.env` sets `OPENSEARCH__HOST=http://opensearch:9200`,
> the Docker-internal name. It resolves inside the compose network but not from
> the host, where OpenSearch is on `localhost:9200`.

Smoke-test the whole path on 3 papers before committing to the full run:

```bash
OPENSEARCH__HOST=http://localhost:9200 python -m src.evaluation.corpus.build --limit 3
```

### What gets built

* **Index `arxiv_evaluation_index`** — separate from the production
  `arxiv-papers-chunks`, with the same mapping, so building or wiping the eval
  corpus cannot disturb the live index.
* **`corpus/corpus_manifest.json`** — the frozen corpus definition. **Commit this.**
  Every evaluation number traces back to it.
* **`logs/corpus_build_<run>.log`** — the full run log.
* **`logs/pdf_failures_<run>.log`** and `.json` — papers that failed, grouped by
  stage, with a copy-pasteable ID list per stage. Written even on a clean run,
  because an empty report is evidence and a missing file is ambiguous.

### Useful flags

| Flag | Effect |
|---|---|
| `--dry-run` | (discover) print the 20 queries and exit, no network |
| `--limit N` | (build) process only the first N papers |
| `--skip-fetch` | (build) re-index from Postgres without re-downloading |
| `--force-index` | (build) delete and recreate the index first |
| `--verbose` | DEBUG on the console |

---

## The evaluation index is frozen

The golden set labels chunks as `(arxiv_id, chunk_index)`, so any write to `arxiv_evaluation_index`
can silently move a label. Once built it is write-blocked (`index.blocks.write`): the cluster refuses
indexing, updates and deletes of documents from any client, including Airflow. Reads and search
are unaffected.

```bash
python -m src.evaluation.corpus.freeze --status   # FROZEN or writable
python -m src.evaluation.corpus.freeze            # verify against the manifest, then freeze
python -m src.evaluation.corpus.freeze --lift     # unfreeze
```

Freezing first checks the index holds exactly the manifest's chunk and paper counts, so a
half-built corpus cannot be frozen. A write block does not stop an index being *deleted*, so
`corpus.build` refuses a frozen index unless given `--unfreeze`. To rebuild deliberately:
`build --unfreeze ...`, verify, then run `freeze` again.

Ingestion is separate: Airflow's indexer always writes to the production index, whatever
`OPENSEARCH__ACTIVE_INDEX` says. That setting only changes what the API *searches*.

---

## Phase 4 — generation analysis

```bash
OPENSEARCH__HOST=http://localhost:9200 python -m src.evaluation.generation.runner --limit 5        # smoke test
OPENSEARCH__HOST=http://localhost:9200 python -m src.evaluation.generation.runner                  # full run, k=10
OPENSEARCH__HOST=http://localhost:9200 python -m src.evaluation.generation.runner --top-k 3 5 10    # compare cutoffs
```

Answers every accepted row of `datasets/golden_v1.reviewed.jsonl` (answerable and
unanswerable; rejected rows are skipped) through the `/ask` path — hybrid search at
each requested k (default 10) → `create_bedrock_structured_prompt` → Llama 3.3 70B
at temperature 0 — and scores each answer:

| | Metric | How |
|---|---|---|
| (a) | Faithfulness, contradictions (Q7) | Judge extracts claims, then labels each against the retrieved context only |
| (b) | Relevance 0-3 | Judge, against question + reference answer |
| (c) | Citation precision / recall / presence (Q8) | Regex, no LLM |
| (d) | Refusal recall, false refusal rate (Q9) | Judge classifies refused / attempted |

Plus a retrieval-hit × answered cross-tab for Q10. Outputs go to
`results/generation_<run>/<mode>_k<k>/` (`generation_raw.jsonl`, `generation_summary.json`,
`report.md`), and one row per cutoff is appended to `results/generation_leaderboard.md`.
Rubrics are in `generation/prompts/*_v1.txt`; bump `JUDGE_PROMPT_VERSION` in
`config.py` when one changes.

**Judge model.** `JUDGE_MODEL` in `config.py`: DeepSeek V3.2 (`deepseek.v3.2`) at
temperature 0 — a different model family from both the generator (Llama) and the
reference-answer writer (Qwen). Its labels repeat across identical calls; only the
`reason` wording varies. It replaced Kimi K3 (`us.moonshotai.kimi-k3`), which cannot be
pinned to temperature 0 on Converse but was stricter on one hard claim — if calibration
shows DeepSeek is lenient on contradictions, Kimi K3 is the alternative
(`--judge-model us.moonshotai.kimi-k3`, with `JUDGE_PARAMS` both `None`). Claude, the
original choice, is blocked on this account.

| Flag | Effect |
|---|---|
| `--judge-model ID` | Override the judge (must not be the generator) |
| `--top-k N [N ...]` / `--mode` | Cutoffs to evaluate, one leaderboard row each (default 10; k=10 prompts are ~10-14K tokens) |
| `--no-judge-cache` | Ignore cached verdicts and re-judge (measures judge run-to-run variance) |
| `--regenerate` | Ignore cached Llama answers |
| `--concurrency N` | Questions in flight (default 4) |

Judge verdicts and Llama answers are cached in `results/judge_cache.jsonl` and
`results/answer_cache.jsonl`, so an interrupted or repeated run pays only for
what changed. A run where more than 5% of questions hit an error is not
appended to the leaderboard.

**Calibration (do once per rubric version).** κ < 0.6 means the rubric is broken:

```bash
python -m src.evaluation.generation.calibrate sample --run <run_id> --cutoff hybrid_k10   # 40 blind items
# set human_label in results/generation_<run_id>/hybrid_k10/calibration_items.jsonl
python -m src.evaluation.generation.calibrate score --run <run_id> --cutoff hybrid_k10    # writes results/judge_calibration.md
```

---

## Phase 5 — end-to-end over HTTP

Sends the golden set through the real `/api/v1/ask` and `/api/v1/stream` routes
(Q11: does the deployed API behave like the harness?).

**The API must be serving the evaluation index.** `OPENSEARCH__ACTIVE_INDEX` selects what
the API searches: `production` (`arxiv-papers-chunks`, empty until papers are ingested) or
`evaluation` (the frozen `arxiv_evaluation_index`). The runner's preflight stops if the API
answers from an empty index.

The runner targets the compose `rag-api` on :8000 by default:

```bash
# .env: OPENSEARCH__ACTIVE_INDEX=evaluation, then recreate the container (restart does not re-read .env)
docker compose up -d api

OPENSEARCH__HOST=http://localhost:9200 python -m src.evaluation.e2e.runner               # both routes, k=3 and k=10
OPENSEARCH__HOST=http://localhost:9200 python -m src.evaluation.e2e.runner --limit 5     # smoke test
```

Run from the repo root with `-m`. The `OPENSEARCH__HOST` prefix matters because the runner also
queries OpenSearch directly for its reference retrieval, and `.env`'s `opensearch:9200` only
resolves inside Docker. To test uncommitted code without touching the container, start the
current code on another port (`OPENSEARCH__ACTIVE_INDEX=evaluation .venv/bin/uvicorn src.main:app
--port 8001`, plus the same host overrides) and pass `--base-url http://localhost:8001`.

Per route and k it reports:
- **agreement**: the API's `sources` against the in-process harness retrieval for the same query;
- **search_mode**: catches the silent BM25 fallback in `ask.py`;
- **latency**: p50/p95, plus time to first token on `/stream`;
- **answer quality**: the Phase 4 judge and citation checks on the served answer. Faithfulness
  is scored only where the API's papers match the harness, since the API doesn't return chunk text.

The routes sample at `BEDROCK__RAG_TEMPERATURE` (default 0.0; the first two runs, before 2026-10-25, used a hardcoded 0.7), so above 0 each answer is one sample, not pinned. Outputs go to
`results/e2e_<run>/` (per-config raw and summary files, plus a side-by-side `report.md`), with
one row per config appended to `results/e2e_leaderboard.md`.

---

## How the 50 papers are chosen

A single `cat:cs.AI` query sorted by date returns whatever was posted that week,
which clusters hard around a few trending subjects. Retrieval metrics on such a
corpus are biased: every query looks like every document, so BM25 and vector
search both look fine and neither is distinguished.

Instead, one query runs per **topic slice** with a fixed quota, so the spread is
a property of the design rather than an accident of the calendar. Slices are
defined in [corpus/topics.py](corpus/topics.py).

* **cs.AI — 30 papers**, 10 Gen-AI topics x 3: LLMs, LoRA/PEFT, open-weight
  models (Llama/Mistral/Qwen), transformer architectures, pretraining and
  scaling, agentic AI, RAG, RLHF and preference optimization, instruction
  tuning and alignment, reasoning.
* **cs.LG — 20 papers**, 10 topics x 2: computer vision, NLP, recommendation,
  time series, graph learning, generative models, optimization, federated and
  privacy, RL, self-supervised learning.
* **Window** — submitted 2024-01-01 or later.

Selection rules live in [corpus/selection.py](corpus/selection.py):

1. Drop candidates not genuinely in the category or outside the date window.
   arXiv's `cat:` matching is permissive and does return these.
2. Claim in topic order, quota-limited. A paper is claimed **once**, keyed on the
   version-stripped ID, and the claim set is shared across both categories.
   Gen-AI papers are routinely cross-listed as both cs.AI and cs.LG; without
   this they would be selected twice and the corpus would look more diverse
   than it is.
3. Backfill shortfalls round-robin, so a topic whose query returns too little
   degrades spread as little as possible.

### On "cs.AI category"

arXiv's `cat:cs.AI` matches papers where cs.AI is *any* category, not
necessarily the primary one — and cs.AI is largely a cross-list category, so
most LLM papers carry cs.CL or cs.LG as primary. The 30 cs.AI papers are all
*tagged* cs.AI; typically only a handful have it as primary.

This is deliberate. Requiring primary=cs.AI would shrink the pool to a narrow
slice of the field and reduce topical diversity, which is the opposite of the
requirement. To change it, filter on `candidate.primary_category` in
`_is_eligible`.

---

## Why the manifest records text hashes

Ground truth in Phase 2 is addressed as `(arxiv_id, chunk_index)` — a
*positional* label. Its meaning depends on the extracted text and the chunking
being byte-identical to when it was recorded.

If arXiv serves a new version, or docling changes, chunk 7 becomes different
text, every label silently points somewhere wrong, and **the scores stay
plausible**. That is the expensive failure: numbers you trust that are wrong.

So `build` records a SHA-256 of each paper's extracted text and re-checks it on
every rebuild. A mismatch exits non-zero and names the affected papers. Two
tripwires at two levels, with different recovery costs:

| Changed | What moves | Recovery |
|---|---|---|
| Chunker settings | `raw_text` unchanged, chunk boundaries move | Re-label from recorded answer snippets (Phase 2) |
| Parser / paper version | `raw_text` itself changes | Affected papers need ground-truth re-review |

---

## Tests

```bash
uv run pytest          # or: ./.venv/bin/python -m pytest
```

`tests/` is hermetic — no OpenSearch, Postgres, Bedrock, Jina, or arXiv calls —
and covers query construction, selection and dedup rules, manifest round-trip
and hash verification, and the index-payload boundary. Anything needing live
services is a CLI here, run deliberately.
