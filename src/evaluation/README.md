# Evaluation harness

Measures the RAG pipeline over a frozen cs.AI + cs.LG corpus. See
[docs/evaluation_plan.md](docs/evaluation_plan.md) for the full analysis plan.

**Status: Phase 1 (corpus) implemented. Phases 2-5 not started.**

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
