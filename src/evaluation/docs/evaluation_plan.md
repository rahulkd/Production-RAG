# RAG Pipeline Evaluation — Analysis Plan (cs.AI + cs.LG)

## Context

Four stages are built and working: arXiv ingestion → hybrid OpenSearch indexing → Jina chunking/embeddings + RRF hybrid search → Bedrock Llama 3 70B generation. **Nothing currently measures whether any of it is good.** A repo-wide search finds zero hits for `recall@`, `ndcg`, `mrr`, `precision@`, LLM-as-judge, or any labeled QA set. The only quantitative artifacts are latency tables (`testing/rag_pipeline/results/rag_api_profile_20260801_113248.md`) and embedding sanity checks; `notebooks/week4/README.md:239` has a "Recall@10 / Precision@10" table header backed by no code.

Consequently every configuration choice in the pipeline is an unvalidated guess: hybrid vs BM25, `top_k=3`, RRF `rank_constant=60`, `chunk_size=600/overlap=100`, the `chunk_text^3` field boost. There is no way to tell whether a change helps or hurts, and no way to defend the current settings.

**Outcome sought:** a repeatable analysis producing numbers that are comparable *across runs and across config changes*, over a frozen cs.AI + cs.LG corpus — plus a written record of what the pipeline's actual weak point is.

---

## What this evaluation answers

This is the core of the plan. Each row is a question about the pipeline, the metric that settles it, and the decision it drives.

| # | Question | Metric | Decision it drives |
|---|---|---|---|
| Q1 | Is hybrid search actually better than BM25 alone or vector alone? | nDCG@5, Recall@5 across the three modes | Whether the RRF pipeline earns its complexity and latency |
| Q2 | Is `top_k=3` starving the generator? | Recall@k for k ∈ {3,5,10,20} | The production default in `AskRequest.top_k` |
| Q3 | Is `rank_constant=60` right for this corpus? | MRR, nDCG@5 across rank_constant ∈ {10,20,60,100} | The RRF search-pipeline config |
| Q4 | Is the retriever semantic, or is it lexically flattered? | Score gap between paraphrase pairs, per mode | Whether the embedding leg is contributing anything |
| Q5 | Does the `title^2`/`abstract^1` boost help, given the abstract is already inside every chunk? | Recall@5 with and without the boost | `QueryBuilder` field weights |
| Q6 | Is `fuzziness: AUTO` helping or adding noise on dense technical text? | Recall@5, precision@5 with/without | `QueryBuilder` match config |
| Q7 | Does the answer stay inside the retrieved context? | Faithfulness (supported claims / total), contradiction count | Whether the prompt needs tightening or the model needs changing |
| Q8 | Are the `[arXiv:id]` citations real? | Citation precision — cited IDs present in the retrieved set | Whether citations are trustworthy enough to surface in the UI |
| Q9 | Does it refuse when the corpus can't answer? | Refusal recall, false-refusal rate | Whether the system hallucinates on out-of-scope questions |
| Q10 | Is retrieval or generation the bottleneck? | Recall@k vs faithfulness on the same questions | **Where to spend the next sprint** |
| Q11 | Does the deployed API behave like the harness? | Paper-level Recall@k in-process vs via `POST /api/v1/ask` | Whether the silent BM25 fallback in `ask.py` is firing in production |

Q10 is the headline. Recall@k upper-bounds faithfulness — the generator cannot ground an answer in a chunk that was never retrieved — so reading the two together tells you which half to fix.

---

## Methodology

**Evaluate retrieval and generation separately, then end-to-end.** A single end-to-end score tells you an answer was bad but not why, and the two failure modes need opposite fixes.

1. **Retrieval — deterministic and exact.** The golden set records ground truth as `(arxiv_id, chunk_index)` pairs, so Recall@k / Precision@k / MRR / nDCG@k are computable in pure numpy. No LLM, no cost, no variance. This is an unusual advantage and it is the reason a custom harness beats RAGAS here — RAGAS's `context_precision`/`context_recall` are LLM-judged over context *strings*, which would replace exact metrics with noisy ones.
2. **Generation — LLM-as-judge against a rubric.** "Is this answer grounded in the context" has no closed form. Judged at temperature 0, with versioned rubric prompts and a one-time human calibration check.
3. **End-to-end — the same golden set through the live HTTP API**, to catch harness-vs-production divergence.

**Framework decision: custom harness, zero new production dependencies.** `metrics.py` is ~120 lines of numpy; `judge.py` ~150 lines against the existing `BedrockClient`. Steal 's *mRAGASetric definitions* (claim-decomposition faithfulness, the answer-relevance formulation) as prior art. The only dependency change is pinning `numpy>=2.0` explicitly, since it is currently satisfied only transitively via docling.

**Judge model: `us.anthropic.claude-sonnet-4-6`**, distinct from the generator (`meta.llama3-70b-instruct-v1:0`) to avoid self-preference bias — a model scoring its own output is not a measurement. Two prerequisites to confirm before Phase 4: Anthropic models on Bedrock need the `us.` cross-region inference-profile prefix, and the account needs the Anthropic use-case form submitted in the Bedrock console. Sonnet 4.6 is chosen over Opus 5 / Sonnet 5 because it still accepts `temperature`/`top_p`, which `BedrockClient._build_inference_config` always sends. `JUDGE_MODEL` is a config value, so substituting another non-Llama model is a one-line change.

---

## Constraints in the current code that shape the design

These came out of reading the pipeline, and each one changes how the evaluation must be built.

| # | Finding | Consequence for the analysis |
|---|---|---|
| C1 | `TextChunker` prepends `f"{title}\n\nAbstract: {abstract}\n\n"` to **every** chunk (`text_chunker.py:213`) | Title+abstract text is duplicated across all ~32 chunks of a paper, so BM25 over `chunk_text^3` scores every chunk of a topically-relevant paper about equally. **The single biggest confound.** Strip the header before showing chunks to the question generator, and report paper-level metrics alongside chunk-level. Also makes Q5 worth asking. |
| C2 | The indexer never populates the `chunk_id` field; `client.py` sets `chunk["chunk_id"] = hit["_id"]`, which is auto-generated and changes on reindex | Ground truth **must** key on `(arxiv_id, chunk_index)`. Never persist `_id` into the golden set. |
| C3 | `_search_hybrid_native` overwrites `results["total"] = len(results["hits"])` (`client.py:284`) | `total` is a post-filter hit count, not a corpus match count. Never usable as a recall denominator. |
| C4 | `AskRequest.top_k` is `Field(3, ge=1, le=10)` (`src/schemas/api/ask.py:10`) | `top_k=20` can't be swept over HTTP → Q2 must be answered in-process. |
| C5 | `generate_rag_answer` (`LLM/client.py:346`) hardcodes `temperature=0.7` and accepts no kwargs | Generation is unpinnable through it. The harness calls `BedrockClient.generate(prompt=..., temperature=0.0, top_p=1.0)` with a prompt from `RAGPromptBuilder` instead. |
| C6 | `/ask` uses `create_bedrock_structured_prompt`; `/stream` uses the plain `create_rag_prompt` | Two different systems under test. Evaluate the `/ask` path and **report the divergence as a finding** — the UI only ever calls `/stream`, so the streamed path never produces `confidence`/`citations`. |
| C7 | `_prepare_chunks_and_sources` (`src/routers/ask.py:57`) passes only `{arxiv_id, chunk_text}` to the LLM | Citations can only resolve to a paper → Q8 is a **paper-level** metric. Adding `chunk_index` to that dict would make it chunk-level; flag as a follow-up, not a prerequisite. |
| C8 | `text_chunker.py:122` calls `self._reconstruct_text(words, text)`; the method takes one arg | `TypeError` on any paper under `min_chunk_size` words with no usable sections. One-line fix, needed before the corpus build. |
| C9 | `hybrid_indexer.py` and `indexing/factory.py` import `from services.opensearch...` with no `src.` prefix | Only resolves inside Airflow (`PYTHONPATH=/opt/airflow/src`). Every eval entry point must put **both** the repo root and `src/` on `sys.path` before importing — centralise in one bootstrap module. |
| C10 | `pyproject.toml` sets `env_files = ".env.test"` (file absent) with no `testpaths`, and `assistant-ui/python/**/tests/` holds ~15 unrelated test files | The moment a `tests/` dir exists, pytest starts collecting the vendored assistant-ui suite. Needs `testpaths` + `norecursedirs`. Separately, `pytest-env` and `pytest-dotenv` are both in dev deps and both register `--envfile`, which aborts pytest at plugin load — dropping `pytest-env` (nothing uses its `env =` option) fixes it. |
| C11 | `hybrid_search_size_multiplier` (`src/config.py:105`) is ignored — `_search_hybrid_native` hardcodes `* 2` | A config that does nothing. Note in the findings; don't fix as part of this work. |

---

## Phase 1 — Frozen corpus

Current state is unusable for measurement: **3 papers / 96 chunks in OpenSearch, 6 papers in Postgres** (2 with empty `raw_text`). Target **30–40 papers**, ~15–20 primary `cs.AI` and ~15–20 primary `cs.LG`, ≈1,000–1,300 chunks.

> **Stated tradeoff (decision recorded):** at ~1,200 chunks, top-5 retrieval is a 1-in-240 discrimination task. Enough to rank hybrid vs BM25 vs vector and to prove the harness, but metrics will sit high and deltas under ~10 points won't resolve cleanly. The manifest format makes growing to 80 papers later a re-run, not a rewrite.

**Ingest change.** Do not alter `fetch_papers`'s signature — `MetadataFetcher` calls it positionally and the DAG depends on that shape. Add two optional pass-through params to `MetadataFetcher.fetch_and_process_papers` (`src/services/metadata_fetcher.py:54`):

- `search_query` → `ArxivClient.fetch_papers_with_query()` (`arxiv/client.py:200`), the existing raw-query escape hatch that supports `cat:cs.AI OR cat:cs.LG`
- `arxiv_ids` → loop `ArxivClient.fetch_paper_by_id()` (`arxiv/client.py:278`), the frozen-corpus path

Fall through to the existing `fetch_papers(...)` when both are `None`, so the DAG is untouched. Everything downstream (`_process_pdfs_batch`, `_store_papers_to_db`) is already category-agnostic.

**Two-phase build.** Leave the DAG's `max_results=3` hardcode (`airflow/dags/arxiv_ingestion/fetching.py:37-43`) alone — that DAG does daily *incremental* ingest, the opposite requirement to a frozen corpus. Separate ticket.

- *Discover once* (`corpus/discover.py`) — over-fetch via `fetch_papers_with_query(max_results=200)`, then filter client-side: drop papers whose `categories` contains neither target category; balance by **primary** category (`categories[0]`), since the observed corpus skews heavily cs.AI + cs.CL and a naive query won't balance; deterministic tiebreak by sorting on `arxiv_id`. Writes the manifest.
- *Build reproducibly* (`corpus/build.py`) — read manifest → `fetch_and_process_papers(arxiv_ids=[...])` → `HybridIndexingService.index_papers_batch(papers, replace_existing=True)`.

**Manifest** (`corpus/corpus_manifest.json`, committed) records the query, date window, chunking params, embedding model/dim, and per paper: `arxiv_id`, `primary_category`, `categories`, `title`, **`raw_text_sha256`**, `n_chunks`. The hash is what makes the corpus genuinely frozen — written on first build, **asserted** on every rebuild. If arXiv serves a `v2` or Docling changes its extraction, you get a loud mismatch instead of silently invalidated ground-truth `chunk_index` values.

---

## Phase 2 — Golden question set

`datasets/golden_v1.jsonl`, one question per line, committed. Per row: `qid`, `question`, `question_type`, `generated_from {arxiv_id, chunk_index, section_title}`, `relevant_chunks [{arxiv_id, chunk_index, grade}]`, `relevant_papers`, `answerable`, `reference_answer`, `must_mention`, `paraphrase_of`, `generator {model, prompt_version, temperature}`, `review {status, reviewer, reviewed_at}`.

- `(arxiv_id, chunk_index)` is the only chunk identity (C2).
- **Graded relevance** — `grade 2` = contains the answer, `grade 1` = supporting/adjacent context. Enables nDCG rather than binary-only, and matters here because of the 100-word chunk overlap.
- `relevant_papers` stored explicitly so paper-level metrics survive a re-chunk that invalidates `chunk_index`, and because C1 makes paper-level the more robust headline number.
- `must_mention` — a few literal strings — powers a cheap deterministic pre-check before the LLM judge.
- `review.status ∈ accepted | edited | rejected`; rejected rows stay for provenance, the loader filters them.

**Build pipeline:** stratified sample of 3–4 chunks per paper read **from OpenSearch** (guarantees the exact indexed text), weighted away from `section_title ∈ {References, Acknowledgments, Appendix}` and away from `chunk_index == 0` → generate with the judge-tier model at temperature 0 → automated leakage filter → **human review**. The review pass is non-negotiable and costs ~1 hour for 120 questions; an unreviewed LLM-generated golden set measures the generator, not the retriever.

### The lexical-leakage trap

Generate a question from chunk text and BM25 wins trivially, because the question reuses the chunk's rare terms — this is the most common way a RAG eval quietly measures nothing. Four counter-measures:

1. **Strip the prepended title/abstract header before showing the chunk to the generator.** Reuse `strip_prepended_abstract()` already written at `testing/hybrid_search/search_queries_test/hybrid_search.py:47`. Otherwise the generator writes abstract-flavoured questions that match every chunk of the paper (C1).
2. **Instruct paraphrase explicitly** — *"Ask using different vocabulary from the excerpt. Do not reuse distinctive multi-word phrases. Prefer the field's generic terminology over the authors' coined names."*
3. **Automated leakage filter** (`dataset/leakage.py`) — reject if >15% of the question's content-word 3-grams appear in the source chunk; also reject any question containing a token occurring in exactly one chunk corpus-wide (a giveaway hapax). Highest-value quality gate in the harness.
4. **Explicit paraphrase pairs** — a second surface form with the *same* `relevant_chunks`, linked by `paraphrase_of`. This is how Q4 gets answered: a genuinely semantic config scores near-identically across the pair, a BM25-flattered one diverges.

### Composition (~120 questions)

| Type | N | Ground truth |
|---|---|---|
| `single_hop` factual | 55 | 1–2 chunks, 1 paper |
| `multi_hop` cross-paper | 20 | ≥2 chunks across ≥2 papers — prompt with both excerpts, *"ask one question answerable from neither alone"* |
| `comparative` | 15 | 2+ chunks, 2 papers on adjacent topics |
| `unanswerable` (negative control) | 20 | `relevant_chunks: []`. Two flavours: out-of-corpus (verify absence with a hybrid search; keep only if top-1 score is low) and counterfactual (mutate a real fact) |
| `paraphrase` pairs | 10 | same as parent |

The 20 unanswerables are what make Q9 measurable — without them, a system that never refuses scores perfectly.

At n≈120, Recall@5 has a standard error of ~4.6 points, so the analysis honestly resolves deltas of roughly 10 points or more. Smaller effects need the larger corpus.

---

## Phase 3 — Retrieval analysis

**Metrics** (pure functions, `(ranked, relevant, k) -> float`): `recall_at_k` (primary), `hit_rate_at_k`, `precision_at_k`, `reciprocal_rank`, `ndcg_at_k` (the only one using graded relevance), plus `paper_recall_at_k` / `paper_mrr` reported alongside because of C1. Report `mean ± 1.96·SEM`. `answerable: false` rows skip retrieval metrics entirely.

**Dispatch — direct service calls, no HTTP** (so the run needs only OpenSearch + Jina, not the API container, and isn't capped by C4):

| mode | call |
|---|---|
| `bm25` | `search_unified(q, query_embedding=None, size=k, use_hybrid=False)` → `_search_bm25_only` (`client.py:213`) |
| `vector` | `search_chunks_vector(emb, size=k)` (`client.py:135`) |
| `hybrid` | `search_unified(q, query_embedding=emb, size=k, use_hybrid=True)` → `_search_hybrid_native` (`client.py:244`) |

**Sweep A (12 runs)** — `mode ∈ {bm25, vector, hybrid} × top_k ∈ {3,5,10,20}`. Answers **Q1, Q2, Q4** and produces the latency profile.

**Sweep B** — one lever at a time from the Sweep A winner:

- `rank_constant ∈ {10,20,60,100}` (**Q3**) — add a `rank_constant` param to `_create_rrf_pipeline` (`client.py:92`) rather than mutating the `HYBRID_RRF_PIPELINE` module constant
- `bm25_fields` with and without `title^2`/`abstract^1` (**Q5**)
- `fuzziness: AUTO` vs `None` (**Q6**) — `query_builder.py:110` currently sets `fuzziness: "AUTO", prefix_length: 2`
- `chunk_size × overlap` — **expensive, do last**: full re-chunk + re-embed + re-index, and it invalidates every ground-truth `chunk_index`. Re-map ground truth by matching `start_char`/`end_char` spans (already on `ChunkMetadata`) rather than regenerating questions.

Retrieval is deterministic → one run per query per config. **Embed each query once globally** and cache by `sha256(question)`, or Sweep A alone costs 12 × 120 = 1,440 redundant Jina calls.

**Outputs** — `retrieval_raw.jsonl` (per-question ranked lists + per-question metrics, so any metric can be re-derived without re-querying) and `retrieval_summary.json` per run, plus `results/leaderboard.md`, **appended never overwritten** and committed. That file is what makes this a harness rather than a one-off measurement:

```
| run_id | config | R@5 | P@5 | nDCG@5 | MRR | Hit@5 | paperR@5 | p50 ms | p95 ms |
|--------|--------|-----|-----|--------|-----|-------|----------|--------|--------|
| 20260810-01 | hybrid k=5 rc=60 | 0.71 ±.05 | 0.34 | 0.68 | 0.62 | 0.89 | 0.85 | 141 | 402 |
| 20260810-02 | bm25   k=5       | 0.58 ±.05 | 0.27 | 0.54 | 0.49 | 0.78 | 0.79 |  38 |  96 |
```

Lift the existing `time_call(fn, runs=N)` timing helper from `testing/hybrid_search/search_queries_test/hybrid_search.py`.

---

## Phase 4 — Generation analysis

**System under test, pinned:** Sweep A winning config → `RAGPromptBuilder.create_bedrock_structured_prompt` (what `/ask` actually uses, C6) → `meta.llama3-70b-instruct-v1:0` at `temperature=0.0, top_p=1.0` via `BedrockClient.generate()` (C5) → `ResponseParser.parse_structured_response`. Run in-process so the retrieved `chunk_index` survives alongside the answer.

**(a) Faithfulness (Q7).** Judge call 1: answer → `{"claims": [...]}`. Judge call 2: each claim → `supported | contradicted | not_in_context`, against the retrieved context **only** — never show the judge the reference answer, or it grades against the reference instead of the context. Score = supported/total; report `contradicted_count` **separately**, since one contradiction is worse than several merely-unsupported claims and averaging hides it.

**(b) Answer relevance.** 0–3 against `question` + `reference_answer`: does it answer *this* question. Deliberately not "is it true in the world" — that is (a)'s job.

**(c) Citation accuracy (Q8) — deterministic, no LLM.** Regex `\[arXiv:([^\]]+)\]`, normalize by stripping the `v\d+` suffix exactly as `ask.py:65` does. Then `citation_precision` (cited IDs present in the retrieved set — a citation to an unretrieved paper is a hallucinated citation, the headline number), `citation_recall`, and `citation_present_rate` (answers with ≥1 citation, despite `prompts/rag_system.txt` demanding them).

**(d) Refusal correctness (Q9).** Classify `refused | attempted`, then `refusal_recall` = refused/unanswerable and `false_refusal_rate` = refused/answerable — the second guards against gaming (a) by refusing everything.

Report (a)–(d) as a table. There is no single scalar, and inventing one hides the (a)↔(d) tradeoff.

**Judge design.** Rubric prompts as versioned text files under `generation/prompts/`, matching the repo's existing file-backed prompt convention (`src/services/LLM/prompts/rag_system.txt`). Rules enforced in code:

1. Hard `assert JUDGE_MODEL != generator_model` at runner start.
2. `temperature=0.0, top_p=1.0` — `_build_inference_config` (`client.py:118`) already reads both from kwargs, so no client change is needed.
3. Structured JSON requested in the prompt (Converse has no schema param), same technique as `create_bedrock_structured_prompt`; strip ```` ```json ```` fences before `json.loads`, then mirror `ResponseParser._extract_json_fallback`.
4. **Rubric anchors, not adjectives** — every scale point gets a definition and a one-line example. Ask for `{"score": int, "reason": "<=25 words"}`; the forced short reason reduces score drift and gives you something to audit.
5. Bias controls — one answer at a time, no pairwise comparison, never reveal the generator's identity.
6. **Judge-agreement check, once.** Sample 40 judged items, human-label them, report Cohen's κ in `results/judge_calibration.md`. **If κ < 0.6 the rubric is broken and every downstream number is decoration.** This is the step most harnesses skip.
7. Cache judgments by `sha256(judge_model + prompt_version + prompt_text)`, so re-running after a retrieval-only change doesn't re-pay.

Cost: ~120 questions × ~3 judge calls ≈ 360 judge calls per full run. Iterate rubrics on Haiku 4.5, report on Sonnet 4.6.

---

## Phase 5 — End-to-end (Q11)

The same golden set through `POST /api/v1/ask` over HTTP. Paper-level Recall@k derived from the returned `sources` (C7) should sit within noise of the in-process paper-level number for the same config. A gap means `_prepare_chunks_and_sources` diverges from what was measured — most likely its silent BM25 degradation path (`ask.py:31-34`) firing on a Jina error. Log `search_mode` from every response and confirm it reads `"hybrid"`.

---

## Deliverables

1. `results/leaderboard.md` — the append-only config comparison table (committed).
2. `results/judge_calibration.md` — Cohen's κ against human labels (committed).
3. A findings write-up answering Q1–Q11, including the code-level findings surfaced along the way (C6 prompt divergence, C11 dead config, the `pytest-env`/`pytest-dotenv` collision in C10).
4. `corpus/corpus_manifest.json` + `datasets/golden_v1.jsonl` — the frozen, version-controlled inputs that make every number reproducible.

---

## File layout

The harness lives under `src/evaluation/` (alongside this document).

```
src/evaluation/
├── docs/evaluation_plan.md        # this document
├── _bootstrap.py                  # dual sys.path insert (C9) — imported first, always
├── _timing.py  _text.py           # lifted from testing/hybrid_search/... (note origin in docstring)
├── config.py                      # JUDGE_MODEL, GENERATOR_MODEL, paths, run_id
├── report.py                      # markdown render + append to leaderboard.md
├── corpus/     discover.py  build.py  corpus_manifest.json      # manifest COMMITTED
├── dataset/    sample.py  generate.py  leakage.py  review.py  prompts/qgen_v1.txt
├── datasets/   golden_v1.jsonl                                   # COMMITTED
├── retrieval/  metrics.py  configs.py  runner.py
├── generation/ judge.py  citations.py  runner.py  prompts/{faithfulness,relevance,refusal}_v1.txt
├── e2e/        runner.py
└── results/                       # gitignored except leaderboard.md + judge_calibration.md

tests/                             # repo root; hermetic — no services, no network, <5s in CI
├── conftest.py
├── test_metrics.py                # hand-worked Recall/nDCG/MRR cases incl. an oracle run
├── test_leakage.py                # known leaky/clean question pairs
├── test_citations.py              # [arXiv:...] parsing incl. version-suffix stripping
├── test_golden_schema.py          # every row validates; no dup qids; chunk_index >= 0
├── test_judge_parsing.py          # fenced/unfenced/garbage JSON -> fallback
└── test_chunker_regression.py     # pins the C8 fix
```

Entry points are all `python -m src.evaluation.<module>` with argparse, so the bootstrap always runs. `src/` has no `__init__.py` and resolves as a namespace package, which is how the existing `from src.services...` imports already work. Makefile targets: `eval-corpus`, `eval-dataset`, `eval-retrieval`, `eval-generation`, `eval-e2e`, `eval`.

`pyproject.toml` needs `testpaths`, `norecursedirs`, and `markers` (C10), plus a `.env.test` with a committed `.env.test.example`. The existing `testing/` tree stays untouched — migrate nothing, copy two helpers.

---

## Verification

Each layer verifies independently, in order. Do not proceed past a failing step.

**V0 — prerequisites.** OpenSearch and Postgres healthy; a one-line Bedrock `converse` call to `us.anthropic.claude-sonnet-4-6` at `temperature=0` succeeds. If it doesn't, Phase 4 is blocked regardless of everything else.

**V1 — corpus.** ≥30 papers in Postgres with zero empty `raw_text`; unique `arxiv_id` count in OpenSearch equals the manifest length; both categories present with ≥15 each. Then **re-run the build** — it must be a clean no-op with every `raw_text_sha256` assertion passing. That is the proof the corpus is frozen.

**V2 — golden set.** Schema test passes; type counts match the composition table. Spot-check 10 random answerable rows by running the question through hybrid search and confirming the declared `relevant_chunks` appear in the top-10. If they never do, the **ground truth** is wrong, not the retriever.

**V3 — metrics correctness.** Unit tests pass, and they **must include an oracle run**: feed the ground-truth chunks in as if retrieved → Recall@k = nDCG@k = MRR = 1.0. A metrics bug that *inflates* scores is otherwise invisible against real output.

**V4 — retrieval sweep.** 12 rows in `leaderboard.md`. Sanity gates: Recall monotonic in k; p50 latency bm25 < vector < hybrid; paraphrase-pair gap small for hybrid/vector. If hybrid *loses* to BM25 on nDCG@5, that is a real finding about `rank_constant` (Q3), not necessarily a bug — investigate rather than assume.

**V5 — generation.** Judge run twice with the cache disabled → **identical scores** (if not, `temperature=0` isn't reaching the model). `judge_calibration.md` exists with κ ≥ 0.6. At least one `citation_precision < 1.0` case, manually confirmed as a genuine hallucinated citation, proving the metric fires.

**V6 — end-to-end.** Paper-level Recall@k via `/ask` within noise of the in-process figure; `search_mode == "hybrid"` on every response.

**V7 — regression gate.** Re-run the retrieval sweep after any change to `text_chunker.py`, `query_builder.py`, `index_config_hybrid.py`, or `jina_client.py`, and diff against `leaderboard.md`.

---

## Sequencing

| # | Step | Blocks | Rough effort |
|---|---|---|---|
| 1 | Submit the Anthropic use-case form in the Bedrock console | Phase 4 | minutes + external wait — **start first** |
| 2 | Fix the `_reconstruct_text` arity bug (C8) | Corpus build | one line |
| 3 | `_bootstrap.py` + pytest config (C9, C10) | Everything | small |
| 4 | `MetadataFetcher` params → discover → **commit manifest** → build | Phases 2–5 | ~1h code, 30–45 min ingest |
| 5 | `metrics.py` + unit tests — no services needed, do while the corpus builds | Phase 3 | ~2h |
| 6 | Dataset generate → leakage filter → human review | Phases 3–5 | ~2h code + ~1h review time |
| 7 | Retrieval Sweep A → leaderboard | Q1, Q2, Q4 | ~2h |
| 8 | Judge + generation runner + calibration | Q7–Q10 | ~3h + calibration labelling |
| 9 | E2E runner, Sweep B, findings write-up | Q3, Q5, Q6, Q11 | ~2h |

Steps 1–4 are the critical path — nothing is measurable until the corpus is real.
