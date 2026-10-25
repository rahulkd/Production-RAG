# Generation evaluation: findings and issues

**Run:** `20261024-083836` · `hybrid k=10` · [full report](generation_20261024-083836/hybrid_k10/report.md) · [retrieval findings](findings_issues.md)

## Verdict

**Not ready for production as "answers users can trust and rely on".** In the configuration that was tested, the generator behaves well. But the deployed system runs a different configuration on all four main settings (§ Deployment gap). The judge behind the headline scores has not been calibrated. And about 1 answer in 14 contains a claim that contradicts the source papers. Fix the deployment gap, calibrate the judge, then re-run against the *deployed* path.

**Update after Phase 5 (end-to-end over HTTP, run `e2e_20261024-115413`):** the API's code path is now validated (§ Phase 5). Both routes retrieve exactly what the harness retrieves, always in hybrid mode, at a p95 latency of 3.8 s or less, and `/stream` (the route the UI uses) is as good as `/ask` while citing far more often. But Phase 5 also found that **the deployed `rag-api` searches an empty index and answers nothing**, which blocks deployment outright. The verdict stands until that, the judge calibration, and the 0.7 temperature are fixed.

## Summary

1. **Out-of-scope questions were never answered with made-up facts.** All 20 unanswerable questions were refused (refusal recall 1.000). This is evidence on 20 easy probes, not proof that hallucination is zero (F1).
2. **About 1 answer in 14 contains a contradiction.** 6 of 87 answers (6.9%) include at least one claim that the retrieved text contradicts, and 5.7% of all claims are not supported. 5 of the 6 had the answer chunk in context (F2).
3. **Multi-hop is the weak spot.** Faithfulness is 0.750 on multi-hop questions versus 0.977 on single-hop. 11 of the 16 not_in_context claims come from multi-hop answers (F3).
4. **Grounded doesn't mean correct.** When retrieval misses the paper, Llama can answer fluently about a *different* paper and still score 1.0 on faithfulness (F4).
5. **Citations are never made up, but are often missing.** All 44 answers with inline citations cite only retrieved papers, but 60% of answers carry no inline citation at all (F5).
6. **The deployed path was not measured.** Users get `/stream` with a plain prompt, Llama 3 70B, k=3 and temperature 0.7. This run measured `/ask` with a JSON prompt, Llama 3.3, k=10 and temperature 0 (F6).
7. **The judge is uncalibrated.** Every faithfulness and relevance number depends on DeepSeek V3.2, which has not been checked against human labels and has visibly erred in both directions (F7).

## Run setup

| Item | Value |
|---|---|
| Dataset | `golden_v1.reviewed.jsonl`: 110 accepted rows; 90 answerable, 20 unanswerable |
| By type | single_hop 65 · multi_hop 13 · paraphrase 8 · comparative 4 · unanswerable 20 |
| Index | `arxiv_evaluation_index`: 1,036 chunks from 43 papers |
| Retrieval | Hybrid (RRF, `rank_constant=60`), k=10 |
| Generator | `us.meta.llama3-3-70b-instruct-v1:0`, temperature 0, `/ask` structured JSON prompt |
| Judge | `deepseek.v3.2`, temperature 0, rubric v1. **Not calibrated** |
| Run health | 0 generation errors, 0 judge errors, 0 truncated answers; 357 judge calls |
| Prompt size | ~8,500 input tokens per answer on average |

## Results

| Metric | Value (95% CI) | n | Question |
|---|---|---|---|
| Faithfulness (supported claims / claims) | 0.943 ±.028 | 87 answers, 389 claims | Q7 |
| Answers with ≥1 contradicted claim | 0.069 ±.054 (6 answers) | 87 | Q7 |
| Answer relevance (0–3) | 2.733 ±.145 | 90 | — |
| Answered (relevance ≥ 2) | 0.944 ±.048 | 90 | — |
| `must_mention` coverage (no LLM) | 0.565 ±.070 | 90 | — |
| Inline citation precision | 1.000 | 44 | Q8 |
| Answers with ≥1 inline citation | 0.400 ±.092 | 110 | Q8 |
| Refusal recall (unanswerable refused) | 1.000 (20/20) | 20 | Q9 |
| False refusal rate (answerable refused) | 0.033 (3/90) | 90 | Q9 |
| Valid structured JSON | 0.982 (108/110) | 110 | C6 |

| Type | n | Faithfulness | Relevance | Refused |
|---|---|---|---|---|
| single_hop | 65 | 0.977 | 2.862 | 0.015 |
| paraphrase | 8 | 0.975 | 2.250 | 0.000 |
| comparative | 4 | 0.889 | 2.250 | 0.250 |
| multi_hop | 13 | **0.750** | 2.538 | 0.077 |

---

## Reading the headline numbers correctly

Three readings that over-state what the run shows:

**"Refusal recall 1.0 means 0 hallucinations."** No. These measure different things:
- **What refusal recall shows:** all 20 questions with *no* answer in the corpus were declined.
- **What it doesn't show:** whether the answers to the 90 answerable questions are accurate. Hallucination there shows up as contradicted claims: 6 answers had at least one.
- **The sample is small and easy.** With 20 out of 20, the true refusal rate could plausibly be as low as ~84%. And the probes invent obviously absent topics ("F1 of a Vision Transformer on protein structure"). Near-miss questions, where the corpus holds closely related content, were not tested.
- **What is true:** no invented citations (0 in 110 answers) and no fabricated answers to out-of-scope questions.

**"Faithfulness 0.943 can be trusted for real-time use."** It's an encouraging score, but not yet a trustworthy one:
- **The judge is uncalibrated** (F7). Until κ ≥ 0.6 against human labels, 0.943 is the judge's opinion, not a measurement.
- **It's an average that hides the weak types.** Multi-hop is 0.750, and 6.9% of answers carry a contradiction. A user can't tell which answer is the bad one.
- **It measures grounding, not correctness.** q0171 scores 1.0 while answering about the wrong paper (F4).

**"Relevance 2.733 is reliable."** Also judge-dependent, and it disagrees with the deterministic check:
- **22 of the 75 answers scored 3/3 mention fewer than half of the expected `must_mention` strings.** Part of this is the check being too literal ("9.5× fewer *average* trainable parameters" fails "9.5× fewer trainable parameters"). Part may be a generous judge. Calibration settles which.
- **It's scored against Qwen-written reference answers**, which no human has verified.

---

## Findings

### F1. Refusal is strong on clear out-of-scope questions; false refusals are rare

- **Refusals:** 20 of 20 unanswerable questions refused; 0 attempted, so 0 claims from out-of-scope answers.
- **False refusals:** 3 of 90 (3.3%), each with a different cause:

| qid | Type | Answer chunk retrieved? | Cause |
|---|---|---|---|
| q0053 | single_hop | No (recall 0) | Retrieval. Refusing was the correct behaviour given the context |
| q0113 | multi_hop | Yes (recall 1.0) | Generation. The evidence was present; the model wouldn't combine it |
| q0133 | comparative | Half (recall 0.5) | Mixed. Only one side of the comparison was retrieved |

### F2. About 1 answer in 14 contradicts its sources

The 6 answers with a contradicted claim:

| qid | Type | Contradicted claim (summary) | Judge's call |
|---|---|---|---|
| q0008 | single_hop | AFLoRA "more computationally efficient" than LoRA; the excerpt says *comparable* | Correct |
| q0031 | single_hop | "Figure 9 illustrates GraLoRA reduces gradient deviation" (2 claims) | **Likely too strict**: the figure shows the distributions |
| q0100 | multi_hop | AFLoRA "adapts the rank"; it freezes matrices, and rank adaptation is AdaLoRA | Correct |
| q0111 | multi_hop | VAE and NF-VAE "do not share" an encoder-decoder structure; they do | Correct |
| q0114 | multi_hop | Equation-of-Thought "executes symbolic programs"; it models problems as equations | Correct |
| q0120 | multi_hop | Optimal transport "does not enforce equality constraints"; the excerpt says it does | Correct |

- 5 of the 6 had the answer chunk in context, so these are **generation errors, not retrieval errors**.
- The pattern: under synthesis pressure, the model **swaps attributes between similar methods** or **reverses a negation**.
- These errors read fluently and confidently; a user can't spot them.

### F3. Multi-hop is where faithfulness drops

- Faithfulness on multi-hop is **0.750**, versus 0.977 on single-hop.
- 4 of the 6 contradiction cases are multi-hop.
- 11 of the 16 not_in_context claims are multi-hop: the model fills gaps between two papers with its own connective reasoning.
- Only 13 multi-hop and 4 comparative questions were tested, so the exact numbers are rough; the direction is consistent.

### F4. A grounded answer can still be the wrong answer

- **q0171** ("What smaller models underwent fine-tuning, and how many parameters did each have?"):
  - Retrieval missed the target paper.
  - Llama answered fully from a different paper (LLaMA 3.1/3.2, Gemma 2) with a citation.
  - Faithfulness was **1.0**; relevance was **0**.
- **q0172:** the same pattern. It described LoRA generally from the wrong paper.
- **Why:** the system has no step that checks whether the context actually matches the question's intent. Short or vague questions make this more likely; both are paraphrase questions.
- Faithfulness can't catch this, so it should never be read alone as "trustworthy".

### F5. Citations are never invented, but often absent

- **Inline:** precision is 1.000 (44 of 44 answers cite only retrieved papers), and 0 hallucinated IDs across 110 answers.
- **But 60% of answers have no inline `[arXiv:id]`**, despite `rag_system.txt` asking for citations. Citation recall is 0.456.
- **The JSON `citations` field** has an arXiv ID in 76% of answers (precision 1.000), but it isn't shown to users:
  - `AskResponse` doesn't return it.
  - The UI's `/stream` path never produces it.
  - The "sources" the API returns are every retrieved paper, not the papers the answer used. At k=10 that's up to 10 links for a one-paper answer.

### F6. The deployed path was not measured

| | This run | Production today |
|---|---|---|
| Endpoint and prompt | `/ask`, `create_bedrock_structured_prompt` (JSON) | **`/stream`**, `create_rag_prompt` (plain): the only path the UI calls |
| Model | Llama 3.3 70B (128K) | **Llama 3 70B (8K)**: `rag-api` started 2026-10-23 and hasn't been restarted since the switch |
| top_k | 10 | **3** (`AskRequest` and frontend `DEFAULT_TOP_K`) |
| Sampling | temperature 0 | **temperature 0.7, top_p 0.9** (hard-coded in `generate_rag_answer` and `generate_rag_answer_stream`) |

- Each of these can move the scores.
- **k=3 alone is likely to hurt.** Retrieval Hit@3 is 0.756 versus 0.933 at k=10. In this run, when the answer chunk was missed, only 3 of 6 questions were answered acceptably. Roughly 1 in 4 production questions starts without the evidence.

### F7. The judge is uncalibrated and errs in both directions

- **Too strict:** q0031's "Figure 9 illustrates…" claims were marked contradicted although the figure shows the distributions in question.
- **Too lenient:** in an earlier side-by-side on q0100, DeepSeek marked "AFLoRA adapts the rank" supported where Kimi K3 correctly marked it contradicted.
- **Not done yet:**
  - the 40-item human calibration (κ ≥ 0.6 per rubric);
  - the run-twice determinism check (`--no-judge-cache`).

### F8. 2% of answers would show raw JSON to the user

- **What happens:** q0103 and q0175 contain LaTeX (`$9.5\times$`). `\t…` and similar sequences are invalid JSON escapes, so `json.loads` fails. `ResponseParser._extract_json_fallback` then returns the **raw JSON string** as the answer, and the user sees `{"answer": "Yes, dividing…`.
- **Who it affects:** any answer quoting maths notation from a paper.
- **Scope:** affects `/ask` only, since `/stream` doesn't use JSON.

---

## Phase 5 — end-to-end over HTTP

**Run:** `e2e_20261024-115413` · [report](e2e_20261024-115413/report.md) · [leaderboard](e2e_leaderboard.md)

All 110 golden questions went through the real `/api/v1/ask` and `/api/v1/stream` routes at k=3 (what the UI sends) and k=10: 440 requests in total. The API under test ran the current code (Llama 3.3) on :8001 against the evaluation corpus. The routes keep their hardcoded temperature 0.7, so each answer is one sample.

### F9. The deployed API answers nothing

- The compose `rag-api` (:8000) searches `arxiv-papers-chunks`, which holds **0 documents**. The whole corpus is in `arxiv_evaluation_index`.
- A live request to either route returns the canned "couldn't find any relevant information" reply, with `chunks_used: 0`.
- Every user question currently gets a non-answer, whatever the model or prompt.
- The Phase 5 runner now checks for this before starting and stops.

### F10. The API's retrieval matches the harness exactly (Q11)

| | /ask k=3 | /stream k=3 | /ask k=10 | /stream k=10 |
|---|---|---|---|---|
| HTTP errors | 0 | 0 | 0 | 0 |
| `search_mode = hybrid` | 110/110 | 110/110 | 110/110 | 110/110 |
| Source papers = harness papers | 110/110 | 110/110 | 110/110 | 110/110 |
| Paper recall (API = in-process) | 0.928 | 0.928 | 0.972 | 0.972 |

- **No silent BM25 fallback** fired in 440 requests.
- **The routes' chunk handling doesn't diverge** from what Phases 3–4 measured.
- So the retrieval numbers from Phases 3–4 apply to the served API.

### F11. Latency is fit for interactive use

| Seconds | /ask k=3 | /stream k=3 | /ask k=10 | /stream k=10 |
|---|---|---|---|---|
| p50 total | 1.9 | 1.9 | 2.4 | 2.5 |
| p95 total | 2.5 | 3.0 | 3.8 | 3.4 |
| Time to first token, p50 / p95 | – | 1.3 / 2.3 | – | 1.7 / 2.1 |

- k=10 adds about 0.5 s at the median.
- The first streamed text arrives in under 2 s for half of requests and about 2 s at p95.
- Caveat: measured at 4 concurrent requests on one worker. There was no load test.

### F12. k=10 is clearly better than k=3 on the served routes

| | /ask k=3 → k=10 | /stream k=3 → k=10 |
|---|---|---|
| Relevance (0–3) | 2.30 → **2.67** | 2.43 → **2.66** |
| Answered (relevance ≥ 2) | 78.9% → **92.2%** | 82.2% → **90.0%** |
| False refusal rate | 12.2% → **3.3%** | 8.9% → **2.2%** |

- At k=3, about 1 in 10 answerable questions is refused because the evidence wasn't retrieved.
- The relevance confidence intervals at k=3 and k=10 don't overlap on `/ask` and barely touch on `/stream`.
- Faithfulness stays in the 0.94–0.96 range at both cutoffs.

### F13. `/stream` is as good as `/ask` and cites far more

| | /ask k=10 | /stream k=10 |
|---|---|---|
| Faithfulness | 0.946 ±.031 | 0.952 ±.022 |
| Relevance | 2.67 | 2.66 |
| Answers with inline citation | 41.8% | **86.4%** |
| Hallucinated citations | 0 | 0 |

- The plain prompt (`/stream`) gets the model to cite inline. The JSON prompt (`/ask`) moves citations into a field nobody sees (F5).
- So the route the UI already uses is the better one for trust. The C6 divergence works in `/stream`'s favour.
- **The answer formats are inconsistent.** `/stream` sometimes cites as `[arXiv:id, Section 2.1]`, `[2. arXiv:id]` or a bare `[5]`. That should be standardised in the prompt.

### F14. Phase 5 confirmed the earlier weaknesses on the live routes

- **Contradictions:** 5.0–10.3% of attempted answers contain one, depending on config. At temperature 0.7 these rates move between samples, so the config differences are within the noise.
- **Multi-hop:** still the weakest type, with faithfulness from 0.71 to 0.91.
- **Near-miss unanswerable question:**
  - q0166 asks why the authors disabled *gradient* checkpointing; the paper discusses disabling *layerwise activation* checkpointing.
  - It was answered in 3 of the 4 configs. Llama corrected the premise rather than refusing, but each answer had one contradicted claim.
  - This is exactly the near-miss case the Phase 4 probes didn't cover (#12).
- **Raw JSON reached a client:** one `/ask` answer at k=10 came through as raw JSON (F8, confirmed on the live route).

### Harness fixes made during Phase 5

- **The judge's JSON parser repairs invalid backslash escapes.** DeepSeek quotes LaTeX such as `$0.85\%$` in its reasons; without the repair those verdicts were dropped as parse errors.
- **The citation extractor ignores section locators.** It used to count `Section 2.1` in `[arXiv:id, Section 2.1]` as a made-up paper ID, which reported 8 non-existent "hallucinated citations" in `/stream`. All saved runs were rescored; Phase 4 wasn't affected.

---

## Issues and improvements for the RAG pipeline

Ordered by impact on whether users can trust an answer.

### Blocking: before any "trustworthy" claim

| # | Issue | Fix |
|---|---|---|
| 0 | **The deployed API searches an empty index (F9)** | **Switch added, not yet applied to the container.** `OPENSEARCH__ACTIVE_INDEX` (`.env`) selects `production` or the now-frozen `evaluation` index; `.env` is currently `production`, which is empty. For production, ingest papers into `arxiv-papers-chunks` through the Airflow DAG. To run the API on the frozen corpus meanwhile, set `evaluation` and `docker compose up -d --build api`. Then confirm with `python -m src.evaluation.e2e.runner --base-url http://localhost:8000 --limit 5`, whose preflight fails on an empty index. |
| 1 | The deployed path isn't the one measured (F6) | **Partly closed by Phase 5:** both routes were evaluated with the current code over HTTP (F10–F13). Still open: re-run against the *deployed* container once #0, #2 and #3 are done. |
| 2 | ~~Temperature 0.7 in production~~ | **Done in code (2026-10-25), not yet deployed.** `generate_rag_answer` and `generate_rag_answer_stream` now sample at `BEDROCK__RAG_TEMPERATURE` / `BEDROCK__RAG_TOP_P` (default 0.0 / 1.0, the values Phase 4 measured; the evaluation harness reads the same settings). Rebuild the `api` container to apply. |
| 3 | `rag-api` and `rag-frontend` still run the old code and model | Rebuild and restart: `docker compose up -d --build api frontend`. |
| 4 | Uncalibrated judge (F7) | `calibrate sample --run 20261024-083836 --cutoff hybrid_k10`, label the 40 items, then `calibrate score`. Fix the rubric if κ < 0.6. |

### High: correctness

| # | Issue | Fix |
|---|---|---|
| 5 | ~~k=3 starves the generator~~ (F6, F12; retrieval F4) | **Done in code (2026-10-25), not yet deployed.** `AskRequest.top_k` and the frontend `DEFAULT_TOP_K` now default to 10 (the API still caps at 10). Phase 5 measured the cost: about +0.5 s at the median, p95 3.4 s on `/stream`; in return false refusals drop from 8.9% to 2.2% and the answered rate rises from 82% to 90%. Rebuild `api` and `frontend` to apply. |
| 6 | Contradictions in multi-hop (F2, F3) | Add instructions to the prompt: "attribute each claim to the excerpt it came from; do not combine properties of different methods". Consider a claim-check pass for multi-paper answers. |
| 7 | Grounded but wrong answers (F4) | Make the model say which paper it is answering from, and refuse when the excerpts are about a different study. Add a relevance threshold on retrieval scores. |
| 8 | Raw JSON shown to users (F8) | In `ResponseParser`, double invalid backslash escapes before parsing (or use a tolerant parser), and never return text that starts with `{"answer"`. Add a regression test with the q0103 output. |

### Medium: trust signals for users

| # | Issue | Fix |
|---|---|---|
| 9 | 60% of `/ask` answers have no inline citation (F5) | `/stream` already cites in 86–97% of answers (F13). Keep `/stream` as the user-facing route, standardise the citation format to `[arXiv:id]` in `rag_system.txt`, and align or retire the `/ask` JSON prompt (C6). |
| 10 | "Sources" list every retrieved paper | Show only the papers the answer cited. |
| 11 | The title and abstract repeat in every chunk (C1) | This inflates every prompt (up to 10 copies of the same abstract at k=10) and blurs which passage supports a claim. Strip it from `chunk_text` sent to the LLM, or send it once per paper. |

### Evaluation gaps to close

| # | Gap | Fix |
|---|---|---|
| 12 | Unanswerable probes are easy, n=20 | Add near-miss unanswerable questions: on-topic but missing the specific fact. |
| 13 | Small multi-hop (13) and comparative (4) samples | Generate more, so these types can be reported with usable confidence intervals. |
| 14 | No k=3 baseline in this run | `--top-k 3 10` reuses the cache and gives the production comparison directly. |
| 15 | Reference answers have no human review | Spot-check 15 answers against their `chunk_text`; `must_mention` strings are too literal for some rows. |
| 16 | ~~No latency measurement~~ | **Done in Phase 5 (F11).** Still open: a load test at expected concurrency. |

## Path to a production decision

1. Fix the empty production index (#0), restart the services (#3), and set production temperature (#2) and top_k=10 (#5).
2. Fix the JSON escape bug (#8) and tighten the prompt (#6, #7, #9).
3. Re-run Phase 5 against the deployed container (`--base-url http://localhost:8000`) with these settings (#1).
4. Calibrate the judge (#4). Only then are faithfulness and relevance decision-grade numbers.
5. Ship when **calibrated** faithfulness on multi-hop and the rate of answers with a contradiction meet a bar you set in advance (for example, ≤3% of answers contradicted), with refusal recall holding on near-miss probes (#12).
