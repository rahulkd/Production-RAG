# Retrieval evaluation: findings and issues

**Run:** `20261022-135717` (Sweep A) · [full report](retrieval_20261022-135717/report.md) · [metric definitions](metrics_def.md)

## Summary

1. **Hybrid search does not beat BM25 alone.** The scores differ by 0.03 or less at every k, well inside the ±0.06–0.09 confidence ranges. Per question, hybrid and BM25 tie on 84 of 90.
2. **The RRF fusion step adds nothing so far.** Merging the two result lists never recovers a chunk that both BM25 and vector missed. At k=3 it drops 6 answer chunks that one of the two legs had found.
3. **Vector search alone is the weakest mode.** nDCG@5 is 0.44 vs 0.61–0.62 for the others, and it is worse than BM25 on 24 of 90 questions.
4. **`top_k=3`, the production default, loses about 30% of answers.** Hybrid Recall rises from 0.69 at k=3 to 0.88 at k=10.
5. **At k=3, no two-chunk question ever gets both chunks.** Hybrid finds both chunks for 0 of 17 multi-hop and comparative questions.
6. **The retriever often finds the right paper but the wrong chunk.** Paper Recall@3 is 0.93 vs chunk Recall@3 of 0.69.

## Run setup

| Item | Value |
|---|---|
| Dataset | `golden_v1.reviewed.jsonl`: 152 rows, of which 110 were accepted and 90 are answerable and scored |
| Scored by type | single_hop 65 · multi_hop 13 · paraphrase 8 · comparative 4 |
| Index | `arxiv_evaluation_index`: 1,036 chunks from 43 papers |
| Configs | {bm25, vector, hybrid} × k ∈ {3, 5, 10, 20} = 12 |
| Hybrid settings | RRF `rank_constant=60`, BM25 fields as in production `QueryBuilder` |

## Results

| mode | Recall@3 | Recall@5 | Recall@10 | Recall@20 | nDCG@5 | MRR@10 | Paper Recall@3 |
|---|---|---|---|---|---|---|---|
| bm25 | **0.706** | **0.794** | **0.889** | 0.933 | **0.619** | 0.675 | 0.922 |
| vector | 0.517 | 0.622 | 0.761 | 0.828 | 0.440 | 0.487 | 0.822 |
| hybrid | 0.694 | 0.778 | 0.878 | 0.933 | 0.613 | **0.701** | **0.928** |

The confidence range on each value is ±0.05–0.10, so differences under about 0.08 between modes are not meaningful.

---

## Findings

### F1. Hybrid search is not measurably better than BM25

| k | Hybrid better than BM25 | Hybrid worse | Tie |
|---|---|---|---|
| 3 | 2 | 4 | 84 |
| 5 | 3 | 6 | 81 |
| 10 | 2 | 4 | 84 |

Hybrid's only consistent edge is a slightly higher MRR at k ≥ 5 (0.685–0.701 vs 0.662–0.677), and even that is inside the noise. Vector search runs on every query and adds latency, but it doesn't change which chunks come back.

### F2. RRF fusion only loses chunks, never gains them

| k | Hybrid missed, though BM25 or vector found it | Hybrid found it, though neither did |
|---|---|---|
| 3 | 6 | 0 |
| 5 | 5 | 0 |

So far fusion has only cost answer chunks. When the vector leg ranks a wrong chunk highly, it can push a correct BM25 result out of the top k.

### F3. Vector search alone is substantially weaker

- It is worse than BM25 per question on 24 of 90 at k=3 and k=5, and better on only 5–6.
- Paper Recall stays flat at 0.82 for k=3–10. Even with more results, vector search often never reaches the right paper.
- It is the fastest mode (p50 3–7 ms vs 60–100 ms), but that doesn't matter while its results are this weak.

### F4. `top_k=3` cuts off too many answers

| | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| Hybrid Recall | 0.694 | 0.778 | 0.878 | 0.933 |
| Hybrid Hit | 0.756 | 0.844 | 0.933 | 0.956 |

Where hybrid first places an answer chunk, using the k=20 run:

| Rank of first answer chunk | 1 | 2–3 | 4–5 | 6–10 | 11–20 | not in top 20 |
|---|---|---|---|---|---|---|
| Questions | 51 | 21 | 7 | 3 | 4 | 4 |

80% of answer chunks rank in the top 3, and 88% in the top 5. The retriever usually finds the answer. At `top_k=3`, the API's default and the only value the UI uses, about 1 in 4 questions reaches the generator without its answer chunk.

### F5. Two-chunk questions fail at small k

Questions that need 2 chunks from 2 papers (13 multi-hop and 4 comparative, 17 total):

| mode | Both found, k=3 | Neither found, k=3 | Both found, k=10 | Neither found, k=10 |
|---|---|---|---|---|
| bm25 | 2 / 17 | 6 | 8 / 17 | 1 |
| vector | 0 / 17 | 10 | 2 / 17 | 4 |
| hybrid | **0 / 17** | 6 | 6 / 17 | 1 |

Recall by type for hybrid at k=3: single_hop 0.79, multi_hop 0.35, comparative 0.25. With k=3, at most 3 chunks can come back, and they tend to cluster in one paper. So the generator rarely sees both halves of the evidence for a cross-paper question.

### F6. The right paper is found, but not the right chunk

- With hybrid at k=3, 25 of 90 questions retrieve a chunk from the correct paper but not the answer chunk.
- Most of those same-paper chunks are far from the answer chunk, not next to it:

| Distance from answer chunk | ±1 (adjacent) | 2–3 chunks | > 3 chunks |
|---|---|---|---|
| Retrieved same-paper chunks | 7 | 14 | 40 |

If the 100-word overlap between neighbouring chunks were the cause, the misses would be adjacent chunks. They are mostly far away, which fits issue C1 below: every chunk carries the paper's title and abstract, so all of a paper's chunks match a topical query about equally well.

### F7. Paraphrase test can't say anything yet

With only 8 pairs, every paraphrase gap is inside a ±0.13–0.37 range. Q4 ("is the retriever matching meaning or just words?") stays unanswered until there are more paraphrase pairs.

---

## Issues found

### Retrieval pipeline

| ID | Issue | Evidence | Impact | Severity |
|---|---|---|---|---|
| R1 | **Vector search adds no value inside hybrid** | F1, F2: 84/90 ties with BM25; fusion never gains a chunk and loses 5–6 | Embedding cost and latency on every query with no quality gain; can push correct BM25 results out | High |
| R2 | **`top_k=3` default is too small** | F4: Recall@3 0.69 vs Recall@10 0.88 | About 25–30% of answers never reach the generator | High |
| R3 | **Multi-hop and comparative questions get almost no full evidence at k=3** | F5: 0/17 both chunks found (hybrid) | Answers to cross-paper questions are incomplete | High |
| R4 | **Title and abstract are repeated in every chunk (C1)** | F6: right paper 93%, right chunk 69%; misses mostly > 3 chunks away | BM25 scores all chunks of a relevant paper about equally, so ranking within the paper is weak | High |
| R5 | **Vector search alone is weak** | F3: nDCG@5 0.44; worse than BM25 on 24/90 | Either the Jina embeddings of the full chunk text (header included) don't separate chunks well, or the questions favour exact wording (see E1) | Medium |
| R6 | **RRF `rank_constant=60` hasn't been tested** | F2: fusion loses chunks one leg found | The default may give the weaker vector leg too much weight on this corpus | Medium |
| R7 | **Hybrid results change with `top_k`** | Hybrid MRR@20 (0.695) < MRR@10 (0.701); each leg fetches `size × 2` candidates | Changing `top_k` silently changes ranking quality, not just list length | Low |
| R8 | **BM25 is slower than vector search** | p50 55–103 ms vs 3–7 ms | The plan expected the opposite (V4). Likely highlighting and `fuzziness: AUTO` in `QueryBuilder`. Hybrid inherits this cost. | Low |

### Code issues found while building the harness

| ID | Issue | Location | Impact | Severity |
|---|---|---|---|---|
| H1 | `search_unified` and `search_chunks_vector` catch every exception and return an empty result | `src/services/opensearch/client.py` | In production, an OpenSearch failure looks like "no relevant documents" rather than an error. The harness avoids these two methods for this reason. | Medium |
| H2 | `.env` points `OPENSEARCH__HOST` at the Docker hostname `opensearch` | `.env` | Every host-run evaluation needs `OPENSEARCH__HOST=http://localhost:9200` | Low |
| H3 | `pytest-env` is still installed in the venv and conflicts with `pytest-dotenv` (C10) | `.venv` | `pytest` aborts at plugin load unless run with `-p no:env` | Low |
| H4 | Results files are gitignored except `leaderboard.md` | `.gitignore` | `metrics_def.md` and this file won't be committed from `results/` | Low |

### Evaluation dataset

| ID | Issue | Evidence | Impact | Severity |
|---|---|---|---|---|
| E1 | **The questions may favour BM25** | Questions are written by an LLM from chunk text. The leakage filter removes heavy copying, but some shared wording remains. | BM25's lead over vector, and hybrid's lack of gain, may be partly a dataset effect. Real user queries may behave differently. | Medium |
| E2 | **Small groups for multi-hop, comparative and paraphrase** | n = 13, 4, 8 | Per-type results for these groups have wide error bars; F5 and F7 are directional only | Medium |
| E3 | **Some questions don't point to any specific paper** | q0053 ("How many new questions did GPT-4 generate…") retrieves no correct paper in any mode. Its answer chunk is a dataset table in an appendix (`D.1 Overview`), and several corpus papers cover the same math-distillation topic. q0171 ("What smaller models underwent fine-tuning…") is similarly generic. | These count as retrieval failures, but no retriever could reliably answer them | Medium |
| E4 | **Appendix chunks passed the eligibility filter** | q0053's answer chunk is in section `D.1 Overview`. The exclusion pattern matches the word "appendix", not appendix sections numbered `A.1`, `D.1` and so on. | Questions can be built from table and appendix text that isn't representative of user queries | Low |
| E5 | **7 questions are never fully retrieved by any mode, even at k=20** | multi_hop 3, comparative 2, single_hop 1, paraphrase 1. Three get zero recall everywhere: q0053, q0146, q0171. | Either hard cases or label problems. q0146 retrieves the right papers but chunks 1–2 away from the labelled ones. These need a manual check before they are treated as retriever failures. | Low |

## Not yet measured

- RRF `rank_constant`, BM25 field boosts and `fuzziness` (Sweep B: Q3, Q5, Q6).
- Whether these retrieval gaps cause bad answers. That is the generation evaluation (Phase 4, Q7–Q10).
- Whether the deployed `/ask` API matches these in-process numbers (Phase 5, Q11).
