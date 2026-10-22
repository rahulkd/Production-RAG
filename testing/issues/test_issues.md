# Test Issues Log — Hybrid Search / Chunk Index

Issues found while testing the OpenSearch hybrid chunk index (`arxiv-papers-chunks`
/ `arxiv_evaluation_index`) via the scripts in `testing/hybrid_search/`.

_Last updated: 2026-08-15_

---

## 1. Title + abstract is prepended to every chunk  ⚠️ High impact

**What:** The chunker prepends `"{title}\n\nAbstract: {abstract}\n\n"` to the
`chunk_text` of *every* chunk of a paper.

**Impact:**
- **Vector search:** every chunk of a paper embeds into nearly the same region of
  vector space, so *within-paper* ranking is weakly discriminated. Observed cosine
  scores bunched in a ~0.009 range (e.g. 0.7463 / 0.7405 / 0.7373).
- **BM25:** the abstract's terms appear in all chunks, inflating and flattening
  scores and reducing discrimination.
- Retrieval fails when the relevant content is NOT in the abstract. Query
  `"evaluation protocol for semantic faithfulness"` returned the wrong paper at
  ranks 1–2 because the target paper's abstract is about "editing diagrams," while
  "faithfulness" lived only in body sections that got drowned out.

**Evidence:** `search_queries_test/vector_search.py`, `test_paper_chunks.py`
(prepended abstract visible at the top of chunk #0 and #1).

**Suggested fix:** Do not prepend the abstract to every chunk. Index `title` /
`abstract` as their own fields (already in the mapping) and embed only the chunk's
own text, or prepend the title alone.

---

## 2. Reference / bibliography chunks dominate results  ⚠️ High impact

**What:** Chunks that are reference lists (`section_title` like `References`,
`References (Part 4)`) are keyword-dense with paper titles and rank at the top for
both BM25 and vector search, crowding out substantive content.

**Impact:** For `"reinforcement learning"`, the BM25 top-2 were both `References`
sections; the genuinely relevant Conclusion chunk ranked #3. Only 4/96 chunks
contained the phrase, and 2 of the top 3 were reference dumps.

**Evidence:** `search_queries_test/bm25_keyword.py`, `vector_search.py`.

**Suggested fix:** Exclude reference/bibliography sections at index time, or add a
query-time `must_not` filter on `section_title`. Hybrid (RRF) mitigates but does
not eliminate this.

---

## 3. `section_title` is mis-detected on some chunks  ⚠️ Medium

**What:** Section detection assigns wrong values. Example: for paper
`2604.05432v1`, chunk #1's `section_title` is `"Shichao Pei"` — an author name, not
a section heading. Reference chunks also get split into arbitrary
`References (Part 1..6)` labels.

**Impact:** Makes any `section_title`-based filtering (see issue #2) unreliable, and
pollutes the per-section metadata.

**Evidence:** `chunking/results/2604.05432v1_chunks.txt` (chunk #1).

**Suggested fix:** Review the section-splitting logic in the chunker; validate
detected section titles against a heading pattern before storing.

---

## 4. Query analyzer keeps noise words (partial stopword removal)  ⚠️ Medium

**What:** The `text_analyzer` uses the default `_english_` stoplist, which removes
`is` but keeps `me`, `where`, `find`, `present`, etc. Combined with
`multi_match` `operator: "or"`, these non-content words match unrelated chunks.

**Impact:** Conversational queries like `"find me where transformer architecture is
present"` search on tokens `[find, me, where, trasnform, architecur, present]` —
four of six are noise that can pull in irrelevant chunks.

**Evidence:** `_analyze` output on `arxiv-papers-chunks` (documented while testing
the BM25 query builder).

**Suggested fix:** Extend the stopword list (add `find`, `show`, `me`, `where`,
`present`, …) and/or switch BM25 to `operator: "and"` or a `minimum_should_match`
so stray words don't each count as a match.

---

## 5. Jina API key empty inside the Airflow container  ⚠️ High (ops)

**What:** Hybrid indexing DAG failed every paper with
`Error embedding passages: Illegal header value b'Bearer '` → `0 chunks indexed`.
Root cause: the running `rag-airflow` container had `JINA_API_KEY` empty even though
`.env` has it, because Docker Compose only reads `env_file` at container creation.

**Impact:** Chunks were created but stored with no embeddings; task still exited 0
(errors caught per-paper), so the failure was silent.

**Evidence:** DAG log for `index_papers_hybrid` (2026-07-18);
`docker exec rag-airflow` showed `JINA_API_KEY` unset while `docker compose config`
resolved it.

**Suggested fix:** Recreate the container: `docker compose up -d --force-recreate
airflow`. Code hardening: `JinaEmbeddingsClient.__init__` should raise a clear error
on an empty key instead of building an illegal `Bearer ` header.

---

## 6. Hardcoded / leaked Jina API key  ⚠️ High (security)

**What:** A live Jina API key was hardcoded in `testing/hybrid_search/
test_embedding_api.py` and as the default for `jina_api_key` in `src/config.py:120`.

**Impact:** The key is exposed in source and session history; must be treated as
compromised.

**Status:** Removed from `test_embedding_api.py` (now reads from `JINA_API_KEY` via
`.env`). `src/config.py:120` default still contains the key.

**Suggested fix:** Rotate the key in the Jina dashboard. Replace the `config.py`
default with `""` (or make it required) so the secret is not in source.
