# Comparison report: BM25 vs pure vector vs hybrid (RRF) on the chunk index.
# - prints top-3 chunk content per approach (with the prepended title/abstract stripped)
# - time-profiles each approach's query search time
# - saves a Markdown report to ./results/
import asyncio
import re
import time
from datetime import datetime

from opensearchpy import OpenSearch

import os as _os, sys as _sys
# Ensure the repo root is importable so `src.*` resolves when run directly.
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "pyproject.toml")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

from src.services.opensearch.factory import make_opensearch_client
from src.services.embeddings.factory import make_embeddings_client

# --- Setup ----------------------------------------------------------------
opensearch_client = make_opensearch_client()
opensearch_client.host = "http://localhost:9200"
opensearch_client.client = OpenSearch(
    hosts=["http://localhost:9200"],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)
embeddings_client = make_embeddings_client()

RESULTS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "results")
TIMING_RUNS = 5  # search calls are timed over this many runs (min + mean reported)

test_queries = [
    #"evaluation protocol for semantic faithfulness",
    "truncated backpropagation through time and dagger distillation",
]


# --- Helpers --------------------------------------------------------------
def strip_prepended_abstract(chunk_text, title=None, abstract=None):
    """Remove the prepended '{title}\\n\\nAbstract: {abstract}' preamble the chunker
    adds to every chunk, so only the chunk's own content is shown."""
    if not chunk_text:
        return chunk_text or ""
    paragraphs = re.split(r"\n\s*\n", chunk_text)
    kept, still_preamble = [], True
    for para in paragraphs:
        p = para.strip()
        if still_preamble:
            if title and p == title.strip():
                continue
            if p.startswith("Abstract:"):
                continue
            still_preamble = False  # first real paragraph -> keep everything after
        kept.append(para)
    return "\n\n".join(kept).strip()


def label(hit):
    return (f"[{hit.get('arxiv_id', 'N/A')} #{hit.get('chunk_index', '?')}] "
            f"{hit.get('section_title') or '(no section)'}")


def time_call(fn, runs=TIMING_RUNS):
    """Return (min_ms, mean_ms) for a sync callable, after one warm-up run."""
    fn()  # warm-up (caches, JIT of connection, etc.)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return min(samples), sum(samples) / len(samples)


async def time_embed(query, runs=3):
    """Return (embedding, min_ms, mean_ms) for the async query embedding."""
    emb = await embeddings_client.embed_query(query)  # warm-up + value we reuse
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        await embeddings_client.embed_query(query)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return emb, min(samples), sum(samples) / len(samples)


def render_hits(hits, score_label):
    """Markdown for the top-3 hits with stripped content."""
    lines = []
    for i, h in enumerate(hits[:3], 1):
        title = (h.get("title") or "N/A")[:60]
        score = h.get("score", 0) or 0
        content = strip_prepended_abstract(h.get("chunk_text"), h.get("title"), h.get("abstract"))
        lines.append(f"**{i}. {label(h)}** — {title} ({score_label}: {score:.4f})\n")
        lines.append("```")
        lines.append(content)
        lines.append("```\n")
    return "\n".join(lines)


# --- Main -----------------------------------------------------------------
async def run_query(query):
    report = []
    report.append(f"# Search comparison report\n")
    report.append(f"- **Query:** `{query}`")
    report.append(f"- **Index:** `{opensearch_client.index_name}`")
    report.append(f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}")
    report.append(f"- **Timing:** min / mean over {TIMING_RUNS} runs (embedding over 3)\n")

    # Embed once (reused by vector + hybrid), and time it.
    embedding, emb_min, emb_mean = await time_embed(query)

    # Fetch results once (for display) ...
    bm25 = opensearch_client.search_papers(query=query, size=3)
    vector = opensearch_client.search_chunks_vector(query_embedding=embedding, size=3)
    hybrid = opensearch_client.search_chunks_hybrid(query=query, query_embedding=embedding, size=3, min_score=0.0)

    # ... and time each search engine call (embedding excluded here, added below).
    bm25_min, bm25_mean = time_call(lambda: opensearch_client.search_papers(query=query, size=3))
    vec_min, vec_mean = time_call(lambda: opensearch_client.search_chunks_vector(query_embedding=embedding, size=3))
    hyb_min, hyb_mean = time_call(
        lambda: opensearch_client.search_chunks_hybrid(query=query, query_embedding=embedding, size=3, min_score=0.0)
    )

    # --- Timing table ---
    report.append("## Time profile (ms)\n")
    report.append("| Approach | Embed (mean) | Search (min / mean) | Total (mean) |")
    report.append("|---|---|---|---|")
    report.append(f"| BM25 (keyword) | – | {bm25_min:.1f} / {bm25_mean:.1f} | {bm25_mean:.1f} |")
    report.append(f"| Vector (kNN) | {emb_mean:.1f} | {vec_min:.1f} / {vec_mean:.1f} | {emb_mean + vec_mean:.1f} |")
    report.append(f"| Hybrid (RRF) | {emb_mean:.1f} | {hyb_min:.1f} / {hyb_mean:.1f} | {emb_mean + hyb_mean:.1f} |")
    report.append("")
    report.append(f"_Query embedding (Jina API): {emb_min:.1f} / {emb_mean:.1f} ms — shared by vector & hybrid._\n")

    # --- Ranking comparison ---
    report.append("## Top-3 ranking comparison\n")
    report.append("| Rank | BM25 | Vector | Hybrid (RRF) |")
    report.append("|---|---|---|---|")
    for i in range(3):
        b = label(bm25["hits"][i]) if i < len(bm25.get("hits", [])) else "–"
        v = label(vector["hits"][i]) if i < len(vector.get("hits", [])) else "–"
        h = label(hybrid["hits"][i]) if i < len(hybrid.get("hits", [])) else "–"
        report.append(f"| {i + 1} | {b} | {v} | {h} |")
    report.append("")

    # --- Full content per approach (abstract preamble stripped) ---
    report.append(f"## BM25 (keyword) — {bm25.get('total', 0)} hits\n")
    report.append(render_hits(bm25.get("hits", []), "bm25"))
    report.append(f"## Vector (kNN) — {vector.get('total', 0)} hits\n")
    report.append(render_hits(vector.get("hits", []), "cos"))
    report.append(f"## Hybrid (RRF) — {hybrid.get('total', 0)} hits\n")
    report.append(render_hits(hybrid.get("hits", []), "rrf"))

    return "\n".join(report)


async def main():
    _os.makedirs(RESULTS_DIR, exist_ok=True)
    for query in test_queries:
        print(f"Running comparison for: '{query}' ...")
        report = await run_query(query)

        slug = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:50]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = _os.path.join(RESULTS_DIR, f"{slug}_{stamp}.md")
        with open(path, "w") as fh:
            fh.write(report)

        print(report)
        print(f"\n✓ Report saved to: {_os.path.relpath(path, _root)}")

    await embeddings_client.close()


if __name__ == "__main__":
    asyncio.run(main())
