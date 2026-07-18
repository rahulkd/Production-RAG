# Test BM25 Keyword Search (hybrid chunk index)
from opensearchpy import OpenSearch

import os as _os, sys as _sys
# Ensure the repo root is importable so `src.*` resolves when run directly.
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "pyproject.toml")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

# Top-level (hybrid) client -> BM25 search runs against the chunk index.
from src.services.opensearch.factory import make_opensearch_client

print("BM25 KEYWORD SEARCH TEST")
print("=" * 40)

# Create OpenSearch client using factory pattern
opensearch_client = make_opensearch_client()

# Override for local execution (localhost instead of container hostname)
opensearch_client.host = "http://localhost:9200"
opensearch_client.client = OpenSearch(
    hosts=["http://localhost:9200"],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

print(f"Index: {opensearch_client.index_name}\n")

test_queries = [
    #"machine learning",
    #"reinforcement learning",
    "robot policies",
]

for query in test_queries:
    print(f"Query: '{query}'")
    try:
        results = opensearch_client.search_papers(
            query=query,
            size=3,
        )

        print(f"  Found: {results.get('total', 0)} results")

        for i, hit in enumerate(results.get('hits', [])[:3], 1):
            arxiv_id = hit.get('arxiv_id', 'N/A')
            chunk_index = hit.get('chunk_index', '?')
            title = (hit.get('title') or 'N/A')[:50]
            score = hit.get('score', 0) or 0
            chunk_text = hit.get('chunk_text') or '(no chunk_text)'
            print(f"    {i}. [{arxiv_id} #{chunk_index}] {title}... (score: {score:.2f})")
            print(f"       ----- chunk content -----")
            print(f"{chunk_text}")
            print(f"       -------------------------")

    except Exception as e:
        print(f"  Error: {e}")
    print()

print("✓ BM25 search completed!")
