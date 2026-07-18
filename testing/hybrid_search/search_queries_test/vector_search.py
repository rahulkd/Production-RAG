# Test pure vector (kNN) search on the hybrid chunk index
import asyncio

from opensearchpy import OpenSearch

import os as _os, sys as _sys
# Ensure the repo root is importable so `src.*` resolves when run directly.
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "pyproject.toml")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

# Top-level (hybrid) client -> vector search runs against the chunk index.
from src.services.opensearch.factory import make_opensearch_client
from src.services.embeddings.factory import make_embeddings_client

print("PURE VECTOR (kNN) SEARCH TEST")
print("=" * 40)

# OpenSearch client (targets the chunk index)
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

# Embeddings client (needs JINA_API_KEY from .env) to embed the query
embeddings_client = make_embeddings_client()

print(f"Index: {opensearch_client.index_name}\n")

test_queries = [
    "evaluation protocol for semantic faithfulness",
]


async def main():
    for query in test_queries:
        print(f"Query: '{query}'")
        try:
            # 1) Embed the query with Jina (retrieval.query task -> 1024-d vector)
            query_embedding = await embeddings_client.embed_query(query)
            print(f"  Query embedded: {len(query_embedding)} dims")

            # 2) Pure kNN search on the 'embedding' field (no BM25 involved)
            results = opensearch_client.search_chunks_vector(
                query_embedding=query_embedding,
                size=3,
            )

            print(f"  Found: {results.get('total', 0)} results")

            for i, hit in enumerate(results.get('hits', [])[:3], 1):
                arxiv_id = hit.get('arxiv_id', 'N/A')
                chunk_index = hit.get('chunk_index', '?')
                section = hit.get('section_title') or '(no section)'
                title = (hit.get('title') or 'N/A')[:50]
                score = hit.get('score', 0) or 0
                chunk_text = hit.get('chunk_text') or '(no chunk_text)'
                print(f"    {i}. [{arxiv_id} #{chunk_index}] {title}... "
                      f"(score: {score:.4f}, section: {section})")
                print(f"       ----- chunk content -----")
                print(f"{chunk_text}")
                print(f"       -------------------------")

        except Exception as e:
            print(f"  Error: {e}")
        print()

    await embeddings_client.close()
    print("✓ Vector search completed!")


if __name__ == "__main__":
    asyncio.run(main())
