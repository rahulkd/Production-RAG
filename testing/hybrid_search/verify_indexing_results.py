# Verify Hybrid Chunk Index Results
from opensearchpy import OpenSearch

import os as _os, sys as _sys
# Ensure the repo root is importable so `src.*` resolves when run directly.
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "pyproject.toml")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

# Top-level (hybrid) client -> targets the chunk index: {index_name}-{chunk_index_suffix}
from src.services.opensearch.factory import make_opensearch_client

print("VERIFYING HYBRID CHUNK INDEX")
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

print(f"Index: {opensearch_client.index_name}")

stats = opensearch_client.get_index_stats()

if not stats or 'error' in stats:
    error = stats.get('error', 'unknown error') if stats else 'no stats returned'
    print(f"✗ Could not retrieve index stats: {error}")
elif not stats.get('exists'):
    print(f"⚠️  Index '{stats.get('index_name')}' does not exist yet")
    print("\nRun setup_indices() and the Airflow hybrid indexing DAG first.")
else:
    chunk_count = stats.get('document_count', 0)

    if chunk_count > 0:
        print(f"✓ Success! Found {chunk_count} chunks in the hybrid index")
        print(f"  Size: {stats.get('size_in_bytes', 0):,} bytes")

        # Show sample chunks (empty query -> match_all on the chunk index)
        sample = opensearch_client.search_papers("", size=min(chunk_count, 5))
        first_chunk_id = None
        if sample.get('hits'):
            print("\nSample chunks:")
            for i, chunk in enumerate(sample['hits'], 1):
                if first_chunk_id is None:
                    first_chunk_id = chunk.get('chunk_id')
                arxiv_id = chunk.get('arxiv_id', 'unknown')
                chunk_index = chunk.get('chunk_index', '?')
                title = (chunk.get('title') or 'Unknown')[:50]
                text = (chunk.get('chunk_text') or '').replace("\n", " ")[:80]
                print(f"  {i}. [{arxiv_id} #{chunk_index}] {title}")
                print(f"       text: {text}...")

        # Verify the embedding vector is actually stored for a given chunk id.
        # (search _source excludes 'embedding', so fetch the doc directly by _id.)
        if first_chunk_id:
            print(f"\nEmbedding check for chunk id: {first_chunk_id}")
            try:
                doc = opensearch_client.client.get(
                    index=opensearch_client.index_name,
                    id=first_chunk_id,
                    _source_includes=["embedding"],
                )
                embedding = doc.get("_source", {}).get("embedding")
                if embedding and isinstance(embedding, list):
                    expected_dim = opensearch_client.settings.opensearch.vector_dimension
                    dim_ok = "✓" if len(embedding) == expected_dim else "✗"
                    print(f"  ✓ Embedding stored: {len(embedding)} dims "
                          f"(expected {expected_dim} {dim_ok})")
                    preview = ", ".join(f"{v:.4f}" for v in embedding[:5])
                    print(f"       first values: [{preview}, ...]")
                else:
                    print("  ✗ No embedding stored on this chunk "
                          "(indexing likely ran without a valid JINA_API_KEY)")
            except Exception as e:
                print(f"  ✗ Could not fetch chunk to check embedding: {e}")

        # Distinct papers represented in the chunk index
        distinct = {c.get('arxiv_id') for c in opensearch_client.search_papers("", size=chunk_count).get('hits', [])}
        distinct.discard(None)
        print(f"\nChunks span {len(distinct)} distinct paper(s): {', '.join(sorted(distinct)) or 'n/a'}")
    else:
        print("⚠️  No chunks in the hybrid index yet")
        print("\nRun the Airflow hybrid indexing DAG first (embeddings + chunk indexing).")
