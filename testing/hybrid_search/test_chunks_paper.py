# Test chunk counts per paper in the OpenSearch hybrid chunk index
from opensearchpy import OpenSearch

import os as _os, sys as _sys
# Ensure the repo root is importable so `src.*` resolves when run directly.
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "pyproject.toml")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

# Top-level (hybrid) client -> targets the chunk index: {index_name}-{chunk_index_suffix}
from src.services.opensearch.factory import make_opensearch_client

print("CHUNKS PER PAPER (hybrid chunk index)")
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

index_name = opensearch_client.index_name
print(f"Index: {index_name}\n")

if not opensearch_client.client.indices.exists(index=index_name):
    print(f"⚠️  Index '{index_name}' does not exist yet. Run the hybrid indexing DAG first.")
    raise SystemExit(0)

# Terms aggregation on arxiv_id -> number of chunks per paper.
# (size=0 => don't return documents, just the aggregation buckets.)
agg_body = {
    "size": 0,
    "aggs": {
        "chunks_per_paper": {
            "terms": {
                "field": "arxiv_id",
                "size": 1000,          # up to 1000 distinct papers
                "order": {"_count": "desc"},
            }
        }
    },
}

response = opensearch_client.client.search(index=index_name, body=agg_body)

total_chunks = response["hits"]["total"]["value"]
buckets = response["aggregations"]["chunks_per_paper"]["buckets"]

if not buckets:
    print("No chunks found in the index.")
    raise SystemExit(0)

print(f"{'arxiv_id':<20} {'chunks':>7}")
print("-" * 28)
for bucket in buckets:
    print(f"{bucket['key']:<20} {bucket['doc_count']:>7}")

print("-" * 28)
print(f"{'papers: ' + str(len(buckets)):<20} {total_chunks:>7}  (total chunks)")
