# Delete the OpenSearch index (drops all data + mapping)
from opensearchpy import OpenSearch

import os as _os, sys as _sys
# Ensure the repo root is importable so `src.*` resolves when run directly.
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "pyproject.toml")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

from src.services.opensearch.keyword.factory import make_opensearch_client

print("DELETING OPENSEARCH INDEX")
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

# Drop the entire index (data + mapping). It will be recreated on next ingestion.
if opensearch_client.client.indices.exists(index=index_name):
    stats = opensearch_client.get_index_stats()
    before = stats.get("document_count", 0) if stats and "error" not in stats else 0
    print(f"Index '{index_name}' exists with {before} documents")

    response = opensearch_client.client.indices.delete(index=index_name)
    if response.get("acknowledged"):
        print(f"✓ Deleted index '{index_name}'")
    else:
        print(f"✗ Failed to delete index: {response}")
else:
    print(f"Index '{index_name}' does not exist. Nothing to delete.")
