from opensearchpy import OpenSearch

import os as _os, sys as _sys
# Ensure the repo root is importable so `src.*` resolves when run directly.
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "pyproject.toml")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

from src.services.opensearch.keyword.factory import make_opensearch_client
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


# Filter Query - Filter by categories
print("FILTER QUERY - Category Filtering")
print("=" * 40)

query = {
    "query": {
        "bool": {
            "must": [
                {
                    "match": {
                        "abstract": "neural"
                    }
                }
            ],
            "filter": [
                {
                    "terms": {
                        "categories": ["cs.AI"]
                    }
                }
            ]
        }
    },
    "size": 3
}

response = opensearch_client.client.search(
    index=opensearch_client.index_name,
    body=query
)

print(f"Found {response['hits']['total']['value']} results\n")

for hit in response['hits']['hits']:
    title = hit['_source']['title'][:70]
    categories = ', '.join(hit['_source']['categories'])
    print(f"Title: {title}...")
    print(f"Categories: {categories}")
    print(f"Score: {hit['_score']:.2f}\n")