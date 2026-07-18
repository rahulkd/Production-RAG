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


# Combined Query - Complex search with multiple criteria
print("COMBINED QUERY - Complex Search")
print("=" * 40)

query = {
    "query": {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": "transformer",
                        "fields": ["title^3", "abstract"],
                        "type": "best_fields"
                    }
                }
            ],
            "filter": [
                {
                    "range": {
                        "published_date": {
                            "gte": "2024-01-01"
                        }
                    }
                }
            ],
            "should": [
                {
                    "match": {
                        "categories": "cs.AI"
                    }
                }
            ]
        }
    },
    "sort": [
        "_score",
        {"published_date": {"order": "desc"}}
    ],
    "size": 3
}

response = opensearch_client.client.search(
    index=opensearch_client.index_name,
    body=query
)

print(f"Complex Query:")
print(f"  • Must contain 'transformer' (title boosted 3x)")
print(f"  • Filter: published after 2024-01-01")
print(f"  • Prefer: cs.AI category")
print(f"  • Sort: by relevance, then date\n")

print(f"Found {response['hits']['total']['value']} results\n")

for hit in response['hits']['hits']:
    title = hit['_source']['title'][:70]
    pub_date = hit['_source']['published_date'][:10]
    score = hit['_score']
    categories = ', '.join(hit['_source']['categories'][:2])
    
    print(f"Title: {title}...")
    print(f"  Date: {pub_date} | Score: {score:.2f}")
    print(f"  Categories: {categories}\n")