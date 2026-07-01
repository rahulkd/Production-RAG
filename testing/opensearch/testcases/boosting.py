from opensearchpy import OpenSearch

from src.services.opensearch.factory import make_opensearch_client
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

# Boosting Query - Promote and demote results
print("BOOSTING QUERY - Promote/Demote Results")
print("=" * 40)

query = {
    "query": {
        "boosting": {
            "positive": {
                "match": {
                    "abstract": "deep learning"
                }
            },
            "negative": {
                "match": {
                    "abstract": "multimodal"
                }
            },
            "negative_boost": 0.1  # Reduce score of negative matches
        }
    },
    "size": 3
}

response = opensearch_client.client.search(
    index=opensearch_client.index_name,
    body=query
)

print(f"Query: Boost 'deep learning', demote 'survey' papers\n")
print(f"Found {response['hits']['total']['value']} results\n")

for hit in response['hits']['hits']:
    title = hit['_source']['title'][:70]
    abstract_snippet = hit['_source']['abstract'][:100]
    print(f"Title: {title}...")
    print(f"Score: {hit['_score']:.2f}")
    print(f"Abstract: {abstract_snippet}...\n")