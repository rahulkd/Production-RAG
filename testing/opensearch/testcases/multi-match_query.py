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



# Multi-Match Query - Search across multiple fields
print("MULTI-MATCH QUERY - Search Multiple Fields")
print("=" * 40)

query = {
    "query": {
        "multi_match": {
            "query": "AI Agents",
            "fields": ["title^2", "abstract", "authors"],  # ^2 boosts title field
            "type": "best_fields"
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
    print(f"Title: {hit['_source']['title'][:70]}...")
    print(f"Score: {hit['_score']:.2f}")
    print(f"Authors: {', '.join(hit['_source']['authors'][:2])}...\n")