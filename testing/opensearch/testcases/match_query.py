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

# Match Query - Search in title field
print("MATCH QUERY - Single Field Search")
print("=" * 40)

query = {
    "query": {
        "match": {
            "title": "Self-Explanation"
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