# Verify Data Pipeline Results
from opensearchpy import OpenSearch

from src.services.opensearch.factory import make_opensearch_client

print("VERIFYING DATA PIPELINE")
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

stats = opensearch_client.get_index_stats()

if stats and 'error' not in stats:
    doc_count = stats.get('document_count', 0)

    if doc_count > 0:
        print(f"✓ Success! Found {doc_count} documents in OpenSearch")

        # Show sample papers (empty query -> match_all)
        sample = opensearch_client.search_papers("", size=3)
        if sample.get('hits'):
            print("\nSample papers:")
            for i, paper in enumerate(sample['hits'], 1):
                title = paper.get('title', 'Unknown')[:60]
                print(f"  {i}. {title}...")
    else:
        print("⚠️  No documents in OpenSearch yet")
        print("\nPlease run the Airflow DAG first (see instructions above)")
else:
    error = stats.get('error', 'unknown error') if stats else 'no stats returned'
    print(f"✗ Could not retrieve index stats: {error}")
