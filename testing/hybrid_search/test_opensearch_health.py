# Test Unified Search System
from src.services.opensearch.factory import make_opensearch_client_fresh
from opensearchpy import OpenSearch

print("UNIFIED SEARCH SYSTEM TEST")
print("=" * 40)

# Create unified OpenSearch client
opensearch_client = make_opensearch_client_fresh()

# Configure for notebook execution
opensearch_client.host = "http://localhost:9200"
opensearch_client.client = OpenSearch(
    hosts=["http://localhost:9200"],
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False,
)

# Check index health
stats = opensearch_client.get_index_stats()
print(f"Index: {stats['index_name']}")
print(f"Documents: {stats['document_count']}")
print(f"Health: {'Healthy' if opensearch_client.health_check() else 'Unhealthy'}")

if stats['document_count'] > 0:
    print("\n✓ Index contains data. Ready for search testing!")
else:
    print("\n⚠ Index is empty. Please run the Airflow DAG first:")
    print("  1. Open http://localhost:8080 (admin/admin)")
    print("  2. Trigger 'arxiv_paper_ingestion' DAG")
    print("  3. Wait for completion (~10 minutes)")