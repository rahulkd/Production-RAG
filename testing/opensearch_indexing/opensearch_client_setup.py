# OpenSearch Client Setup
from src.services.opensearch.factory import make_opensearch_client
from opensearchpy import OpenSearch

print("OPENSEARCH CLIENT SETUP")
print("=" * 40)

# Create OpenSearch client using factory pattern
opensearch_client = make_opensearch_client()

# Override for notebook execution (localhost instead of container hostname)
opensearch_client.host = "http://localhost:9200"
opensearch_client.client = OpenSearch(
    hosts=["http://localhost:9200"],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

print(f"Client configured with host: {opensearch_client.host}")
print(f"Index name: {opensearch_client.index_name}")

# Test health check
is_healthy = opensearch_client.health_check()
if is_healthy:
    print("✓ OpenSearch health check: PASSED")
else:
    print("✗ OpenSearch health check: FAILED")