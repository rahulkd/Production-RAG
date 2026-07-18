# OpenSearch Client Setup
import os as _os, sys as _sys
# Ensure the repo root is importable so `src.*` resolves when run directly.
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "pyproject.toml")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

from src.services.opensearch.keyword.factory import make_opensearch_client
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