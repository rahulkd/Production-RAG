# Simple BM25 Search
from opensearchpy import OpenSearch

from src.services.opensearch.factory import make_opensearch_client

print("SIMPLE BM25 SEARCH")
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

# Change this to any word from your papers
search_term = "Label-Self Compatibility"  # Try different terms!  ## free text search

print(f"Searching for: '{search_term}'\n")

results = opensearch_client.search_papers(
    query=search_term,
    size=3
)

if results.get('hits'):
    print(f"Found {results.get('total', 0)} total matches\n")
    
    for i, paper in enumerate(results['hits'], 1):
        print(f"{i}. {paper.get('title', 'Unknown')[:70]}...")
        print(f"   Score: {paper.get('score', 0):.2f}")
        print(f"   arXiv ID: {paper.get('arxiv_id', 'N/A')}\n")
else:
    print("No results found. Try searching for:")
    print("  • 'neural', 'model', 'algorithm'")
    print("  • Use '*' to see all papers")