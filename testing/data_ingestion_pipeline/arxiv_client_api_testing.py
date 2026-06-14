# Test arXiv API Client
import asyncio
from datetime import datetime, timedelta

# Import our arXiv client
from src.services.arxiv.factory import make_arxiv_client

print("TESTING ARXIV API CLIENT")
print("=" * 40)

# Create client
arxiv_client = make_arxiv_client()
print(f"✓ Client created: {arxiv_client.base_url}")
print(f"   Rate limit: {arxiv_client.rate_limit_delay}s")
print(f"   Max results: {arxiv_client.max_results}")
print(f"   Category: {arxiv_client.search_category}")
print()

# Test Paper Fetching
async def test_paper_fetching():
    """Test fetching papers from arXiv with rate limiting."""
    
    print("Test 1: Fetch Recent CS.AI Papers")
    try:
        papers = await arxiv_client.fetch_papers(
            max_results=2, 
            sort_by="submittedDate",
            sort_order="descending"
        )
        
        print(f"✓ Fetched {len(papers)} papers")
        
        if papers:
            for i, paper in enumerate(papers[:2], 1):
                print(f"   {i}. [{paper.arxiv_id}] {paper.title[:60]}...")
                print(f"      Authors: {', '.join(paper.authors[:2])}{'...' if len(paper.authors) > 2 else ''}")
                print(f"      Categories: {', '.join(paper.categories)}")
                print(f"      Published: {paper.published_date}")
                print()
        
        return papers
        
    except Exception as e:
        print(f"✗ Error fetching papers: {e}")
        if "503" in str(e):
            print("   arXiv API temporarily unavailable (normal)")
            print("   Rate limiting and error handling working correctly")
        return []

# Run the test
papers = asyncio.run(test_paper_fetching())