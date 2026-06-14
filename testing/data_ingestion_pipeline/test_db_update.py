# Test Database Storage
import asyncio

from dateutil import parser as date_parser

from src.db.factory import make_database
from src.repositories.paper import PaperRepository
from src.schemas.arxiv.paper import PaperCreate
from src.services.arxiv.factory import make_arxiv_client

print("Test 5: Database Storage")
print("=" * 40)

# Create database connection
database = make_database()
print("✓ Database connection created")


async def main():
    # Fetch a recent paper to store
    arxiv_client = make_arxiv_client()
    papers = await arxiv_client.fetch_papers(
        max_results=1,
        sort_by="submittedDate",
        sort_order="descending",
    )

    if not papers:
        print("No papers available for database storage test")
        return

    test_paper = papers[0]
    print(f"Storing paper: {test_paper.arxiv_id}")

    try:
        with database.get_session() as session:
            paper_repo = PaperRepository(session)
            
            # Convert to database format
            published_date = date_parser.parse(test_paper.published_date) if isinstance(test_paper.published_date, str) else test_paper.published_date
            
            paper_create = PaperCreate(
                arxiv_id=test_paper.arxiv_id,
                title=test_paper.title,
                authors=test_paper.authors,
                abstract=test_paper.abstract,
                categories=test_paper.categories,
                published_date=published_date,
                pdf_url=test_paper.pdf_url
            )
            
            # Store paper (upsert to avoid duplicates)
            stored_paper = paper_repo.upsert(paper_create)
            
            if stored_paper:
                print(f"✓ Paper stored with ID: {stored_paper.id}")
                print(f"   Database ID: {stored_paper.id}")
                print(f"   arXiv ID: {stored_paper.arxiv_id}")
                print(f"   Title: {stored_paper.title[:50]}...")
                print(f"   Authors: {len(stored_paper.authors)} authors")
                print(f"   Categories: {', '.join(stored_paper.categories)}")
                
                # Test retrieval
                retrieved_paper = paper_repo.get_by_arxiv_id(test_paper.arxiv_id)
                if retrieved_paper:
                    print(f"✓ Paper retrieval test passed")
                else:
                    print(f"✗ Paper retrieval failed")
            else:
                print("✗ Paper storage failed")
                
    except Exception as e:
        print(f"✗ Database error: {e}")


asyncio.run(main())