# Test Complete Pipeline
import asyncio

from src.db.factory import make_database
from src.services.arxiv.factory import make_arxiv_client
from src.services.metadata_fetcher import make_metadata_fetcher
from src.services.pdf_parser.factory import make_pdf_parser_service

print("Test 6: Complete Metadata Fetcher Pipeline")
print("=" * 50)

# Create dependencies
arxiv_client = make_arxiv_client()
pdf_parser = make_pdf_parser_service()
database = make_database()

# Create metadata fetcher
metadata_fetcher = make_metadata_fetcher(arxiv_client, pdf_parser)
print("✓ Metadata fetcher service created")


async def main():
    # Test with small batch
    print("Running small batch test (2 papers, no PDF processing for speed)...")

    try:
        with database.get_session() as session:
            results = await metadata_fetcher.fetch_and_process_papers(
                max_results=2,
                process_pdfs=False,
                store_to_db=True,
                db_session=session
            )

        print("\nPIPELINE RESULTS:")
        print(f"   Papers fetched: {results.get('papers_fetched', 0)}")
        print(f"   PDFs downloaded: {results.get('pdfs_downloaded', 0)}")
        print(f"   PDFs parsed: {results.get('pdfs_parsed', 0)}")
        print(f"   Papers stored: {results.get('papers_stored', 0)}")
        print(f"   Processing time: {results.get('processing_time', 0):.1f}s")
        print(f"   Errors: {len(results.get('errors', []))}")

        if results.get('errors'):
            print("\nErrors encountered:")
            for error in results.get('errors', [])[:3]:  # Show first 3 errors
                print(f"   - {error}")

        if results.get('papers_fetched', 0) > 0:
            print("\n✓ Pipeline test successful!")
        else:
            print("\nNo papers fetched - may be arXiv API unavailability")

    except Exception as e:
        print(f"✗ Pipeline error: {e}")


asyncio.run(main())