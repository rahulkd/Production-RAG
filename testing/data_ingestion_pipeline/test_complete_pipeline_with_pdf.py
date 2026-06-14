# Test Complete Pipeline with PDF Processing
import asyncio

from src.db.factory import make_database
from src.services.arxiv.factory import make_arxiv_client
from src.services.metadata_fetcher import make_metadata_fetcher
from src.services.pdf_parser.factory import make_pdf_parser_service

print("Test 8: Complete Pipeline with PDF Processing")
print("=" * 50)

# Create dependencies
arxiv_client = make_arxiv_client()
pdf_parser = make_pdf_parser_service()
database = make_database()
metadata_fetcher = make_metadata_fetcher(arxiv_client, pdf_parser)
print("✓ Metadata fetcher service created")


async def main():
    # Test with small batch including PDF processing
    print("Running enhanced test (3 papers with PDF processing)...")

    try:
        with database.get_session() as session:
            results = await metadata_fetcher.fetch_and_process_papers(
                max_results=3,  # Small batch
                from_date="20250813",  # Recent date
                to_date="20250814",
                process_pdfs=True,
                store_to_db=True,
                db_session=session
            )

        print("\nENHANCED PIPELINE RESULTS:")
        print(f"   Papers fetched: {results.get('papers_fetched', 0)}")
        print(f"   PDFs downloaded: {results.get('pdfs_downloaded', 0)}")
        print(f"   PDFs parsed: {results.get('pdfs_parsed', 0)}")
        print(f"   Papers stored: {results.get('papers_stored', 0)}")
        print(f"   Processing time: {results.get('processing_time', 0):.1f}s")
        print(f"   Errors: {len(results.get('errors', []))}")

        # Show success rates
        if results.get('papers_fetched', 0) > 0:
            download_rate = (results['pdfs_downloaded'] / results['papers_fetched']) * 100
            parse_rate = (results['pdfs_parsed'] / results['pdfs_downloaded']) * 100 if results.get('pdfs_downloaded', 0) > 0 else 0
            print(f"   Download success rate: {download_rate:.1f}%")
            print(f"   Parse success rate: {parse_rate:.1f}%")

        if results.get('errors'):
            print("\nErrors encountered (showing graceful error handling):")
            for error in results.get('errors', [])[:3]:  # Show first 3 errors
                print(f"   - {error}")

        if results.get('papers_fetched', 0) > 0:
            print("\n✓ Enhanced pipeline test successful!")
            if results.get('errors'):
                print("✓ System continued processing despite PDF failures")
        else:
            print("\n! No papers fetched - may be arXiv API unavailability")

    except Exception as e:
        print(f"✗ Pipeline error: {e}")


asyncio.run(main())