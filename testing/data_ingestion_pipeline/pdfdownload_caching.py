# Test PDF Download
import asyncio

from src.services.arxiv.factory import make_arxiv_client

arxiv_client = make_arxiv_client()


async def test_pdf_download(test_papers):
    """Test PDF downloading with caching."""

    print("Test 3: PDF Download & Caching")
    
    if not test_papers:
        print("No papers available for PDF download test")
        return None
    
    # Test with first paper
    test_paper = test_papers[0]
    print(f"Testing PDF download for: {test_paper.arxiv_id}")
    print(f"Title: {test_paper.title[:60]}...")
    
    try:
        # Download PDF 
        pdf_path = await arxiv_client.download_pdf(test_paper)
        
        if pdf_path and pdf_path.exists():
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"✓ PDF downloaded: {pdf_path.name} ({size_mb:.2f} MB)")
            
            return pdf_path
        else:
            print("✗ PDF download failed")
            return None
            
    except Exception as e:
        print(f"✗ PDF download error: {e}")
        return None

# Run PDF download test
async def main():
    # Fetch a recent paper to download
    papers = await arxiv_client.fetch_papers(
        max_results=1,
        sort_by="submittedDate",
        sort_order="descending",
    )
    return await test_pdf_download(papers[:1])


pdf_path = asyncio.run(main())