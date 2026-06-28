# Read and display the contents of the PostgreSQL `papers` table.
from src.db.factory import make_database
from src.repositories.paper import PaperRepository


def read_papers(limit: int = 100, offset: int = 0) -> None:
    """Read papers from the PostgreSQL database and print a summary of each.

    Args:
        limit: Maximum number of papers to read.
        offset: Number of papers to skip (for pagination).
    """
    database = make_database()

    with database.get_session() as session:
        paper_repo = PaperRepository(session)
        papers = paper_repo.get_all(limit=limit, offset=offset)

        print(f"Found {len(papers)} paper(s) in the database")
        print("=" * 60)

        for i, paper in enumerate(papers, 1):
            authors = ", ".join(paper.authors) if isinstance(paper.authors, list) else str(paper.authors)
            categories = ", ".join(paper.categories) if isinstance(paper.categories, list) else str(paper.categories)
            raw_text_len = len(paper.raw_text) if paper.raw_text else 0

            print(f"\n[{i}] {paper.arxiv_id} - {paper.title}")
            print(f"    Authors:        {authors}")
            print(f"    Categories:     {categories}")
            print(f"    Published:      {paper.published_date}")
            print(f"    PDF URL:        {paper.pdf_url}")
            print(f"    PDF processed:  {paper.pdf_processed} ({raw_text_len} chars of raw text)")
            print(f"    Created at:     {paper.created_at}")

    database.teardown()


if __name__ == "__main__":
    read_papers()
