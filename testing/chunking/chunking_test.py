# Get Sample Papers from Database
import textwrap
from pathlib import Path

from src.db.factory import make_database
from src.models.paper import Paper

# Anchor output to this script so it works regardless of cwd
SCRIPT_DIR = Path(__file__).resolve().parent

print("FETCHING PAPERS")
print("=" * 50)

database = make_database()

with database.get_session() as session:
    # Get every paper that has processed text (with or without sections).
    # raw_text is required to chunk at all; papers without sections exercise
    # the word-based fallback path.
    papers = session.query(Paper).filter(
        Paper.raw_text != None,
        Paper.raw_text != ""
    ).all()

    if papers:
        print(f"Found {len(papers)} papers with processed text:\n")
        sample_papers = []
        
        for i, paper in enumerate(papers, 1):
            print(f"{i}. [{paper.arxiv_id}] {paper.title[:60]}...")
            print(f"   Text length: {len(paper.raw_text):,} characters")
            print(f"   Sections available: {'Yes' if paper.sections else 'No'}\n")
            
            sample_papers.append({
                'arxiv_id': paper.arxiv_id,
                'title': paper.title,
                'abstract': paper.abstract,
                'raw_text': paper.raw_text,
                'sections': paper.sections,
                'authors': paper.authors,
                'categories': paper.categories,
                'published_date': paper.published_date
            })
        
        test_paper = sample_papers[0]
        print(f"Selected paper for analysis: {test_paper['arxiv_id']}")
        
    else:
        print("No papers with processed text found.")
        print("Please run the Airflow DAG 'arxiv_paper_ingestion' first.")
        test_paper = None
        sample_papers = []


## ------------------------------------------------------------------------------------------------ ##
##  TEST THE TEXT CHUNKER ALGORITHM
## ------------------------------------------------------------------------------------------------ ##

from src.config import get_settings
from src.services.indexing.text_chunker import TextChunker

# Tally of checks so we can print a summary at the end
_results = {"passed": 0, "failed": 0}


def check(name: str, condition: bool, detail: str = "") -> bool:
    """Record and print the outcome of a single assertion."""
    status = "PASS" if condition else "FAIL"
    if condition:
        _results["passed"] += 1
    else:
        _results["failed"] += 1
    suffix = f" - {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return condition


# Build the chunker with the same configuration used in production
settings = get_settings()
chunker = TextChunker(
    chunk_size=settings.chunking.chunk_size,
    overlap_size=settings.chunking.overlap_size,
    min_chunk_size=settings.chunking.min_chunk_size,
)

print("\nCHUNKER CONFIGURATION")
print("=" * 50)
print(f"  chunk_size     = {chunker.chunk_size}")
print(f"  overlap_size   = {chunker.overlap_size}")
print(f"  min_chunk_size = {chunker.min_chunk_size}")


# --- Output formatting (ported from data_ingestion_pipeline/results/format_text.py) ---
# Rewraps prose to a fixed column width so long single-line paragraphs (title +
# abstract + section text joined into one chunk) read like they do in the PDF,
# while leaving structural marker lines untouched.
WRAP_WIDTH = 100

# Structural lines kept verbatim: chunk headers, metadata lines, and the file header.
_KEEP_VERBATIM = ("[Chunk ", "words=", "Paper:", "Source length:", "Sections available:", "Total chunks:")


def _is_separator(line: str) -> bool:
    """A ruler line made entirely of '=' or '-'."""
    stripped = line.strip()
    return len(stripped) >= 3 and set(stripped) <= {"=", "-"}


def _format_lines(lines: list[str], width: int = WRAP_WIDTH) -> list[str]:
    """Rewrap prose lines to `width` while leaving structural lines untouched."""
    out: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # Preserve blank lines, separators, and marker lines exactly.
        if not stripped:
            out.append("")
            continue
        if _is_separator(line):
            # Normalise ruler width so everything lines up horizontally.
            out.append(stripped[0] * width)
            continue
        if stripped.startswith(_KEEP_VERBATIM):
            out.append(line)
            continue

        # Word-wrap prose. break_long_words=False keeps tokens like URLs and
        # math notation intact instead of chopping them mid-token.
        wrapped = textwrap.fill(stripped, width=width, break_long_words=False, break_on_hyphens=False)
        out.extend(wrapped.split("\n"))

    return out


def save_chunks_to_file(chunks: list, paper: dict) -> Path:
    """Write every extracted chunk (text + metadata) to results/<arxiv_id>.txt, pre-formatted."""
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{paper['arxiv_id']}.txt"

    lines: list[str] = []
    lines.append(f"Paper: [{paper['arxiv_id']}] {paper['title']}")
    lines.append(f"Source length: {len(paper['raw_text']):,} characters")
    lines.append(f"Sections available: {'Yes' if paper['sections'] else 'No'}")
    lines.append(f"Total chunks: {len(chunks)}")
    lines.append("=" * 80)

    for c in chunks:
        m = c.metadata
        lines.append("")
        lines.append(f"[Chunk {m.chunk_index}] section='{m.section_title or '(word-based)'}'")
        lines.append(
            f"words={m.word_count} | chars={m.start_char}-{m.end_char} | "
            f"overlap prev={m.overlap_with_previous} next={m.overlap_with_next}"
        )
        lines.append("-" * 80)
        lines.append(c.text)

    # Expand embedded newlines, then word-wrap prose to a fixed width before writing.
    raw_lines = "\n".join(lines).splitlines()
    formatted = _format_lines(raw_lines)
    out_path.write_text("\n".join(formatted) + "\n", encoding="utf-8")
    return out_path


def test_chunk_paper(paper: dict) -> None:
    """Run chunk_paper() on a real fetched paper and validate the output."""
    has_sections = "with sections" if paper["sections"] else "no sections"
    print(f"\nTEST: chunk_paper() on [{paper['arxiv_id']}] ({has_sections})")
    print("=" * 50)

    chunks = chunker.chunk_paper(
        title=paper["title"],
        abstract=paper["abstract"],
        full_text=paper["raw_text"],
        arxiv_id=paper["arxiv_id"],
        # sample_papers has no DB id; arxiv_id is a fine stand-in for the test
        paper_id=paper.get("paper_id", paper["arxiv_id"]),
        sections=paper["sections"],
    )

    print(f"  Produced {len(chunks)} chunks from {len(paper['raw_text']):,} chars\n")

    check("returns at least one chunk", len(chunks) > 0)
    if not chunks:
        return

    # chunk_index must be sequential starting at 0
    indices = [c.metadata.chunk_index for c in chunks]
    check("chunk_index is sequential from 0", indices == list(range(len(chunks))), detail=f"{indices[:5]}...")

    # Every chunk carries non-empty text and correct identifiers
    check("all chunks have non-empty text", all(c.text.strip() for c in chunks))
    check("all chunks carry the arxiv_id", all(c.arxiv_id == paper["arxiv_id"] for c in chunks))

    # word_count metadata should match the actual text
    wc_ok = all(c.metadata.word_count == len(c.text.split()) for c in chunks)
    check("word_count metadata matches text", wc_ok)

    # When sections drive chunking, the header (title) should be embedded in each chunk
    if paper["sections"]:
        header_ok = all(paper["title"][:30] in c.text for c in chunks)
        check("section chunks include the title header", header_ok)

    # Report the size distribution for a human sanity check
    sizes = sorted(c.metadata.word_count for c in chunks)
    print(f"\n  word_count -> min={sizes[0]}, max={sizes[-1]}, median={sizes[len(sizes) // 2]}")
    for i, c in enumerate(chunks[:3]):
        title = c.metadata.section_title or "(word-based)"
        print(f"    chunk {i}: {c.metadata.word_count} words | section='{title}'")

    # Persist the full set of chunks for inspection
    out_path = save_chunks_to_file(chunks, paper)
    print(f"\n  Saved {len(chunks)} chunks to: {out_path}")


def test_word_based_chunking() -> None:
    """Validate the sliding-window mechanics of chunk_text() with synthetic text."""
    print("\nTEST: chunk_text() sliding window")
    print("=" * 50)

    # 1600 distinct words -> forces multiple windows (stride = 600 - 100 = 500)
    words = [f"w{i}" for i in range(1600)]
    text = " ".join(words)
    chunks = chunker.chunk_text(text, arxiv_id="synthetic", paper_id="synthetic")

    # Expected windows: [0:600], [500:1100], [1000:1600] -> 3 chunks
    check("expected number of windows", len(chunks) == 3, detail=f"got {len(chunks)}")
    check("first chunk holds chunk_size words", chunks[0].metadata.word_count == chunker.chunk_size)

    # Consecutive chunks must share exactly overlap_size words
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    shared = set(first_words[-chunker.overlap_size:]) & set(second_words[: chunker.overlap_size])
    check("neighbours overlap by overlap_size words", len(shared) == chunker.overlap_size, detail=f"{len(shared)} shared")

    # No words should be dropped: last window must reach the final word
    check("last chunk reaches the final word", "w1599" in chunks[-1].text)


def test_edge_cases() -> None:
    """Validate behaviour on empty and undersized inputs."""
    print("\nTEST: edge cases")
    print("=" * 50)

    check("empty string yields no chunks", chunker.chunk_text("", "e", "e") == [])
    check("whitespace-only yields no chunks", chunker.chunk_text("   \n  ", "e", "e") == [])

    # Text shorter than min_chunk_size takes the single-chunk branch.
    # NOTE: this currently raises TypeError (chunk_text calls _reconstruct_text
    # with two args at text_chunker.py:114) - caught here so the suite still finishes.
    short_text = " ".join(f"w{i}" for i in range(30))
    try:
        small = chunker.chunk_text(short_text, "s", "s")
        check("short text yields exactly one chunk", len(small) == 1, detail=f"got {len(small)}")
    except TypeError as exc:
        check("short text single-chunk path (known bug)", False, detail=f"TypeError: {exc}")


# Run the suite
test_word_based_chunking()
test_edge_cases()

# Run the real-paper test against every paper in the database (with or without sections)
if sample_papers:
    with_sections = sum(1 for p in sample_papers if p["sections"])
    print(f"\nRunning chunker over {len(sample_papers)} papers "
          f"({with_sections} with sections, {len(sample_papers) - with_sections} without)")
    for paper in sample_papers:
        test_chunk_paper(paper)
else:
    print("\nSkipping real-paper tests: no papers fetched from the database.")

print("\nSUMMARY")
print("=" * 50)
print(f"  Papers tested: {len(sample_papers)}")
print(f"  Passed: {_results['passed']}")
print(f"  Failed: {_results['failed']}")
