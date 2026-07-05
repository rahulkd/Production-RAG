# Test PDF Parsing with Docling
import asyncio
import sys
from pathlib import Path

from src.config import get_settings
from src.schemas.pdf_parser.models import PdfContent
from src.services.pdf_parser.factory import make_pdf_parser_service

# Anchor input/output dirs to this script so they work regardless of cwd
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

print("Test 4: PDF Parsing with Docling")
print("=" * 40)

# Create PDF parser
pdf_parser = make_pdf_parser_service()
settings = get_settings()
print("PDF parser service created")
print(f"Config: {settings.pdf_parser.max_pages} pages, {settings.pdf_parser.max_file_size_mb}MB")


def save_sections_to_file(pdf_content: PdfContent, source_name: str) -> Path:
    """Dump all parsed sections to a single text file under results/ for inspection.

    Args:
        pdf_content: Parsed content returned by the PDF parser.
        source_name: Name of the source PDF (used for the output filename).

    Returns:
        Path to the written text file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{Path(source_name).stem}_sections.txt"

    lines: list[str] = []
    lines.append(f"Source PDF: {source_name}")
    lines.append(f"Parser used: {pdf_content.parser_used}")
    lines.append(f"Total sections: {len(pdf_content.sections)}")
    lines.append(f"Raw text length: {len(pdf_content.raw_text)} characters")
    lines.append("=" * 80)

    for i, section in enumerate(pdf_content.sections, 1):
        lines.append("")
        lines.append(f"[Section {i}] (level {section.level}) {section.title}")
        lines.append(f"({len(section.content)} chars)")
        lines.append("-" * 80)
        lines.append(section.content)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def save_raw_text_to_file(pdf_content: PdfContent, source_name: str) -> Path:
    """Dump the full raw text of a parsed PDF to a text file under results/.

    Args:
        pdf_content: Parsed content returned by the PDF parser.
        source_name: Name of the source PDF (used for the output filename).

    Returns:
        Path to the written text file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{Path(source_name).stem}_raw_text.txt"
    out_path.write_text(pdf_content.raw_text, encoding="utf-8")
    return out_path


async def main():
    # Allow parsing a single, user-supplied PDF: `python pdfparser_docling.py /path/to/file.pdf`
    if len(sys.argv) > 1:
        test_pdf = Path(sys.argv[1]).expanduser().resolve()
        if not test_pdf.is_file():
            print(f"No PDF file found at {test_pdf}")
            return
    else:
        # Test parsing with actual PDF files from the cache directory
        cache_dir = SCRIPT_DIR / "data/arxiv_pdfs"
        if not cache_dir.exists():
            print(f"No PDF cache directory found at {cache_dir}")
            return

        pdf_files = list(cache_dir.glob("*.pdf"))
        print(f"\nFound {len(pdf_files)} PDF files to test parsing")

        if not pdf_files:
            print("No PDF files available for parsing test")
            return

        # Test parsing the first PDF
        test_pdf = pdf_files[0]

    print(f"Testing PDF parsing with: {test_pdf.name}")

    try:
        pdf_content = await pdf_parser.parse_pdf(test_pdf)

        if pdf_content:
            print("✓ PDF parsing successful!")
            print(f"  Sections: {len(pdf_content.sections)}")
            print(f"  Raw text length: {len(pdf_content.raw_text)} characters")
            print(f"  Parser used: {pdf_content.parser_used}")

            # Show first section as example
            if pdf_content.sections:
                first_section = pdf_content.sections[0]
                print(f"  First section: '{first_section.title}' ({len(first_section.content)} chars)")

            # Save all parsed sections to a single text file for inspection
            out_path = save_sections_to_file(pdf_content, test_pdf.name)
            print(f"  Saved parsed sections to: {out_path}")

            # Save the full raw text to its own text file
            raw_text_path = save_raw_text_to_file(pdf_content, test_pdf.name)
            print(f"  Saved raw text to: {raw_text_path}")
        else:
            print("✗ PDF parsing failed (Docling compatibility issue)")
            print("This is expected - not all PDFs work with Docling")

    except Exception as e:
        print(f"✗ PDF parsing error: {e}")
        print("This demonstrates the error handling in action")


asyncio.run(main())