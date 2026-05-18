from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ParserType(str, Enum):
    """Enum of supported PDF parsers.

    Inherits from str so values serialize to plain strings (e.g. "docling")
    rather than "ParserType.DOCLING" — safe for JSON responses and DB storage.
    GROBID is defined but not yet implemented; reserved for future use.
    """

    DOCLING = "docling"
    GROBID = "grobid"  # For future use


class PaperSection(BaseModel):
    """A single logical section extracted from a parsed PDF.

    Papers are structured documents — Introduction, Methods, Results, etc.
    level indicates the heading hierarchy: 1 = top-level heading,
    2 = sub-heading, and so on. Used to reconstruct document structure
    for chunking or display.
    """

    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    level: int = Field(default=1, description="Section hierarchy level")


class PaperFigure(BaseModel):
    """A figure reference extracted from a parsed PDF.

    Stores the caption and identifier of a figure. The actual image binary
    is not captured here — only the textual metadata, which is sufficient
    for indexing and search purposes.
    """

    caption: str = Field(..., description="Figure caption")
    id: str = Field(..., description="Figure identifier")


class PaperTable(BaseModel):
    """A table reference extracted from a parsed PDF.

    Mirrors PaperFigure in structure — captures caption and identifier.
    Table content (rows/cells) is not modelled here; the caption alone
    is used for search and metadata purposes.
    """

    caption: str = Field(..., description="Table caption")
    id: str = Field(..., description="Table identifier")


class PdfContent(BaseModel):
    """All structured content extracted from a PDF by a parser.

    This is the output contract for any parser (Docling, GROBID, etc.).
    Aggregates every artefact the parser produces: sections, figures, tables,
    raw text, references, and parser-specific metadata. parser_used records
    which parser produced this output so results can be compared or reproduced.
    All list fields default to empty lists — a parser may not extract every
    artefact type from every paper.
    """

    sections: List[PaperSection] = Field(default_factory=list, description="Paper sections")
    figures: List[PaperFigure] = Field(default_factory=list, description="Figures")
    tables: List[PaperTable] = Field(default_factory=list, description="Tables")
    raw_text: str = Field(..., description="Full extracted text")
    references: List[str] = Field(default_factory=list, description="References")
    parser_used: ParserType = Field(..., description="Parser used for extraction")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Parser metadata")


class ArxivMetadata(BaseModel):
    """Paper metadata sourced from the arXiv API (not the PDF).

    Mirrors the fields in ArxivPaper (schemas/arxiv/paper.py) but lives in
    the pdf_parser schema layer to keep parser models self-contained.
    published_date is a plain string here — no datetime parsing — consistent
    with ArxivPaper which also preserves the raw ISO string at the API boundary.
    """

    title: str = Field(..., description="Paper title from arXiv")
    authors: List[str] = Field(..., description="Authors from arXiv")
    abstract: str = Field(..., description="Abstract from arXiv")
    arxiv_id: str = Field(..., description="arXiv identifier")
    categories: List[str] = Field(default_factory=list, description="arXiv categories")
    published_date: str = Field(..., description="Publication date")
    pdf_url: str = Field(..., description="PDF download URL")


class ParsedPaper(BaseModel):
    """Top-level output of the PDF parsing pipeline.

    Combines arXiv metadata (always present) with PDF content (optional).
    pdf_content is None when the PDF has not been downloaded or parsed yet —
    this mirrors the progressive enrichment pattern used in PaperCreate,
    where a paper record can exist with just metadata before PDF processing runs.
    """

    arxiv_metadata: ArxivMetadata = Field(..., description="Metadata from arXiv API")
    pdf_content: Optional[PdfContent] = Field(None, description="Content extracted from PDF")
