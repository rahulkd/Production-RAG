from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ArxivPaper(BaseModel):
    """Raw paper shape as returned directly from the arXiv API.

    Used immediately after XML parsing in client.py. published_date is kept
    as a plain string (ISO format) because no datetime conversion is applied
    at the API boundary — that happens downstream in PaperBase.
    """

    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of author names")
    abstract: str = Field(..., description="Paper abstract")
    categories: List[str] = Field(..., description="Paper categories")
    published_date: str = Field(..., description="Date published on arXiv (ISO format)")
    pdf_url: str = Field(..., description="URL to PDF")


class PaperBase(BaseModel):
    """Shared core metadata fields inherited by PaperCreate and PaperResponse.

    Acts as the single source of truth for required paper fields. Any change
    here propagates to both subclasses. published_date is a proper datetime
    object here (unlike ArxivPaper which keeps it as a raw string).
    """

    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of author names")
    abstract: str = Field(..., description="Paper abstract")
    categories: List[str] = Field(..., description="Paper categories")
    published_date: datetime = Field(..., description="Date published on arXiv")
    pdf_url: str = Field(..., description="URL to PDF")


class PaperCreate(PaperBase):
    """Schema for inserting a paper into the database.

    Extends PaperBase with optional PDF content and processing metadata.
    All extra fields are Optional because a paper is saved with just core
    metadata first — PDF content is attached later when the parser runs
    (progressive enrichment pattern).
    """

    # Parsed PDF content (populated after PDF processing)
    raw_text: Optional[str] = Field(None, description="Full raw text extracted from PDF")
    sections: Optional[List[Dict[str, Any]]] = Field(None, description="List of sections with titles and content")
    references: Optional[List[Dict[str, Any]]] = Field(None, description="List of references if extracted")

    # PDF processing metadata
    parser_used: Optional[str] = Field(None, description="Which parser was used (DOCLING, GROBID, etc.)")
    parser_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional parser metadata")
    pdf_processed: Optional[bool] = Field(False, description="Whether PDF was successfully processed")
    pdf_processing_date: Optional[datetime] = Field(None, description="When PDF was processed")


class PaperResponse(PaperBase):
    """Schema for returning a paper record from the API.

    Represents a fully hydrated DB row. Adds three things PaperCreate lacks:
    - id (UUID): DB-assigned primary key, unique across all records even when
      papers share metadata (same title, authors, etc.)
    - created_at / updated_at: set by the DB, never by the caller
    from_attributes = True allows Pydantic to build this model directly from
    a SQLAlchemy ORM object without converting it to a dict first.
    """

    id: UUID

    # Parsed PDF content (optional — absent until PDF is processed)
    raw_text: Optional[str] = Field(None, description="Full raw text extracted from PDF")
    sections: Optional[List[Dict[str, Any]]] = Field(None, description="List of sections with titles and content")
    references: Optional[List[Dict[str, Any]]] = Field(None, description="List of references if extracted")

    # PDF processing metadata
    parser_used: Optional[str] = Field(None, description="Which parser was used")
    parser_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional parser metadata")
    pdf_processed: bool = Field(False, description="Whether PDF was successfully processed")
    pdf_processing_date: Optional[datetime] = Field(None, description="When PDF was processed")

    # Timestamps set by the database
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PaperSearchResponse(BaseModel):
    """Wrapper for paginated search/list endpoints.

    Returns a list of fully hydrated PaperResponse objects alongside the
    total count of matching records — total is used by the frontend/caller
    to implement pagination without issuing a separate COUNT query.
    """

    papers: List[PaperResponse]
    total: int
