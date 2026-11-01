from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request model for RAG question answering."""

    query: str = Field(..., description="User's question", min_length=1, max_length=1000)
    top_k: int = Field(10, description="Number of top chunks to retrieve", ge=1, le=10)
    use_hybrid: bool = Field(True, description="Use hybrid search (BM25 + vector)")
    #model: str = Field("llama3.2:1b", description="Ollama model to use for generation")  ## ollama
    model: str = Field("us.meta.llama3-3-70b-instruct-v1:0", description="Bedrock model id to use for generation")
    categories: Optional[List[str]] = Field(None, description="Filter by arXiv categories")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "top_k": 10,
                "use_hybrid": True,
                "model": "us.meta.llama3-3-70b-instruct-v1:0",
                "categories": ["cs.AI", "cs.LG"],
            }
        }


class AskResponse(BaseModel):
    """Response model for RAG question answering."""

    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer from LLM")
    sources: List[str] = Field(..., description="PDF URLs of source papers")
    chunks_used: int = Field(..., description="Number of chunks used for generation")
    search_mode: str = Field(..., description="Search mode used: bm25 or hybrid")
    trace_id: Optional[str] = Field(None, description="Langfuse trace id, for submitting feedback; None when tracing is off")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "answer": "Transformers are a neural network architecture...",
                "sources": ["https://arxiv.org/pdf/1706.03762.pdf", "https://arxiv.org/pdf/1810.04805.pdf"],
                "chunks_used": 3,
                "search_mode": "hybrid",
            }
        }


class FeedbackRequest(BaseModel):
    """User feedback on one answer, recorded as a score on its Langfuse trace."""

    trace_id: str = Field(..., description="trace_id from the AskResponse or the stream's metadata frame", min_length=1)
    value: int = Field(..., description="1 = helpful (thumbs up), 0 = not helpful (thumbs down)", ge=0, le=1)
    comment: Optional[str] = Field(None, description="Optional free-text comment", max_length=1000)
