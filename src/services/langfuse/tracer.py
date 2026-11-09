"""Simple, efficient Langfuse tracing utility for RAG pipeline."""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from src.config import get_settings

from .client import LangfuseTracer


def bedrock_usage(
    usage: Optional[Dict[str, Any]], input_price_per_1m: float, output_price_per_1m: float
) -> Optional[Dict[str, Any]]:
    """Convert Bedrock Converse usage into Langfuse usage with cost.

    :param usage: Converse ``usage`` (``inputTokens``, ``outputTokens``, ``totalTokens``).
    :param input_price_per_1m: USD per 1M input tokens of the model that was called.
    :param output_price_per_1m: USD per 1M output tokens.
    :returns: Langfuse usage dict, or None when Bedrock reported no usage.
    """
    if not usage:
        return None

    input_tokens = int(usage.get("inputTokens", 0))
    output_tokens = int(usage.get("outputTokens", 0))
    input_cost = input_tokens * input_price_per_1m / 1_000_000
    output_cost = output_tokens * output_price_per_1m / 1_000_000
    return {
        "unit": "TOKENS",
        "input": input_tokens,
        "output": output_tokens,
        "total": int(usage.get("totalTokens", input_tokens + output_tokens)),
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


class RAGTracer:
    """Clean, purpose-built tracer for RAG operations."""

    def __init__(self, tracer: LangfuseTracer, endpoint: str = "ask"):
        self.tracer = tracer
        # "ask" or "stream"; tagged on the trace so the two can be filtered apart
        self.endpoint = endpoint
        self.prices = get_settings().bedrock

    @staticmethod
    def trace_id(trace) -> Optional[str]:
        """Id of a trace, for feedback and background scoring (None when tracing is off)."""
        return getattr(trace, "trace_id", None) if trace else None

    def mark_cache(self, trace, hit: bool):
        """Tag the trace with the endpoint and whether the answer came from the Redis cache."""
        self.tracer.update_trace(
            trace,
            tags=[self.endpoint, "cache_hit" if hit else "cache_miss"],
            metadata={"simplified_tracing": True, "endpoint": self.endpoint, "cache_hit": hit},
        )

    def score_retrieval(self, trace, chunks: List[Dict], use_hybrid: bool, search_mode: str):
        """Heuristic quality scores, computed from this request alone (no LLM calls).

        * ``retrieval_hit``   — retrieval returned at least one chunk
        * ``unique_papers``   — distinct papers among the retrieved chunks
        * ``hybrid_fallback`` — hybrid was requested but embeddings failed, so BM25 ran
        """
        trace_id = self.trace_id(trace)
        if not trace_id:
            return

        self.tracer.score_by_id(trace_id, "retrieval_hit", 1 if chunks else 0, data_type="BOOLEAN")
        if chunks:
            papers = {chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")}
            self.tracer.score_by_id(trace_id, "unique_papers", len(papers), data_type="NUMERIC")
        if use_hybrid:
            self.tracer.score_by_id(trace_id, "hybrid_fallback", 1 if search_mode == "bm25" else 0, data_type="BOOLEAN")

    @contextmanager
    def trace_request(self, user_id: str, query: str):
        """Main request trace context manager."""
        with self.tracer.trace_rag_request(
            query=query, user_id=user_id, session_id=f"session_{user_id}", metadata={"simplified_tracing": True}
        ) as trace:
            try:
                yield trace
            finally:
                if trace:
                    self.tracer.flush()

    @contextmanager
    def trace_embedding(self, trace, query: str):
        """Query embedding operation with timing."""
        start_time = time.time()
        span = self.tracer.create_span(
            trace=trace, name="query_embedding", input_data={"query": query, "query_length": len(query)}
        )
        try:
            yield span
        finally:
            duration = time.time() - start_time
            if span:
                self.tracer.update_span(span=span, output={"embedding_duration_ms": round(duration * 1000, 2), "success": True})
                span.end()

    @contextmanager
    def trace_search(self, trace, query: str, top_k: int):
        """Search operation with timing."""
        span = self.tracer.create_span(trace=trace, name="search_retrieval", input_data={"query": query, "top_k": top_k})
        try:
            yield span
        finally:
            if span:
                span.end()

    def end_search(self, span, chunks: List[Dict], arxiv_ids: List[str], total_hits: int):
        """End search span with essential results."""
        if not span:
            return

        self.tracer.update_span(
            span=span,
            output={
                "chunks_returned": len(chunks),
                "unique_papers": len(set(arxiv_ids)),
                "total_hits": total_hits,
                "arxiv_ids": list(set(arxiv_ids)),
            },
        )

    @contextmanager
    def trace_prompt_construction(self, trace, chunks: List[Dict]):
        """Prompt building with timing."""
        span = self.tracer.create_span(trace=trace, name="prompt_construction", input_data={"chunk_count": len(chunks)})
        try:
            yield span
        finally:
            if span:
                span.end()

    def end_prompt(self, span, prompt: str):
        """End prompt span with final prompt."""
        if not span:
            return

        self.tracer.update_span(
            span=span,
            output={
                "prompt_length": len(prompt),
                # Don't duplicate the full prompt here since it's in llm_generation input
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            },
        )

    @contextmanager
    def trace_generation(self, trace, model: str, prompt: str):
        """LLM call with timing, recorded as a Langfuse generation so token usage and cost show up."""
        generation = self.tracer.create_generation(
            trace=trace,
            name="llm_generation",
            model=model,
            input_data={"prompt": prompt},
            metadata={"prompt_length": len(prompt), "endpoint": self.endpoint},
        )
        try:
            yield generation
        finally:
            if generation:
                generation.end()

    def end_generation(self, generation, response: str, model: str, usage: Optional[Dict[str, Any]] = None):
        """Record the generation's response, token usage and cost.

        :param usage: Bedrock Converse ``usage`` from the LLM client.
        """
        if not generation:
            return

        self.tracer.update_generation(
            generation,
            output={"response": response, "response_length": len(response), "model_used": model},
            usage=bedrock_usage(usage, self.prices.input_price_per_1m_tokens, self.prices.output_price_per_1m_tokens),
        )

    def end_request(self, trace, response: str, total_duration: float):
        """End main request trace."""
        if not trace:
            return

        try:
            trace.update(
                output={"answer": response, "total_duration_seconds": round(total_duration, 3), "response_length": len(response)}
            )
        except Exception:
            # Silently fail - don't break the request for tracing issues
            pass
