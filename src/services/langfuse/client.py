import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

from langfuse import Langfuse
from src.config import Settings

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Wrapper for Langfuse tracing client."""

    def __init__(self, settings: Settings):
        self.settings = settings.langfuse
        self.client: Optional[Langfuse] = None

        if self.settings.enabled and self.settings.public_key and self.settings.secret_key:
            try:
                self.client = Langfuse(
                    public_key=self.settings.public_key,
                    secret_key=self.settings.secret_key,
                    host=self.settings.host,
                    flush_at=self.settings.flush_at,
                    flush_interval=self.settings.flush_interval,
                    debug=self.settings.debug,
                )
                logger.info(f"Langfuse tracing initialized (host: {self.settings.host})")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                self.client = None
        else:
            logger.info("Langfuse tracing disabled or missing credentials")

    @contextmanager
    def trace_rag_request(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracing a RAG request.

        Args:
            query: The user's query
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Additional metadata to attach to the trace

        Yields:
            Trace object if Langfuse is enabled, None otherwise
        """
        if not self.client:
            yield None
            return

        try:
            # Create a trace using v2 API
            trace = self.client.trace(
                name="rag_request",
                input={"query": query},
                metadata=metadata or {},
                user_id=user_id,
                session_id=session_id,
            )
            yield trace
        except Exception as e:
            logger.error(f"Error creating Langfuse trace: {e}")
            yield None

    def create_trace(
        self,
        name: str,
        trace_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
    ):
        """
        Create a standalone trace (outside a request), e.g. for background judging.

        Args:
            name: Trace name
            trace_id: Optional explicit id
            input_data: Trace input
            metadata: Trace metadata
            tags: Trace tags

        Returns:
            Trace object if successful, None otherwise
        """
        if not self.client:
            return None

        try:
            return self.client.trace(id=trace_id, name=name, input=input_data, metadata=metadata or {}, tags=tags)
        except Exception as e:
            logger.error(f"Error creating trace {name}: {e}")
            return None

    def create_span(
        self,
        trace,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a span within a trace.

        Args:
            trace: Parent trace object
            name: Name of the span
            input_data: Input data for the span
            metadata: Additional metadata

        Returns:
            Span object if successful, None otherwise
        """
        if not trace or not self.client:
            return None

        try:
            # Create a span using v2 API
            return self.client.span(
                trace_id=trace.trace_id,
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.error(f"Error creating span {name}: {e}")
            return None

    def create_generation(
        self,
        trace,
        name: str,
        model: str,
        input_data: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a generation (LLM call) within a trace.

        Args:
            trace: Parent trace object
            name: Name of the generation
            model: Model name
            input_data: Input/prompt data
            output: Generated output
            metadata: Additional metadata
            usage: Token usage information

        Returns:
            Generation object if successful, None otherwise
        """
        if not trace or not self.client:
            return None

        try:
            # Create a generation using v2 API
            return self.client.generation(
                trace_id=trace.trace_id,
                name=name,
                model=model,
                input=input_data,
                output=output,
                metadata=metadata or {},
                usage=usage,
            )
        except Exception as e:
            logger.error(f"Error creating generation {name}: {e}")
            return None

    def score_trace(
        self,
        trace,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ):
        """
        Add a score to a trace.

        Args:
            trace: Trace object
            name: Score name (e.g., "relevance", "accuracy")
            value: Score value
            comment: Optional comment
        """
        if not trace or not self.client:
            return

        try:
            # Create a score using v2 API
            self.client.score(
                trace_id=trace.trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as e:
            logger.error(f"Error scoring trace: {e}")

    def score_by_id(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        data_type: Optional[str] = None,
        score_id: Optional[str] = None,
    ) -> bool:
        """
        Add a score to a trace by its id, for callers that no longer hold the
        trace object (user feedback, background judge).

        Args:
            trace_id: Langfuse trace id
            name: Score name
            value: Score value
            comment: Optional comment
            data_type: "NUMERIC" or "BOOLEAN" (None lets Langfuse infer)
            score_id: Stable id; re-sending the same id overwrites the score

        Returns:
            True if the score was queued, False if tracing is off or it failed
        """
        if not self.client:
            return False

        try:
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
                data_type=data_type,
                id=score_id,
            )
            return True
        except Exception as e:
            logger.error(f"Error scoring trace {trace_id}: {e}")
            return False

    def update_generation(self, generation, output: Optional[Any] = None, usage: Optional[Dict[str, Any]] = None):
        """
        Set a generation's output and token usage/cost.

        Args:
            generation: Generation object from create_generation
            output: Generated output
            usage: Langfuse usage dict (input, output, total, unit, *_cost)
        """
        if not generation:
            return

        try:
            generation.update(output=output, usage=usage)
        except Exception as e:
            logger.error(f"Error updating generation: {e}")

    def update_trace(self, trace, tags: Optional[list] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Set a trace's tags and/or metadata.

        Args:
            trace: Trace object
            tags: Tags (replace the trace's tags)
            metadata: Metadata
        """
        if not trace:
            return

        try:
            fields: Dict[str, Any] = {}
            if tags is not None:
                fields["tags"] = tags
            if metadata is not None:
                fields["metadata"] = metadata
            if fields:
                trace.update(**fields)
        except Exception as e:
            logger.error(f"Error updating trace: {e}")

    def update_span(
        self,
        span,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ):
        """
        Update a span with output or additional metadata.

        Args:
            span: Span object to update
            output: Output data
            metadata: Additional metadata
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            status_message: Status message
        """
        if not span:
            return

        try:
            # For v2 API, we can update spans with end_time and output
            if output is not None:
                # Update the span with output data
                span.update(output=output)
            if metadata:
                span.update(metadata=metadata)
            if level:
                span.update(level=level)
            if status_message:
                span.update(status_message=status_message)
        except Exception as e:
            logger.error(f"Error updating span: {e}")

    def end_span(self, span, output: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        End a span with optional final output and metadata.

        Args:
            span: Span object to end
            output: Final output data
            metadata: Final metadata
        """
        if not span:
            return

        try:
            # Update with final data if provided
            if output is not None or metadata is not None:
                self.update_span(span, output=output, metadata=metadata)

            # End the span to capture proper timing
            span.end()
        except Exception as e:
            logger.error(f"Error ending span: {e}")

    def flush(self):
        """Flush any pending traces."""
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Error flushing Langfuse: {e}")

    def shutdown(self):
        """Shutdown the Langfuse client."""
        if self.client:
            try:
                self.client.flush()
                self.client.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down Langfuse: {e}")
