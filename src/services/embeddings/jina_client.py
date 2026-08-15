import asyncio
import logging
import random
from typing import Any, Dict, List

import httpx
from src.schemas.embeddings.jina import JinaEmbeddingRequest, JinaEmbeddingResponse

logger = logging.getLogger(__name__)

# Statuses worth retrying: 429 is Jina's rate limiter, the 5xx range is
# transient server trouble. Anything else (401, 400, ...) is a real error that
# retrying cannot fix.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class JinaEmbeddingsClient:
    """Client for Jina AI embeddings API.

    Uses Jina embeddings v3 model with 1024 dimensions optimized for retrieval.
    Documentation: https://jina.ai/embeddings

    Requests are retried with exponential backoff on 429 and 5xx. Without this,
    bulk indexing walks straight into Jina's token-per-minute limit: a 50-paper
    corpus is roughly 1M embedding tokens, and issuing it back-to-back exceeds
    the quota in seconds. A single unretried 429 then permanently loses that
    paper, since the exception propagates out of the indexing service.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.jina.ai/v1",
        max_retries: int = 5,
        base_retry_delay: float = 2.0,
        max_retry_delay: float = 90.0,
    ):
        """Initialize Jina embeddings client.

        :param api_key: Jina API key
        :param base_url: API base URL
        :param max_retries: Retry attempts after the initial request
        :param base_retry_delay: First backoff delay in seconds; doubles each attempt
        :param max_retry_delay: Ceiling for a single backoff wait. Defaults above
            60s because the rate limit is per *minute* — a shorter cap would
            retry inside the same exhausted window and fail again.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.max_retry_delay = max_retry_delay
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("Jina embeddings client initialized")

    def _retry_delay(self, attempt: int, response: httpx.Response | None) -> float:
        """Compute how long to wait before the next attempt.

        Honours a ``Retry-After`` header when the server sends one, otherwise
        backs off exponentially. Jitter is added so concurrent callers do not
        resynchronise and hit the limit together on the next attempt.

        :param attempt: Zero-based index of the attempt that just failed.
        :param response: The failed response, if there was one.
        :returns: Seconds to sleep.
        """
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), self.max_retry_delay)
                except ValueError:
                    pass  # Retry-After may be an HTTP date; fall through to backoff.

        delay = min(self.base_retry_delay * (2**attempt), self.max_retry_delay)
        return delay + random.uniform(0, delay * 0.25)

    async def _post_embeddings(self, request_data: JinaEmbeddingRequest, description: str) -> Dict[str, Any]:
        """POST to the embeddings endpoint, retrying transient failures.

        :param request_data: The request body.
        :param description: Short label for log messages, e.g. "batch of 50 passages".
        :returns: The parsed JSON response.
        :raises httpx.HTTPError: If every attempt fails, or the failure is not retryable.
        """
        url = f"{self.base_url}/embeddings"
        payload = request_data.model_dump()

        for attempt in range(self.max_retries + 1):
            response = None
            try:
                response = await self.client.post(url, headers=self.headers, json=payload)

                if response.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                    delay = self._retry_delay(attempt, response)
                    logger.warning(
                        f"Jina returned {response.status_code} for {description}; "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                return response.json()

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # Network-level failure: no response to inspect, but still worth retrying.
                if attempt >= self.max_retries:
                    logger.error(f"Jina request failed for {description} after {attempt + 1} attempts: {exc}")
                    raise
                delay = self._retry_delay(attempt, None)
                logger.warning(
                    f"Jina request error for {description}: {exc}; "
                    f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(delay)

        # Loop exhausted on retryable statuses; surface the last one.
        logger.error(f"Jina rate limit not cleared for {description} after {self.max_retries} retries")
        response.raise_for_status()
        raise httpx.HTTPError(f"Jina request failed for {description}")  # pragma: no cover - defensive

    async def embed_passages(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Embed text passages for indexing.

        :param texts: List of text passages to embed
        :param batch_size: Number of texts to process in each API call
        :returns: List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            request_data = JinaEmbeddingRequest(
                model="jina-embeddings-v3", task="retrieval.passage", dimensions=1024, input=batch
            )

            try:
                payload = await self._post_embeddings(request_data, f"batch of {len(batch)} passages")

                result = JinaEmbeddingResponse(**payload)
                batch_embeddings = [item["embedding"] for item in result.data]
                embeddings.extend(batch_embeddings)

                logger.debug(f"Embedded batch of {len(batch)} passages")

            except httpx.HTTPError as e:
                logger.error(f"Error embedding passages: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in embed_passages: {e}")
                raise

        logger.info(f"Successfully embedded {len(texts)} passages")
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a search query.

        :param query: Query text to embed
        :returns: Embedding vector for the query
        """
        request_data = JinaEmbeddingRequest(model="jina-embeddings-v3", task="retrieval.query", dimensions=1024, input=[query])

        try:
            payload = await self._post_embeddings(request_data, "query")

            result = JinaEmbeddingResponse(**payload)
            embedding = result.data[0]["embedding"]

            logger.debug(f"Embedded query: '{query[:50]}...'")
            return embedding

        except httpx.HTTPError as e:
            logger.error(f"Error embedding query: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in embed_query: {e}")
            raise

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
