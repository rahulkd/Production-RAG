"""Tests for the Jina embeddings client's retry behaviour.

A 50-paper corpus is roughly 1M embedding tokens. Issued back-to-back it
exceeds Jina's per-minute token limit within seconds, and before retries were
added a single 429 permanently lost that paper — the exception propagated out
of ``HybridIndexingService.index_paper``, which caught it and returned
``chunks_created=0``. That is how 36 of 50 papers went missing from a build
that otherwise reported success.
"""

import httpx
import pytest
from src.schemas.embeddings.jina import JinaEmbeddingRequest
from src.services.embeddings.jina_client import RETRYABLE_STATUS, JinaEmbeddingsClient


def make_client(**overrides) -> JinaEmbeddingsClient:
    """Build a client with near-zero delays so tests stay fast.

    :param overrides: Constructor overrides.
    :returns: The client.
    """
    kwargs = {"api_key": "test-key", "max_retries": 3, "base_retry_delay": 0.001, "max_retry_delay": 0.01}
    kwargs.update(overrides)
    return JinaEmbeddingsClient(**kwargs)


def embedding_payload(count: int = 1) -> dict:
    """Build a well-formed Jina response body.

    :param count: Number of embeddings to include.
    :returns: The response payload.
    """
    return {
        "model": "jina-embeddings-v3",
        "object": "list",
        "usage": {"total_tokens": 10, "prompt_tokens": 10},
        "data": [{"object": "embedding", "index": i, "embedding": [0.1] * 1024} for i in range(count)],
    }


class FakeTransport(httpx.AsyncBaseTransport):
    """Transport returning a scripted sequence of responses.

    :param statuses: Status codes to return in order; the last repeats.
    :param headers: Optional headers attached to every response.
    """

    def __init__(self, statuses, headers=None):
        self.statuses = list(statuses)
        self.headers = headers or {}
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        body = embedding_payload() if status == 200 else {"detail": "rate limited"}
        return httpx.Response(status, json=body, headers=self.headers, request=request)


def attach(client: JinaEmbeddingsClient, transport: FakeTransport) -> JinaEmbeddingsClient:
    """Point a client at a fake transport.

    :param client: Client to rewire.
    :param transport: Transport to install.
    :returns: The same client.
    """
    client.client = httpx.AsyncClient(transport=transport, timeout=5.0)
    return client


@pytest.mark.anyio
class TestRetryOnRateLimit:
    async def test_succeeds_after_429(self):
        """The exact failure mode from the build: 429 then success."""
        transport = FakeTransport([429, 200])
        client = attach(make_client(), transport)

        result = await client.embed_query("hello")

        assert len(result) == 1024
        assert transport.calls == 2

    async def test_retries_until_max_then_raises(self):
        transport = FakeTransport([429])
        client = attach(make_client(max_retries=3), transport)

        with pytest.raises(httpx.HTTPStatusError):
            await client.embed_query("hello")

        # Initial attempt plus three retries.
        assert transport.calls == 4

    @pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
    async def test_every_retryable_status_is_retried(self, status):
        transport = FakeTransport([status, 200])
        client = attach(make_client(), transport)

        await client.embed_query("hello")

        assert transport.calls == 2

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    async def test_client_errors_are_not_retried(self, status):
        """A bad key or malformed request will never succeed on retry."""
        transport = FakeTransport([status])
        client = attach(make_client(), transport)

        with pytest.raises(httpx.HTTPStatusError):
            await client.embed_query("hello")

        assert transport.calls == 1

    async def test_no_retry_when_first_attempt_succeeds(self):
        transport = FakeTransport([200])
        client = attach(make_client(), transport)

        await client.embed_query("hello")

        assert transport.calls == 1


@pytest.mark.anyio
class TestPassageBatches:
    async def test_batch_recovers_from_rate_limit(self):
        transport = FakeTransport([429, 429, 200])
        client = attach(make_client(), transport)

        result = await client.embed_passages(["a"], batch_size=50)

        assert len(result) == 1
        assert transport.calls == 3

    async def test_exhausted_retries_propagate(self):
        """index_paper relies on this raising, so it can record the failure."""
        transport = FakeTransport([429])
        client = attach(make_client(max_retries=2), transport)

        with pytest.raises(httpx.HTTPStatusError):
            await client.embed_passages(["a"], batch_size=50)


class TestBackoff:
    def test_delay_grows_exponentially(self):
        client = make_client(base_retry_delay=2.0, max_retry_delay=100.0)
        delays = [client._retry_delay(attempt, None) for attempt in range(4)]

        # Jitter adds up to 25%, so compare against the un-jittered floors.
        assert delays[0] >= 2.0
        assert delays[1] >= 4.0
        assert delays[2] >= 8.0
        assert delays[3] >= 16.0

    def test_delay_is_capped(self):
        client = make_client(base_retry_delay=2.0, max_retry_delay=10.0)
        # 2 * 2**10 would be 2048s without the cap; jitter is applied on top.
        assert client._retry_delay(10, None) <= 10.0 * 1.25

    def test_jitter_desynchronises_callers(self):
        """Identical backoff would make concurrent callers collide again."""
        client = make_client(base_retry_delay=2.0, max_retry_delay=100.0)
        delays = {client._retry_delay(2, None) for _ in range(20)}
        assert len(delays) > 1

    def test_retry_after_header_is_honoured(self):
        client = make_client(base_retry_delay=2.0, max_retry_delay=100.0)
        response = httpx.Response(429, headers={"Retry-After": "37"})
        assert client._retry_delay(0, response) == 37.0

    def test_retry_after_is_capped(self):
        client = make_client(max_retry_delay=30.0)
        response = httpx.Response(429, headers={"Retry-After": "3600"})
        assert client._retry_delay(0, response) == 30.0

    def test_non_numeric_retry_after_falls_back_to_backoff(self):
        """Retry-After may be an HTTP date, which float() cannot parse."""
        client = make_client(base_retry_delay=2.0, max_retry_delay=100.0)
        response = httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
        assert client._retry_delay(0, response) >= 2.0

    def test_max_retry_delay_exceeds_a_minute_by_default(self):
        """Jina's limit is per minute; a shorter cap retries inside the same
        exhausted window and fails again."""
        assert JinaEmbeddingsClient(api_key="k").max_retry_delay > 60


@pytest.fixture
def anyio_backend():
    """Restrict anyio to asyncio; trio is not a project dependency at runtime.

    :returns: The backend name.
    """
    return "asyncio"
