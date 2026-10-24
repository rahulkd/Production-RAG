"""Unified factory for OpenSearch client."""

from functools import lru_cache
from typing import Optional

from src.config import Settings, get_settings

from .client import OpenSearchClient


@lru_cache(maxsize=1)
def make_opensearch_client(settings: Optional[Settings] = None) -> OpenSearchClient:
    """Factory function to create cached OpenSearch client.

    Uses lru_cache to maintain a singleton instance for efficiency.

    :param settings: Optional settings instance
    :returns: Cached OpenSearchClient instance
    """
    if settings is None:
        settings = get_settings()

    return OpenSearchClient(host=settings.opensearch.host, settings=settings)


@lru_cache(maxsize=1)
def make_search_client() -> OpenSearchClient:
    """Factory function to create the cached client the API searches with.

    Reads the index chosen by ``OPENSEARCH__ACTIVE_INDEX``: the production
    chunk index, or the frozen evaluation index. Only the API's search path uses
    this. Ingestion goes through :func:`make_opensearch_client_fresh`, which
    always defaults to the production index, so switching to the evaluation
    index never routes new papers into it.

    :returns: Cached OpenSearchClient bound to the active search index
    """
    settings = get_settings()

    return OpenSearchClient(host=settings.opensearch.host, settings=settings, index_name=settings.opensearch.search_index_name)


def make_opensearch_client_fresh(
    settings: Optional[Settings] = None,
    host: Optional[str] = None,
    index_name: Optional[str] = None,
) -> OpenSearchClient:
    """Factory function to create a fresh OpenSearch client (not cached).

    Use this when you need a new client instance (e.g., for testing
    or when connection issues occur).

    :param settings: Optional settings instance
    :param host: Optional host override
    :param index_name: Optional index override, for targeting an isolated
        index such as the evaluation corpus index
    :returns: New OpenSearchClient instance
    """
    if settings is None:
        settings = get_settings()

    # Use provided host or settings host
    opensearch_host = host or settings.opensearch.host

    return OpenSearchClient(host=opensearch_host, settings=settings, index_name=index_name)
