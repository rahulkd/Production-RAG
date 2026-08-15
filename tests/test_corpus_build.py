"""Tests for the build step's pure helpers.

The build CLI itself needs Postgres, OpenSearch, Jina, and the arXiv API, so
it is not exercised here. What *is* testable in isolation is the boundary
where a stored ``Paper`` row is converted into the payload the indexer
consumes — a mismatch there produces chunks with missing metadata rather than
an error, which would be invisible until retrieval metrics came out wrong.
"""

import argparse
from datetime import datetime

import pytest
from src.evaluation.corpus.build import base_id_of, paper_to_index_payload, positive_int
from src.models.paper import Paper


def make_paper(**overrides) -> Paper:
    """Build an unsaved ``Paper`` row for a test.

    :param overrides: Field values to override the defaults.
    :returns: The ORM object, not attached to a session.
    """
    defaults = {
        "arxiv_id": "2501.01234v1",
        "title": "A Paper",
        "authors": ["A. Author", "B. Author"],
        "abstract": "An abstract.",
        "categories": ["cs.AI", "cs.LG"],
        "published_date": datetime(2025, 3, 14, 12, 0, 0),
        "pdf_url": "https://arxiv.org/pdf/2501.01234.pdf",
        "raw_text": "Full extracted text.",
        "sections": [{"title": "1 Intro", "content": "..."}],
    }
    defaults.update(overrides)
    return Paper(**defaults)


class TestPositiveInt:
    def test_accepts_positive(self):
        assert positive_int("3") == 3

    @pytest.mark.parametrize("bad", ["0", "-1"])
    def test_rejects_non_positive(self, bad):
        """`--limit 0` is falsy and would otherwise mean 'process everything'."""
        with pytest.raises(argparse.ArgumentTypeError, match=">= 1"):
            positive_int(bad)

    def test_rejects_non_numeric(self):
        with pytest.raises(argparse.ArgumentTypeError, match="expected an integer"):
            positive_int("all")


class TestBaseIdOf:
    def test_strips_version(self):
        assert base_id_of("2501.01234v2") == "2501.01234"

    def test_idempotent(self):
        assert base_id_of(base_id_of("2501.01234v2")) == "2501.01234"


class TestIndexPayload:
    def test_carries_every_field_the_indexer_reads(self):
        payload = paper_to_index_payload(make_paper())
        # These keys are read by HybridIndexingService.index_paper and the
        # chunk_data it builds; a missing one silently degrades the index.
        for key in ("id", "arxiv_id", "title", "abstract", "authors", "categories", "published_date", "raw_text", "sections"):
            assert key in payload, f"indexer reads {key!r} but the payload omits it"

    def test_published_date_is_iso_string(self):
        """The OpenSearch mapping declares `date`; a string serialises predictably."""
        payload = paper_to_index_payload(make_paper())
        assert payload["published_date"] == "2025-03-14T12:00:00"

    def test_null_published_date_survives(self):
        assert paper_to_index_payload(make_paper(published_date=None))["published_date"] is None

    def test_none_text_fields_become_empty_strings(self):
        """chunk_paper() concatenates these; None would raise mid-index."""
        payload = paper_to_index_payload(make_paper(title=None, abstract=None, raw_text=None))
        assert payload["title"] == ""
        assert payload["abstract"] == ""
        assert payload["raw_text"] == ""

    def test_none_list_fields_become_empty_lists(self):
        payload = paper_to_index_payload(make_paper(authors=None, categories=None))
        assert payload["authors"] == []
        assert payload["categories"] == []

    def test_id_is_stringified(self):
        """paper_id is indexed as a keyword, so it must not stay a UUID object."""
        assert isinstance(paper_to_index_payload(make_paper())["id"], str)

    def test_sections_passed_through_untouched(self):
        """The chunker branches on sections; re-shaping them here would change chunking."""
        sections = [{"title": "2 Method", "content": "text"}]
        assert paper_to_index_payload(make_paper(sections=sections))["sections"] == sections
