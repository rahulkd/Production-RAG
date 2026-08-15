"""Shared fixtures for the hermetic test suite.

Everything under ``tests/`` runs without OpenSearch, Postgres, Bedrock, Jina,
or the arXiv API. Anything needing live services lives under
``src/evaluation`` and is driven by its CLIs.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mirror src/evaluation/_bootstrap.py: parts of src/services/indexing import
# their siblings without the `src.` prefix, so src/ must be importable too.
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@pytest.fixture
def make_candidate():
    """Factory for :class:`Candidate` objects with sensible defaults.

    :returns: A callable taking ``arxiv_id`` plus optional overrides.
    """
    from src.evaluation.corpus.selection import Candidate

    def _make(
        arxiv_id: str,
        categories=("cs.AI",),
        published_date="2025-03-14T00:00:00Z",
        title=None,
        abstract="An abstract.",
    ) -> Candidate:
        return Candidate(
            arxiv_id=arxiv_id,
            title=title if title is not None else f"Paper {arxiv_id}",
            abstract=abstract,
            authors=["A. Author"],
            categories=list(categories),
            published_date=published_date,
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        )

    return _make
