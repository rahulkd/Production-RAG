"""Retrieval configurations under test.

Sweep A crosses every retrieval mode with every cutoff. It answers Q1 (is
hybrid better than either leg alone?) and Q2 (is ``top_k=3`` starving the
generator?), and produces the per-mode latency profile.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

MODES = ("bm25", "vector", "hybrid")

# 3 is the production default (`AskRequest.top_k`). 20 is past the API's
# `le=10` cap (C4), which is why the sweep calls the services in-process.
TOP_KS = (3, 5, 10, 20)


@dataclass(frozen=True)
class RetrievalConfig:
    """One point in the sweep.

    :param mode: ``bm25``, ``vector`` or ``hybrid``.
    :param top_k: Chunks retrieved, and the cutoff every metric is taken at.
    """

    mode: str
    top_k: int

    @property
    def name(self) -> str:
        """Short label for logs and the leaderboard.

        :returns: e.g. ``"hybrid k=5"``.
        """
        return f"{self.mode} k={self.top_k}"

    @property
    def needs_embedding(self) -> bool:
        """Whether the query must be embedded first.

        :returns: True for vector and hybrid.
        """
        return self.mode != "bm25"


def sweep_a(modes: Optional[Sequence[str]] = None, top_ks: Optional[Sequence[int]] = None) -> List[RetrievalConfig]:
    """Build the Sweep A grid, optionally narrowed.

    Each ``(mode, k)`` is a real query at ``size=k`` rather than a truncation
    of one ``k=20`` run. For BM25 and vector the two are identical, but hybrid
    fetches ``size * 2`` candidates per leg before RRF fusion, so its top 3 at
    ``size=3`` can differ from the top 3 of ``size=20``. Querying each cutoff
    measures what production would actually return.

    :param modes: Subset of :data:`MODES`; all when None.
    :param top_ks: Subset of cutoffs; :data:`TOP_KS` when None.
    :returns: Configs ordered by mode, then k.
    :raises ValueError: On an unknown mode or a non-positive k.
    """
    modes = tuple(modes or MODES)
    top_ks = tuple(top_ks or TOP_KS)

    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise ValueError(f"unknown retrieval mode(s) {unknown}; expected a subset of {list(MODES)}")
    if any(k < 1 for k in top_ks):
        raise ValueError(f"top_k values must be >= 1, got {list(top_ks)}")

    return [RetrievalConfig(mode=mode, top_k=k) for mode in modes for k in sorted(top_ks)]
