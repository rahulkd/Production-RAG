"""Selecting the corpus from raw arXiv results.

Pure functions over plain data — no network, no database — so the selection
rules can be tested exactly. :func:`select_corpus` is the whole policy:

1. **Filter** candidates that don't genuinely belong to the topic's category
   or fall outside the date window. ``cat:`` matching on arXiv is permissive,
   so results do come back that need dropping.
2. **Claim in topic order, quota-limited.** Each paper is claimed by at most
   one topic, keyed on the version-stripped arXiv ID. Cross-listed papers are
   common (a Gen-AI paper is routinely both cs.AI and cs.LG) and would
   otherwise be counted twice, inflating apparent diversity.
3. **Backfill shortfalls round-robin.** A topic whose query returns too few
   usable papers would leave the category under target. Leftovers are taken
   one at a time across topics rather than draining a single topic, so a
   shortfall degrades spread as little as possible.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from src.evaluation.corpus.topics import Topic

logger = logging.getLogger(__name__)

# Matches the trailing version marker on an arXiv ID: 2501.01234v2 -> v2.
# Anchored so a "v" inside an old-style ID (e.g. cs/0501001) is left alone.
_VERSION_SUFFIX = re.compile(r"v\d+$")


def base_arxiv_id(arxiv_id: str) -> str:
    """Strip the version suffix from an arXiv ID.

    Deduplication must be version-insensitive: ``2501.01234v1`` and
    ``2501.01234v2`` are the same paper, and two topic queries run seconds
    apart can legitimately return different versions.

    :param arxiv_id: ID with or without a version suffix.
    :returns: The version-stripped ID.
    """
    return _VERSION_SUFFIX.sub("", arxiv_id.strip())


@dataclass(frozen=True)
class Candidate:
    """A paper returned by arXiv, normalised for selection.

    Decoupled from ``ArxivPaper`` so the selection rules can be tested without
    constructing pydantic models.

    :param arxiv_id: Full ID including version suffix.
    :param title: Paper title.
    :param abstract: Paper abstract.
    :param authors: Author names.
    :param categories: arXiv category terms, primary first.
    :param published_date: Submission date as returned by arXiv (ISO-8601).
    :param pdf_url: Direct PDF link.
    """

    arxiv_id: str
    title: str
    abstract: str
    authors: Sequence[str]
    categories: Sequence[str]
    published_date: str
    pdf_url: str = ""

    @property
    def base_id(self) -> str:
        """Version-stripped arXiv ID, used as the dedup key."""
        return base_arxiv_id(self.arxiv_id)

    @property
    def primary_category(self) -> str:
        """arXiv lists the primary category first in the entry's category set."""
        return self.categories[0] if self.categories else ""

    @property
    def published_year_month(self) -> str:
        """Submission month as ``YYYYMM``, for the date-window check."""
        return self.published_date[:7].replace("-", "") if self.published_date else ""


@dataclass
class Selected:
    """A candidate that made it into the corpus, with its provenance.

    :param candidate: The underlying paper.
    :param topic_key: Topic slice that claimed it.
    :param via_backfill: True when it filled another topic's shortfall rather
        than being claimed by its own topic in the first pass.
    """

    candidate: Candidate
    topic_key: str
    via_backfill: bool = False


@dataclass
class SelectionReport:
    """Outcome of a selection pass, for logging and manifest provenance.

    :param selected: Chosen papers in claim order.
    :param per_topic: Count claimed per topic key.
    :param shortfalls: Topic keys that could not meet quota, and by how much.
    :param rejected: Count of discarded candidates keyed by reason.
    """

    selected: List[Selected] = field(default_factory=list)
    per_topic: Dict[str, int] = field(default_factory=dict)
    shortfalls: Dict[str, int] = field(default_factory=dict)
    rejected: Dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Number of papers selected."""
        return len(self.selected)

    def primary_category_counts(self) -> Dict[str, int]:
        """Distribution of selected papers over their primary category.

        :returns: Primary category to count, useful for spotting a corpus that
            has drifted into one sub-field.
        """
        counts: Dict[str, int] = {}
        for item in self.selected:
            counts[item.candidate.primary_category] = counts.get(item.candidate.primary_category, 0) + 1
        return counts


def _is_eligible(
    candidate: Candidate,
    category: str,
    from_yyyymm: str,
    to_yyyymm: str,
    report: SelectionReport,
) -> bool:
    """Check a candidate against the category and date-window requirements.

    :param candidate: Paper to test.
    :param category: Category the topic slice is drawing from.
    :param from_yyyymm: Inclusive lower bound, ``YYYYMM``.
    :param to_yyyymm: Inclusive upper bound, ``YYYYMM``.
    :param report: Report to tally rejection reasons into.
    :returns: True when the candidate may be claimed.
    """
    if category not in candidate.categories:
        report.rejected["wrong_category"] = report.rejected.get("wrong_category", 0) + 1
        return False

    stamp = candidate.published_year_month
    if not stamp:
        report.rejected["missing_date"] = report.rejected.get("missing_date", 0) + 1
        return False
    if stamp < from_yyyymm or stamp > to_yyyymm:
        report.rejected["out_of_date_window"] = report.rejected.get("out_of_date_window", 0) + 1
        return False

    if not candidate.title.strip():
        report.rejected["missing_title"] = report.rejected.get("missing_title", 0) + 1
        return False

    return True


def select_corpus(
    topics: Sequence[Topic],
    candidates_by_topic: Mapping[str, Iterable[Candidate]],
    from_date: str,
    to_date: str,
    claimed: Optional[set] = None,
    target_total: Optional[int] = None,
) -> SelectionReport:
    """Select a deduplicated, quota-balanced corpus slice for one category.

    :param topics: Topic slices to draw from, all of the same category.
    :param candidates_by_topic: Topic key to the papers its query returned,
        best-first.
    :param from_date: Inclusive lower bound, ``YYYYMMDD``.
    :param to_date: Inclusive upper bound, ``YYYYMMDD``.
    :param claimed: Base IDs already taken by an earlier call. Mutated in
        place so a second category's selection cannot re-take a paper.
    :param target_total: Papers wanted for this category. Defaults to the sum
        of the topic quotas.
    :returns: The selection outcome.
    :raises ValueError: If ``topics`` is empty or mixes categories.
    """
    if not topics:
        raise ValueError("at least one topic is required")

    categories = {topic.category for topic in topics}
    if len(categories) > 1:
        raise ValueError(f"all topics must share one category, got {sorted(categories)}")
    category = categories.pop()

    if claimed is None:
        claimed = set()
    if target_total is None:
        target_total = sum(topic.quota for topic in topics)

    from_yyyymm, to_yyyymm = from_date[:6], to_date[:6]
    report = SelectionReport()

    # Pass 1 — each topic claims up to its quota, in declaration order.
    leftovers: Dict[str, List[Candidate]] = {}
    for topic in topics:
        pool = list(candidates_by_topic.get(topic.key, []))
        taken = 0
        remainder: List[Candidate] = []

        for candidate in pool:
            if candidate.base_id in claimed:
                report.rejected["duplicate"] = report.rejected.get("duplicate", 0) + 1
                continue
            if not _is_eligible(candidate, category, from_yyyymm, to_yyyymm, report):
                continue

            if taken < topic.quota:
                claimed.add(candidate.base_id)
                report.selected.append(Selected(candidate=candidate, topic_key=topic.key))
                taken += 1
            else:
                remainder.append(candidate)

        report.per_topic[topic.key] = taken
        leftovers[topic.key] = remainder

        if taken < topic.quota:
            report.shortfalls[topic.key] = topic.quota - taken
            logger.warning(f"Topic {topic.key!r} filled {taken}/{topic.quota} — query returned too few usable papers")

    # Pass 2 — round-robin over leftovers to reach the category target.
    if report.total < target_total:
        deficit = target_total - report.total
        logger.info(f"Backfilling {deficit} paper(s) for {category} from leftover candidates")

        cursors = {key: 0 for key in leftovers}
        progressed = True
        while report.total < target_total and progressed:
            progressed = False
            for topic in topics:
                if report.total >= target_total:
                    break
                pool = leftovers[topic.key]
                index = cursors[topic.key]
                while index < len(pool):
                    candidate = pool[index]
                    index += 1
                    if candidate.base_id in claimed:
                        continue
                    claimed.add(candidate.base_id)
                    report.selected.append(Selected(candidate=candidate, topic_key=topic.key, via_backfill=True))
                    report.per_topic[topic.key] = report.per_topic.get(topic.key, 0) + 1
                    progressed = True
                    break
                cursors[topic.key] = index

        if report.total < target_total:
            logger.warning(
                f"{category}: selected {report.total}/{target_total} papers — "
                "not enough distinct candidates. Widen the date window or add topic terms."
            )

    return report
