"""arXiv query construction for topic-scoped corpus discovery.

Builds the ``search_query`` string consumed by
:meth:`ArxivClient.fetch_papers_with_query`. Kept free of I/O so the exact
query text is unit-testable without touching the network.

Query shape::

    cat:cs.AI AND (abs:"LoRA" OR ti:"LoRA" OR ...) AND submittedDate:[202401010000 TO 202608092359]

Notes on arXiv's syntax, learned the hard way:

* Multi-word phrases must be quoted, otherwise ``abs:large language model``
  parses as ``abs:large`` AND two bare terms.
* ``submittedDate`` bounds are ``YYYYMMDDHHMM``, inclusive on both ends.
* ``cat:`` matches *any* of a paper's categories, not just the primary one, so
  primary-category balancing has to happen client-side after the fetch.
"""

from datetime import datetime
from typing import Sequence

from src.evaluation.corpus.topics import Topic

# Fields a topic term is matched against. Title carries the strongest signal
# but is short; abstract catches papers whose title is a coined name.
SEARCH_FIELDS = ("ti", "abs")


def _quote(term: str) -> str:
    """Wrap a term in the double quotes arXiv needs for phrase matching.

    :param term: Raw search term.
    :returns: The quoted term.
    :raises ValueError: If the term is empty or already contains a quote,
        which would produce a malformed query.
    """
    cleaned = term.strip()
    if not cleaned:
        raise ValueError("search term is empty")
    if '"' in cleaned:
        raise ValueError(f"search term must not contain a quote character: {term!r}")
    return f'"{cleaned}"'


def build_date_clause(from_date: str, to_date: str) -> str:
    """Build the ``submittedDate`` range clause.

    :param from_date: Inclusive lower bound, ``YYYYMMDD``.
    :param to_date: Inclusive upper bound, ``YYYYMMDD``.
    :returns: Clause like ``submittedDate:[202401010000 TO 202608092359]``.
    :raises ValueError: If either date is malformed or the range is inverted.
    """
    for label, value in (("from_date", from_date), ("to_date", to_date)):
        if len(value) != 8 or not value.isdigit():
            raise ValueError(f"{label} must be YYYYMMDD, got {value!r}")
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError as exc:
            raise ValueError(f"{label} is not a valid date: {value!r}") from exc

    if from_date > to_date:
        raise ValueError(f"from_date {from_date} is after to_date {to_date}")

    return f"submittedDate:[{from_date}0000 TO {to_date}2359]"


def build_terms_clause(terms: Sequence[str], fields: Sequence[str] = SEARCH_FIELDS) -> str:
    """Build the parenthesised OR-clause over a topic's terms.

    Every term is matched in every field, so a 4-term topic across 2 fields
    produces 8 OR-ed conditions.

    :param terms: Topic search terms.
    :param fields: arXiv field prefixes to match against.
    :returns: Clause like ``(ti:"LoRA" OR abs:"LoRA")``.
    :raises ValueError: If no terms or no fields are supplied.
    """
    if not terms:
        raise ValueError("at least one search term is required")
    if not fields:
        raise ValueError("at least one search field is required")

    conditions = [f"{field}:{_quote(term)}" for term in terms for field in fields]
    return "(" + " OR ".join(conditions) + ")"


def build_topic_query(topic: Topic, from_date: str, to_date: str) -> str:
    """Build the full arXiv query for one topic slice.

    :param topic: The topic to query for.
    :param from_date: Inclusive lower bound, ``YYYYMMDD``.
    :param to_date: Inclusive upper bound, ``YYYYMMDD``.
    :returns: A complete ``search_query`` string.
    """
    return " AND ".join(
        (
            f"cat:{topic.category}",
            build_terms_clause(topic.terms),
            build_date_clause(from_date, to_date),
        )
    )


def today_stamp() -> str:
    """Today's date as ``YYYYMMDD``, for use as the default upper bound.

    :returns: Date string in arXiv's format.
    """
    return datetime.now().strftime("%Y%m%d")
