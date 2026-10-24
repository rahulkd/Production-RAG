"""Citation accuracy (Q8) — deterministic, no LLM.

The system prompt tells the generator to cite as ``[arXiv:id]``. A cited ID that
is not among the retrieved papers cannot have come from the context, so it is a
hallucinated citation; ``citation_precision`` is the headline number.

Citations are paper-level only: ``_prepare_chunks_and_sources`` hands the LLM
``{arxiv_id, chunk_text}`` and nothing finer (C7).
"""

import re
from typing import Dict, Iterable, List, Optional, Set

# One bracket may hold several IDs: "[arXiv:2401.00001v1, arXiv:2402.00002]".
_BRACKET = re.compile(r"\[\s*arXiv:([^\]]+)\]", re.IGNORECASE)
_ID_SEPARATORS = re.compile(r"[,;]|\band\b")
_VERSION_SUFFIX = re.compile(r"v\d+$")
# New-style arXiv IDs (YYMM.NNNNN), version dropped. The corpus is 2024+, so
# old-style "cs/0101001" IDs cannot occur.
_ARXIV_ID = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])")


def normalize_id(arxiv_id: str) -> str:
    """Strip whitespace, an ``arXiv:`` prefix and the version suffix.

    ``ask.py`` builds source URLs from the version-stripped ID, so ``2401.00001``
    and ``2401.00001v2`` are the same paper here too.

    :param arxiv_id: Raw ID as written by the model or stored in the index.
    :returns: e.g. ``"2401.00001"``.
    """
    cleaned = arxiv_id.strip().strip(".")
    cleaned = re.sub(r"^arxiv:\s*", "", cleaned, flags=re.IGNORECASE)
    return _VERSION_SUFFIX.sub("", cleaned)


def extract_citations(answer: str) -> List[str]:
    """Every distinct arXiv ID cited inline, in first-seen order.

    :param answer: Generated answer text.
    :returns: Normalized IDs.
    """
    cited: List[str] = []
    for bracket in _BRACKET.findall(answer):
        for part in _ID_SEPARATORS.split(bracket):
            arxiv_id = normalize_id(part)
            # Locators share the bracket ("[arXiv:2404.12267v1, Section 2.1]");
            # only parts shaped like an arXiv ID are citations.
            if _ARXIV_ID.fullmatch(arxiv_id) and arxiv_id not in cited:
                cited.append(arxiv_id)
    return cited


def extract_field_citations(entries: Optional[Iterable[str]]) -> Dict[str, List[str]]:
    """arXiv IDs in the structured ``citations`` field of the JSON answer.

    With the structured prompt, Llama puts its citations here instead of inline,
    and mixes in the papers' own bibliography markers ("[11]", "Zhang et al.,
    2023") copied from the chunk text. Those are kept apart as ``other``.

    :param entries: The parsed ``citations`` list, or None.
    :returns: ``ids`` (normalized, first-seen order) and ``other`` (entries with no arXiv ID).
    """
    ids: List[str] = []
    other: List[str] = []
    for entry in entries or []:
        found = _ARXIV_ID.findall(str(entry))
        if not found:
            other.append(str(entry))
        for arxiv_id in found:
            if arxiv_id not in ids:
                ids.append(arxiv_id)
    return {"ids": ids, "other": other}


def score_citations(answer: str, retrieved_papers: Iterable[str], relevant_papers: Iterable[str]) -> Dict[str, Optional[float]]:
    """Citation metrics for one answer's inline ``[arXiv:id]`` citations.

    :param answer: Generated answer text.
    :param retrieved_papers: Papers whose chunks were in the prompt.
    :param relevant_papers: Ground-truth papers; empty for unanswerable questions.
    :returns: See :func:`score_cited_ids`.
    """
    return score_cited_ids(extract_citations(answer), retrieved_papers, relevant_papers)


def score_cited_ids(cited: List[str], retrieved_papers: Iterable[str], relevant_papers: Iterable[str]) -> Dict[str, Optional[float]]:
    """Citation metrics for a list of cited paper IDs.

    :param cited: Normalized cited IDs.
    :param retrieved_papers: Papers whose chunks were in the prompt.
    :param relevant_papers: Ground-truth papers; empty for unanswerable questions.
    :returns: ``cited`` (IDs), ``hallucinated`` (cited but not retrieved),
        ``present`` (1.0 if anything is cited), ``precision`` (cited and
        retrieved / cited; None with no citations) and ``recall`` (relevant
        retrieved papers that are cited / relevant retrieved papers; None when
        no relevant paper was retrieved).
    """
    retrieved: Set[str] = {normalize_id(paper) for paper in retrieved_papers}
    relevant_in_context = {normalize_id(paper) for paper in relevant_papers} & retrieved

    grounded = [arxiv_id for arxiv_id in cited if arxiv_id in retrieved]
    return {
        "cited": cited,
        "hallucinated": [arxiv_id for arxiv_id in cited if arxiv_id not in retrieved],
        "present": 1.0 if cited else 0.0,
        "precision": len(grounded) / len(cited) if cited else None,
        "recall": len(relevant_in_context & set(cited)) / len(relevant_in_context) if relevant_in_context else None,
    }
