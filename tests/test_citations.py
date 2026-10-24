"""Tests for the deterministic citation metrics (Q8)."""

import pytest
from src.evaluation.generation.citations import extract_citations, extract_field_citations, normalize_id, score_citations


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2401.06831v1", "2401.06831"),
        ("2401.06831v12", "2401.06831"),
        ("2401.06831", "2401.06831"),
        (" arXiv:2401.06831v2 ", "2401.06831"),
        ("2401.06831.", "2401.06831"),
    ],
)
def test_normalize_id_strips_prefix_and_version(raw, expected):
    assert normalize_id(raw) == expected


def test_extract_handles_single_multi_and_repeated_brackets():
    answer = (
        "LoRA saves memory [arXiv:2401.00001v1]. Both agree [arXiv:2401.00001v2, arXiv:2402.00002]; "
        "see also [arxiv:2403.00003v1 and 2404.00004]."
    )
    assert extract_citations(answer) == ["2401.00001", "2402.00002", "2403.00003", "2404.00004"]


def test_extract_ignores_non_arxiv_brackets():
    assert extract_citations("As shown in [1] and [Table 2], accuracy rose.") == []


def test_all_citations_grounded():
    scores = score_citations("X [arXiv:2401.00001v1].", ["2401.00001v1", "2402.00002v1"], ["2401.00001v1"])
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["present"] == 1.0
    assert scores["hallucinated"] == []


def test_citation_to_unretrieved_paper_is_hallucinated():
    scores = score_citations("X [arXiv:2401.00001] and Y [arXiv:2499.99999].", ["2401.00001v1"], ["2401.00001v1"])
    assert scores["precision"] == 0.5
    assert scores["hallucinated"] == ["2499.99999"]


def test_no_citations_leaves_precision_undefined():
    scores = score_citations("No citations here.", ["2401.00001v1"], ["2401.00001v1"])
    assert scores["precision"] is None
    assert scores["present"] == 0.0
    assert scores["recall"] == 0.0


def test_recall_counts_only_relevant_papers_that_were_retrieved():
    # 2402 is relevant but was never retrieved, so citing it was impossible.
    scores = score_citations("X [arXiv:2401.00001].", ["2401.00001v1", "2403.00003v1"], ["2401.00001v1", "2402.00002v1"])
    assert scores["recall"] == 1.0


def test_recall_undefined_for_unanswerable():
    assert score_citations("X [arXiv:2401.00001].", ["2401.00001v1"], [])["recall"] is None


def test_field_citations_split_arxiv_ids_from_bibliography_markers():
    entries = ["arXiv:2403.13269v3", "Zhang et al., 2023", "[11]", "2401.00001", "arXiv:2403.13269v1"]
    assert extract_field_citations(entries) == {"ids": ["2403.13269", "2401.00001"], "other": ["Zhang et al., 2023", "[11]"]}
    assert extract_field_citations(None) == {"ids": [], "other": []}


def test_field_citations_ignore_digits_inside_longer_numbers():
    assert extract_field_citations(["version 12024.123456 build"])["ids"] == []


def test_extract_ignores_section_locators_inside_the_bracket():
    # Verbatim /stream citation shapes from the Phase 5 run (q0111, q0068).
    answer = "X [arXiv:2404.12267v1, Section 2.1]. Y [arXiv:2404.12267v1, Abstract]. Z [arXiv:2503.13988v1, Section: A. DATA PREPARATION]."
    assert extract_citations(answer) == ["2404.12267", "2503.13988"]
