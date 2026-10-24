"""Tests for the Phase 4 aggregation: synthetic rows, hand-worked expectations."""

import pytest
from src.evaluation.generation.runner import must_mention_rate, summarise


def cites(cited=(), hallucinated=(), recall=None):
    return {
        "cited": list(cited),
        "hallucinated": list(hallucinated),
        "present": 1.0 if cited else 0.0,
        "precision": (len(cited) - len(hallucinated)) / len(cited) if cited else None,
        "recall": recall,
    }


def row(qid, answerable=True, refused=False, faith=None, contradicted=0, relevance=None, hit=1.0, citations=None, **extra):
    faithfulness = None
    if faith is not None:
        faithfulness = {"score": faith, "claims": 4, "supported": int(faith * 4), "contradicted": contradicted, "not_in_context": 0}
    return {
        "qid": qid,
        "question_type": "single_hop" if answerable else "unanswerable",
        "answerable": answerable,
        "retrieval": {"hit": hit} if answerable else None,
        "json_ok": True,
        "stop_reason": "end_turn",
        "refusal": {"label": "refused" if refused else "attempted", "reason": ""},
        "faithfulness": faithfulness,
        "relevance": {"score": relevance, "reason": ""} if relevance is not None else None,
        "citations": citations or cites(),
        "field_citations": {**cites(), "other": []},
        "must_mention": None,
        "judge_errors": [],
        **extra,
    }


ROWS = [
    row("q1", faith=1.0, relevance=3, citations=cites(["2401.00001"], recall=1.0)),
    row("q2", faith=0.5, contradicted=1, relevance=2, citations=cites(["2401.00001", "2499.00009"], ["2499.00009"])),
    row("q3", refused=True, relevance=0, hit=0.0),  # answer chunk never retrieved: correct to refuse
    row("q4", refused=True, relevance=0, hit=1.0),  # chunk was there: a false refusal
    row("u1", answerable=False, refused=True),
    row("u2", answerable=False, faith=0.25),  # attempted an unanswerable question
    {"qid": "q5", "question_type": "single_hop", "answerable": True, "generation_error": "context too long", "judge_errors": []},
]


def test_refusal_rates():
    summary = summarise(ROWS)
    assert summary["refusal"]["recall"]["mean"] == pytest.approx(0.5)  # u1 of u1, u2
    assert summary["refusal"]["false_refusal_rate"]["mean"] == pytest.approx(0.5)  # q3, q4 of q1-q4


def test_generation_errors_are_excluded_and_counted():
    summary = summarise(ROWS)
    assert summary["rows"] == 7
    assert summary["generation_errors"] == 1
    assert summary["relevance"]["n"] == 4


def test_faithfulness_over_attempted_answers_with_contradictions_separate():
    summary = summarise(ROWS)
    assert summary["faithfulness"]["score"]["mean"] == pytest.approx((1.0 + 0.5 + 0.25) / 3)
    assert summary["faithfulness"]["answers_with_contradiction"]["mean"] == pytest.approx(1 / 3)
    assert summary["faithfulness_answerable"]["score"]["n"] == 2
    assert summary["faithfulness_unanswerable"]["score"]["mean"] == pytest.approx(0.25)


def test_citation_precision_only_over_answers_that_cite():
    citations = summarise(ROWS)["citations"]
    assert citations["precision"]["mean"] == pytest.approx((1.0 + 0.5) / 2)
    assert citations["present_rate"]["mean"] == pytest.approx(2 / 6)
    assert citations["hallucinated_total"] == 1


def test_bottleneck_splits_on_retrieval_hit():
    bottleneck = summarise(ROWS)["bottleneck"]
    assert (bottleneck["retrieved"]["n"], bottleneck["retrieved"]["answered"]) == (3, 2)
    assert (bottleneck["missed"]["n"], bottleneck["missed"]["answered"]) == (1, 0)


def test_must_mention_is_case_insensitive():
    assert must_mention_rate("Uses SVM and Transfer Learning.", ["svm", "transfer learning", "clustering", "NN"]) == 0.5
    assert must_mention_rate("anything", []) is None
