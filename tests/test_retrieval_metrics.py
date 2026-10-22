"""Tests for the retrieval metrics and sweep grid.

Every expected value is worked by hand. The oracle cases matter most: a bug that
*inflates* scores is invisible against real output, so feeding the ground truth
back in as the retrieved list must score exactly 1.0 on every metric.
"""

import math

import pytest
from src.evaluation.retrieval.configs import RetrievalConfig, sweep_a
from src.evaluation.retrieval.metrics import (
    aggregate,
    hit_rate_at_k,
    mean_ci,
    ndcg_at_k,
    paper_recall_at_k,
    recall_at_k,
    reciprocal_rank,
    score_question,
)

A = ("2401.00001v1", 7)  # answer chunk
N = ("2401.00001v1", 8)  # its neighbour
B = ("2402.00002v1", 3)  # second answer chunk, for multi-hop
X = ("2499.00009v1", 0)  # irrelevant
Y = ("2401.00001v1", 2)  # irrelevant chunk of the right paper

SINGLE = {A: 2, N: 1}
MULTI = {A: 2, B: 2}


def test_oracle_single_hop_scores_one_everywhere():
    scores = score_question([A, N], SINGLE, ["2401.00001v1"], k=5)
    assert scores == {"recall": 1.0, "ndcg": 1.0, "mrr": 1.0, "hit": 1.0, "paper_recall": 1.0}


def test_oracle_multi_hop_scores_one_everywhere():
    scores = score_question([B, A], MULTI, ["2401.00001v1", "2402.00002v1"], k=3)
    assert scores == {"recall": 1.0, "ndcg": 1.0, "mrr": 1.0, "hit": 1.0, "paper_recall": 1.0}


def test_recall_counts_answer_chunks_only():
    # The neighbour is supporting context; finding the answer chunk alone is full recall.
    assert recall_at_k([X, A, Y], SINGLE, k=3) == 1.0
    assert recall_at_k([X, N, Y], SINGLE, k=3) == 0.0


def test_recall_partial_for_multi_hop():
    assert recall_at_k([A, X, Y], MULTI, k=3) == 0.5


def test_recall_respects_cutoff():
    assert recall_at_k([X, Y, A], SINGLE, k=2) == 0.0
    assert recall_at_k([X, Y, A], SINGLE, k=3) == 1.0


def test_recall_undefined_without_ground_truth():
    with pytest.raises(ValueError):
        recall_at_k([A], {}, k=3)


def test_hit_rate():
    assert hit_rate_at_k([X, B], MULTI, k=2) == 1.0
    assert hit_rate_at_k([X, N], SINGLE, k=2) == 0.0


def test_reciprocal_rank_uses_first_answer_chunk():
    assert reciprocal_rank([X, N, A], SINGLE, k=3) == pytest.approx(1 / 3)
    assert reciprocal_rank([X, B, A], MULTI, k=3) == pytest.approx(1 / 2)


def test_reciprocal_rank_zero_past_cutoff():
    assert reciprocal_rank([X, Y, A], SINGLE, k=2) == 0.0


def test_ndcg_hand_worked():
    # Ranked [X, A, N]: DCG = 0 + 3/log2(3) + 1/log2(4) = 1.8928 + 0.5
    # Ideal  [A, N]:    IDCG = 3/log2(2) + 1/log2(3) = 3 + 0.6309
    expected = (3 / math.log2(3) + 1 / 2) / (3 + 1 / math.log2(3))
    assert ndcg_at_k([X, A, N], SINGLE, k=3) == pytest.approx(expected)


def test_ndcg_credits_neighbour_partially():
    # Only the neighbour retrieved, at rank 1: 1 / (3 + 1/log2(3)).
    assert ndcg_at_k([N, X], SINGLE, k=2) == pytest.approx(1 / (3 + 1 / math.log2(3)))


def test_ndcg_ideal_truncated_at_k():
    # k=1 with the answer chunk first is perfect even though the neighbour is missing.
    assert ndcg_at_k([A, X], SINGLE, k=1) == 1.0


def test_ndcg_zero_when_nothing_relevant():
    assert ndcg_at_k([X, Y], SINGLE, k=2) == 0.0


def test_paper_recall_credits_wrong_chunk_of_right_paper():
    ranked = [Y, X]
    assert recall_at_k(ranked, SINGLE, k=2) == 0.0
    assert paper_recall_at_k(ranked, ["2401.00001v1"], k=2) == 1.0


def test_paper_recall_partial_for_multi_paper():
    assert paper_recall_at_k([A, X], ["2401.00001v1", "2402.00002v1"], k=2) == 0.5


def test_empty_retrieval_scores_zero():
    scores = score_question([], SINGLE, ["2401.00001v1"], k=5)
    assert scores == {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0, "hit": 0.0, "paper_recall": 0.0}


def test_mean_ci():
    stat = mean_ci([1.0, 0.0, 1.0, 0.0])
    # std(ddof=1) = 0.57735; SEM = 0.288675; 1.96 * SEM = 0.565803
    assert stat["mean"] == 0.5
    assert stat["ci95"] == pytest.approx(0.565803, abs=1e-6)
    assert stat["n"] == 4


def test_mean_ci_single_value_has_zero_width():
    assert mean_ci([0.7]) == {"mean": 0.7, "ci95": 0.0, "n": 1}


def test_aggregate_covers_every_metric():
    summary = aggregate([score_question([A], SINGLE, ["2401.00001v1"], k=3)])
    assert set(summary) == {"recall", "ndcg", "mrr", "hit", "paper_recall"}


def test_sweep_a_is_twelve_configs():
    configs = sweep_a()
    assert len(configs) == 12
    assert configs[0] == RetrievalConfig("bm25", 3)
    assert configs[-1] == RetrievalConfig("hybrid", 20)


def test_sweep_a_narrowed():
    assert [c.name for c in sweep_a(["hybrid"], [10, 3])] == ["hybrid k=3", "hybrid k=10"]


def test_sweep_a_rejects_unknown_mode():
    with pytest.raises(ValueError):
        sweep_a(["splade"])
