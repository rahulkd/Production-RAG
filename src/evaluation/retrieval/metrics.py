"""Retrieval metrics over exact chunk addresses.

Every metric is a pure function of a ranked list of ``(arxiv_id, chunk_index)``
keys and the question's ground truth. No LLM, no services, no randomness — the
same inputs always give the same number, which is what makes runs comparable.

**What counts as relevant.** Ground truth carries two grades: 2 for a chunk
that contains the answer, 1 for its neighbour (which shares ~100 words of
overlap and so often carries part of the claim). Only nDCG uses the grades.
Recall, Hit and MRR count *answer chunks only* (grade >= ``min_grade``,
default 2). Counting the neighbour too would make a single-hop question that
retrieved its answer chunk score Recall = 0.5, which reads as "half the
evidence is missing" when the generator in fact has everything it needs.
For multi-hop and comparative rows both chunks are grade 2, so Recall is the
fraction of the required evidence that was retrieved.
"""

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.evaluation.dataset.schema import GRADE_ANSWER

ChunkKey = Tuple[str, int]


def _answer_keys(relevant: Dict[ChunkKey, int], min_grade: int) -> set:
    """Ground-truth chunks at or above a grade.

    :param relevant: ``(arxiv_id, chunk_index)`` to grade.
    :param min_grade: Lowest grade that counts.
    :returns: The qualifying keys.
    """
    return {key for key, grade in relevant.items() if grade >= min_grade}


def recall_at_k(ranked: Sequence[ChunkKey], relevant: Dict[ChunkKey, int], k: int, min_grade: int = GRADE_ANSWER) -> float:
    """Fraction of answer chunks that appear in the top ``k``.

    :param ranked: Retrieved keys, best first.
    :param relevant: Ground truth, key to grade.
    :param k: Cutoff.
    :param min_grade: Lowest grade counted as relevant.
    :returns: Recall in [0, 1].
    :raises ValueError: If the question has no qualifying ground truth.
    """
    targets = _answer_keys(relevant, min_grade)
    if not targets:
        raise ValueError("recall is undefined for a question with no relevant chunks")
    return len(targets & set(ranked[:k])) / len(targets)


def hit_rate_at_k(ranked: Sequence[ChunkKey], relevant: Dict[ChunkKey, int], k: int, min_grade: int = GRADE_ANSWER) -> float:
    """Whether any answer chunk appears in the top ``k``.

    :param ranked: Retrieved keys, best first.
    :param relevant: Ground truth, key to grade.
    :param k: Cutoff.
    :param min_grade: Lowest grade counted as relevant.
    :returns: 1.0 on a hit, else 0.0.
    """
    targets = _answer_keys(relevant, min_grade)
    return 1.0 if targets & set(ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[ChunkKey], relevant: Dict[ChunkKey, int], k: int, min_grade: int = GRADE_ANSWER) -> float:
    """``1 / rank`` of the first answer chunk within the top ``k``.

    :param ranked: Retrieved keys, best first.
    :param relevant: Ground truth, key to grade.
    :param k: Cutoff; an answer chunk below it scores 0.
    :param min_grade: Lowest grade counted as relevant.
    :returns: Reciprocal rank in [0, 1].
    """
    targets = _answer_keys(relevant, min_grade)
    for position, key in enumerate(ranked[:k], start=1):
        if key in targets:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[ChunkKey], relevant: Dict[ChunkKey, int], k: int) -> float:
    """Normalised discounted cumulative gain with graded relevance.

    Gain is ``2**grade - 1``, so an answer chunk (grade 2, gain 3) is worth three
    times a neighbour (grade 1, gain 1). The ideal ranking places every
    ground-truth chunk first, highest grade first.

    :param ranked: Retrieved keys, best first.
    :param relevant: Ground truth, key to grade.
    :param k: Cutoff.
    :returns: nDCG in [0, 1].
    :raises ValueError: If the question has no ground truth.
    """
    if not relevant:
        raise ValueError("nDCG is undefined for a question with no relevant chunks")

    def dcg(grades: Iterable[int]) -> float:
        return sum((2**grade - 1) / math.log2(position + 1) for position, grade in enumerate(grades, start=1))

    actual = dcg(relevant.get(key, 0) for key in ranked[:k])
    ideal = dcg(sorted(relevant.values(), reverse=True)[:k])
    return actual / ideal


def paper_recall_at_k(ranked: Sequence[ChunkKey], relevant_papers: Sequence[str], k: int) -> float:
    """Fraction of the question's papers that have any chunk in the top ``k``.

    Reported alongside chunk-level recall because every chunk repeats its
    paper's title and abstract (constraint C1 in the plan). A retriever can
    land on the right paper but the wrong chunk; a large gap between this and
    :func:`recall_at_k` is the signature of that.

    :param ranked: Retrieved keys, best first.
    :param relevant_papers: Papers the answer draws on.
    :param k: Cutoff.
    :returns: Paper recall in [0, 1].
    :raises ValueError: If the question names no papers.
    """
    targets = set(relevant_papers)
    if not targets:
        raise ValueError("paper recall is undefined for a question with no relevant papers")
    retrieved = {arxiv_id for arxiv_id, _ in ranked[:k]}
    return len(targets & retrieved) / len(targets)


def score_question(ranked: Sequence[ChunkKey], relevant: Dict[ChunkKey, int], relevant_papers: Sequence[str], k: int) -> Dict[str, float]:
    """Every retrieval metric for one question at one cutoff.

    :param ranked: Retrieved keys, best first.
    :param relevant: Ground truth, key to grade.
    :param relevant_papers: Papers the answer draws on.
    :param k: Cutoff.
    :returns: Metric name to value.
    """
    return {
        "recall": recall_at_k(ranked, relevant, k),
        "ndcg": ndcg_at_k(ranked, relevant, k),
        "mrr": reciprocal_rank(ranked, relevant, k),
        "hit": hit_rate_at_k(ranked, relevant, k),
        "paper_recall": paper_recall_at_k(ranked, relevant_papers, k),
    }


METRIC_NAMES = ("recall", "ndcg", "mrr", "hit", "paper_recall")


def mean_ci(values: Sequence[float]) -> Dict[str, float]:
    """Mean with a 95% confidence half-width (``1.96 * SEM``).

    With ~100 questions the half-width is typically ±0.05–0.08, so two configs
    whose intervals overlap heavily have not been shown to differ.

    :param values: Per-question scores.
    :returns: ``{"mean", "ci95", "n"}``; ``ci95`` is 0 when ``n < 2``.
    """
    array = np.asarray(values, dtype=float)
    n = int(array.size)
    if n == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "n": 0}
    half_width = 1.96 * array.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    return {"mean": float(array.mean()), "ci95": float(half_width), "n": n}


def aggregate(per_question: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Summarise per-question metrics across a question set.

    :param per_question: One metric dict per question.
    :returns: Metric name to :func:`mean_ci` result.
    """
    return {name: mean_ci([row[name] for row in per_question]) for name in METRIC_NAMES}
