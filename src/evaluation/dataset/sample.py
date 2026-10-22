"""Choosing which chunks seed which questions.

Two properties the sampler has to guarantee:

**Every paper is covered.** A corpus paper that no question touches is dead
weight in the index — it adds distractors without ever being a target, so it
inflates the apparent difficulty of retrieval without testing anything. Every
one of the 43 papers gets at least one single-hop question before any paper
gets a second.

**Chunks are spread within a paper.** Sampling the first N eligible chunks
would draw every question from introductions, which are the most abstract-like
and least discriminating part of a paper. Chunks are picked at evenly spaced
positions through the paper instead.

Selection is deterministic given the corpus: a fixed seed plus stable sort
order, so re-running produces the same seeds and the dataset stays
regenerable.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from src.evaluation.dataset.chunks import Chunk, group_by_paper

logger = logging.getLogger(__name__)

# Fixed so the same corpus always yields the same seed chunks.
SAMPLING_SEED = 20260809


@dataclass
class PairSeed:
    """Two chunks from different papers, seeding a multi-hop or comparative question.

    :param first: Chunk from the first paper.
    :param second: Chunk from the second paper.
    :param shared_topic: Topic slice both papers belong to, when they share one.
    """

    first: Chunk
    second: Chunk
    shared_topic: str = ""


def _spread_indices(available: int, wanted: int) -> List[int]:
    """Pick evenly spaced positions from a sequence.

    Sampling positions rather than a random subset keeps questions spread from
    a paper's method sections through to its results, instead of clustering in
    whichever region happens to have the most chunks.

    :param available: Number of items to choose from.
    :param wanted: How many to pick.
    :returns: Sorted, de-duplicated positions.
    """
    if wanted >= available:
        return list(range(available))
    if wanted <= 0:
        return []
    if wanted == 1:
        return [available // 2]  # Middle: past the intro, before the conclusion.

    step = (available - 1) / (wanted - 1)
    return sorted({int(round(position * step)) for position in range(wanted)})


def sample_single_hop(chunks: Sequence[Chunk], target: int) -> List[Chunk]:
    """Pick seed chunks for single-hop questions, covering every paper.

    Round one gives each paper a chunk from its middle. Remaining slots go to
    papers with the most eligible chunks, since those support the most distinct
    questions without repeating themselves.

    :param chunks: Eligible chunks.
    :param target: How many seeds to pick.
    :returns: Seed chunks, one per intended question.
    :raises ValueError: If there are fewer papers than... nothing; a target
        below the paper count is allowed but logged, since it breaks coverage.
    """
    by_paper = group_by_paper(chunks)
    papers = sorted(by_paper)

    if target < len(papers):
        logger.warning(
            f"target {target} is below the {len(papers)} papers in the corpus; "
            "some papers will have no question and become pure distractors"
        )

    seeds: List[Chunk] = []
    used: set = set()

    # Round one: one chunk per paper, from the middle of the paper.
    for arxiv_id in papers:
        if len(seeds) >= target:
            break
        paper_chunks = by_paper[arxiv_id]
        chunk = paper_chunks[len(paper_chunks) // 2]
        seeds.append(chunk)
        used.add(chunk.key)

    # Round two: distribute the remainder, richest papers first.
    remaining = target - len(seeds)
    if remaining > 0:
        by_size = sorted(papers, key=lambda a: (-len(by_paper[a]), a))
        extra_per_paper: Dict[str, int] = {arxiv_id: 0 for arxiv_id in papers}

        while remaining > 0:
            progressed = False
            for arxiv_id in by_size:
                if remaining <= 0:
                    break
                paper_chunks = [c for c in by_paper[arxiv_id] if c.key not in used]
                if not paper_chunks:
                    continue
                positions = _spread_indices(len(paper_chunks), extra_per_paper[arxiv_id] + 1)
                chunk = paper_chunks[positions[-1]]
                seeds.append(chunk)
                used.add(chunk.key)
                extra_per_paper[arxiv_id] += 1
                remaining -= 1
                progressed = True
            if not progressed:
                logger.warning(f"ran out of eligible chunks with {remaining} single-hop slots unfilled")
                break

    covered = {chunk.arxiv_id for chunk in seeds}
    logger.info(f"single-hop: {len(seeds)} seeds covering {len(covered)}/{len(papers)} papers")
    return seeds


def sample_pairs(
    chunks: Sequence[Chunk],
    topic_by_paper: Dict[str, str],
    target: int,
    used_keys: set,
) -> List[PairSeed]:
    """Pick chunk pairs from different papers for multi-hop and comparative questions.

    Pairs are drawn from papers sharing a topic slice where possible. Two
    papers on unrelated subjects cannot support a question that genuinely needs
    both — the generator would produce two questions stapled together, which
    tests nothing about multi-document retrieval.

    :param chunks: Eligible chunks.
    :param topic_by_paper: ``arxiv_id`` to topic slice.
    :param target: How many pairs to pick.
    :param used_keys: Chunk keys already used as seeds; avoided where possible
        so questions do not all point at the same few chunks.
    :returns: The pairs.
    """
    by_paper = group_by_paper(chunks)

    by_topic: Dict[str, List[str]] = {}
    for arxiv_id in sorted(by_paper):
        topic = topic_by_paper.get(arxiv_id, "")
        by_topic.setdefault(topic, []).append(arxiv_id)

    rng = random.Random(SAMPLING_SEED)

    # Every within-topic pairing, plus cross-topic fallbacks if that is not enough.
    candidates: List[Tuple[str, str, str]] = []
    for topic, papers in sorted(by_topic.items()):
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                candidates.append((papers[i], papers[j], topic))

    rng.shuffle(candidates)

    if len(candidates) < target:
        # Deliberately NOT padded with cross-topic pairings. Two papers from
        # unrelated topics share no comparable dimension, so the generator
        # rightly declines them — a run that padded 13 such pairs spent ~20 API
        # calls to produce nothing. The number of same-topic pairings is a
        # property of the corpus (30 for 43 papers over 20 topics), and the
        # honest response is to generate fewer questions, not worse ones.
        logger.warning(
            f"corpus offers only {len(candidates)} same-topic pairings but {target} were requested; "
            f"capping at {len(candidates)}. Add more papers per topic to raise this ceiling."
        )

    pairs: List[PairSeed] = []
    paper_usage: Dict[str, int] = {}

    for first_id, second_id, topic in candidates:
        if len(pairs) >= target:
            break
        # Spread pairings across papers rather than reusing two prolific ones.
        if paper_usage.get(first_id, 0) >= 2 or paper_usage.get(second_id, 0) >= 2:
            continue

        first = _pick_chunk(by_paper[first_id], used_keys, rng)
        second = _pick_chunk(by_paper[second_id], used_keys, rng)
        if first is None or second is None:
            continue

        pairs.append(PairSeed(first=first, second=second, shared_topic=topic))
        used_keys.add(first.key)
        used_keys.add(second.key)
        paper_usage[first_id] = paper_usage.get(first_id, 0) + 1
        paper_usage[second_id] = paper_usage.get(second_id, 0) + 1

    if len(pairs) < target:
        logger.warning(f"only produced {len(pairs)} of {target} requested pairs")

    logger.info(f"pairs: {len(pairs)} ({sum(1 for p in pairs if p.shared_topic)} same-topic)")
    return pairs


def _pick_chunk(paper_chunks: Sequence[Chunk], used_keys: set, rng: random.Random):
    """Pick a chunk from a paper, preferring one not already used as a seed.

    :param paper_chunks: The paper's eligible chunks.
    :param used_keys: Chunk keys already used.
    :param rng: Seeded random source.
    :returns: A chunk, or None if the paper has none.
    """
    if not paper_chunks:
        return None
    fresh = [chunk for chunk in paper_chunks if chunk.key not in used_keys]
    pool = fresh or list(paper_chunks)
    return pool[rng.randrange(len(pool))]


def coverage_report(seeds: Sequence[Chunk], pairs: Sequence[PairSeed], all_papers: Sequence[str]) -> Dict[str, int]:
    """Count how many questions each paper will seed.

    :param seeds: Single-hop seed chunks.
    :param pairs: Pair seeds.
    :param all_papers: Every paper in the corpus.
    :returns: ``arxiv_id`` to seeded-question count, including zeros.
    """
    counts = {arxiv_id: 0 for arxiv_id in all_papers}

    for chunk in seeds:
        counts[chunk.arxiv_id] = counts.get(chunk.arxiv_id, 0) + 1
    for pair in pairs:
        counts[pair.first.arxiv_id] = counts.get(pair.first.arxiv_id, 0) + 1
        counts[pair.second.arxiv_id] = counts.get(pair.second.arxiv_id, 0) + 1

    return counts
