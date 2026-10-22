"""Detecting questions that leak their source chunk's wording.

The failure this prevents: generate a question *from* a chunk and the model
naturally reuses that chunk's distinctive terms. BM25 then finds the chunk by
exact term match, scores brilliantly, and the evaluation concludes that lexical
retrieval works well — when all it measured was the generator copying.

Two independent checks:

1. **Trigram overlap.** What fraction of the question's content-word trigrams
   appear verbatim in the source. High overlap means the question is a
   restatement, not a question.
2. **Hapax terms.** A token occurring in exactly one chunk across the whole
   corpus is a unique fingerprint. A question containing one is trivially
   solvable by any keyword matcher, regardless of overlap.

The second catches what the first misses: a short question can reuse one very
rare term and still show low trigram overlap.
"""

import collections
import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]*")

# Function words carry no retrieval signal, so including them would dilute the
# overlap ratio and let genuinely leaky questions pass.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for
    with without from by as is are was were be been being do does did doing
    have has had having it its it's they them their there here what which who
    whom whose when where why how all any both each few more most other some
    such no nor not only own same so too very can will just should now does
    into during before after above below up down out off over under again
    further once we our you your i he she his her
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Split text into lowercase content words.

    :param text: Input text.
    :returns: Content tokens, stopwords removed.
    """
    return [token for token in _WORD.findall(text.lower()) if token not in STOPWORDS and len(token) > 1]


def trigrams(tokens: Sequence[str]) -> List[tuple]:
    """Build overlapping trigrams from a token sequence.

    Trigrams rather than single tokens: shared vocabulary is expected and fine
    (a question about LoRA will say "LoRA"), but a shared three-word *sequence*
    means the phrasing was copied.

    :param tokens: Content tokens.
    :returns: Trigrams; for fewer than three tokens, a single tuple of all of them.
    """
    if len(tokens) < 3:
        return [tuple(tokens)] if tokens else []
    return [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]


def trigram_overlap(question: str, source_text: str) -> float:
    """Fraction of the question's trigrams that also appear in the source.

    :param question: The generated question.
    :param source_text: The chunk it was generated from.
    :returns: Overlap in [0, 1]; 0.0 when the question has no content words.
    """
    question_grams = trigrams(tokenize(question))
    if not question_grams:
        return 0.0

    source_grams = set(trigrams(tokenize(source_text)))
    hits = sum(1 for gram in question_grams if gram in source_grams)
    return hits / len(question_grams)


def is_identifier_like(token: str) -> bool:
    """Whether a token is a name or identifier rather than ordinary vocabulary.

    This distinction decides what counts as a giveaway. In a 1000-chunk corpus,
    perfectly ordinary words — "quantizing", "curates", "two-stage" — land in
    exactly one chunk by accident. Rejecting a question for using them would
    penalise a researcher for writing normal English, and on a first full run
    that cost four good questions.

    A genuine fingerprint looks like a name: it carries digits (``qwen1``,
    ``110b``), or mixes letters and digits (``glm4``). Those a searcher could
    only know by having read the chunk.

    :param token: Lowercased token.
    :returns: True if the token is identifier-like.
    """
    return any(character.isdigit() for character in token)


def build_hapax_vocabulary(chunk_texts: Iterable[str]) -> Set[str]:
    """Find identifier-like tokens that occur in exactly one chunk.

    Restricted to identifier-like tokens (see :func:`is_identifier_like`);
    ordinary words that happen to be corpus-unique are not giveaways.

    :param chunk_texts: Body text of every chunk in the corpus.
    :returns: Corpus-unique tokens that act as fingerprints.
    """
    document_frequency: collections.Counter = collections.Counter()

    for text in chunk_texts:
        document_frequency.update(set(tokenize(text)))

    unique = {token for token, count in document_frequency.items() if count == 1}
    hapax = {token for token in unique if is_identifier_like(token)}

    logger.info(
        f"hapax vocabulary: {len(unique)} tokens occur in exactly one chunk, "
        f"{len(hapax)} of them are identifier-like and treated as giveaways"
    )
    return hapax


@dataclass
class LeakageVerdict:
    """Outcome of the leakage checks for one question.

    :param overlap: Trigram overlap ratio against the source chunk.
    :param hapax_hits: Corpus-unique tokens the question reuses.
    :param threshold: Overlap ratio above which the question is rejected.
    """

    overlap: float
    hapax_hits: List[str]
    threshold: float

    @property
    def leaks(self) -> bool:
        """Whether the question should be rejected.

        :returns: True if either check trips.
        """
        return self.overlap > self.threshold or bool(self.hapax_hits)

    @property
    def reason(self) -> str:
        """Human-readable explanation.

        :returns: Why it was rejected, or why it passed.
        """
        reasons = []
        if self.overlap > self.threshold:
            # One decimal place: rounding to whole percent renders a 15.4%
            # overlap as the baffling "15% exceeds 15%".
            reasons.append(f"trigram overlap {self.overlap:.1%} exceeds {self.threshold:.1%}")
        if self.hapax_hits:
            reasons.append(f"reuses corpus-unique term(s): {', '.join(sorted(self.hapax_hits)[:5])}")
        return "; ".join(reasons) if reasons else f"clean (overlap {self.overlap:.0%})"


def check_leakage(
    question: str,
    source_text: str,
    hapax: Set[str],
    threshold: float,
) -> LeakageVerdict:
    """Run both leakage checks on a question.

    :param question: The generated question.
    :param source_text: The chunk it was generated from.
    :param hapax: Corpus-unique tokens from :func:`build_hapax_vocabulary`.
    :param threshold: Maximum acceptable trigram overlap.
    :returns: The verdict.
    """
    return LeakageVerdict(
        overlap=trigram_overlap(question, source_text),
        hapax_hits=[token for token in set(tokenize(question)) if token in hapax],
        threshold=threshold,
    )


def summarise(verdicts: Dict[str, LeakageVerdict]) -> List[str]:
    """Build a report over a batch of verdicts.

    :param verdicts: qid to verdict.
    :returns: Report lines.
    """
    if not verdicts:
        return ["No questions checked."]

    leaking = {qid: verdict for qid, verdict in verdicts.items() if verdict.leaks}
    overlaps = sorted(verdict.overlap for verdict in verdicts.values())

    lines = [
        f"Leakage: {len(leaking)} of {len(verdicts)} questions rejected",
        f"  overlap min={overlaps[0]:.0%} median={overlaps[len(overlaps) // 2]:.0%} max={overlaps[-1]:.0%}",
    ]
    for qid, verdict in sorted(leaking.items()):
        lines.append(f"  {qid}: {verdict.reason}")
    return lines
