"""Golden question schema and JSONL persistence.

One row per question. The row carries everything needed to (a) score retrieval
exactly, (b) let a human review the question against its source without a
database round-trip, and (c) detect when the corpus has drifted underneath the
labels.

Chunk identity is ``(arxiv_id, chunk_index)``. OpenSearch's ``_id`` is
auto-generated and changes on every reindex, so it is never ground truth.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Relevance grades. Only nDCG consumes them; the other metrics treat any listed
# chunk as relevant.
GRADE_ANSWER = 2  # Contains the answer.
GRADE_SUPPORTING = 1  # Useful context, but does not answer on its own.

QUESTION_TYPES = ("single_hop", "multi_hop", "comparative", "unanswerable", "paraphrase")
REVIEW_STATUSES = ("pending", "accepted", "edited", "rejected")


@dataclass
class RelevantChunk:
    """A chunk that should be retrieved for a question.

    :param arxiv_id: Owning paper.
    :param chunk_index: Position within the paper. With ``arxiv_id``, the
        stable ground-truth address.
    :param grade: 2 if it contains the answer, 1 if it is supporting context.
    :param chunk_text: The chunk body with the title/abstract header stripped —
        the text the question was actually written from. Stored inline so a
        reviewer can judge the pair without querying OpenSearch, and so the
        dataset stays meaningful if the index is rebuilt.
    :param section_title: Section the chunk came from.
    :param text_sha256: Hash of the chunk **as indexed**, header included.
        Re-chunking changes this, which is how stale labels are detected.
    :param answer_snippet: The sentence(s) the answer comes from. Doubles as the
        re-anchoring key: after a re-chunk, substring-matching this against the
        new chunks recovers the label without regenerating the question.
    """

    arxiv_id: str
    chunk_index: int
    grade: int = GRADE_ANSWER
    chunk_text: str = ""
    section_title: str = ""
    text_sha256: str = ""
    answer_snippet: str = ""

    @property
    def key(self) -> tuple:
        """Ground-truth address.

        :returns: ``(arxiv_id, chunk_index)``.
        """
        return (self.arxiv_id, self.chunk_index)


@dataclass
class GoldenQuestion:
    """One labelled evaluation question.

    :param qid: Stable identifier, e.g. ``q0087``.
    :param question: The question text.
    :param question_type: One of :data:`QUESTION_TYPES`.
    :param answerable: False for negative controls, whose ``relevant_chunks``
        is empty and which test whether the system refuses.
    :param relevant_chunks: Graded chunks that should be retrieved.
    :param relevant_papers: Papers the answer draws on. Stored explicitly so
        paper-level metrics survive a re-chunk that invalidates ``chunk_index``.
    :param reference_answer: A correct answer, used in Phase 4 for relevance
        scoring. Never shown to the faithfulness judge.
    :param must_mention: Literal strings a correct answer should contain. Powers
        a deterministic pre-check before the LLM judge.
    :param generated_from: Provenance of the chunk that seeded the question.
    :param paraphrase_of: For paraphrase pairs, the qid of the parent.
    :param topic_key: Corpus topic slice, for coverage reporting.
    :param generator: Model, prompt version, and sampling params that produced it.
    :param review: Human review outcome.
    """

    qid: str
    question: str
    question_type: str
    answerable: bool = True
    relevant_chunks: List[RelevantChunk] = field(default_factory=list)
    relevant_papers: List[str] = field(default_factory=list)
    reference_answer: str = ""
    must_mention: List[str] = field(default_factory=list)
    generated_from: Dict[str, Any] = field(default_factory=dict)
    paraphrase_of: Optional[str] = None
    topic_key: str = ""
    generator: Dict[str, Any] = field(default_factory=dict)
    review: Dict[str, Any] = field(default_factory=lambda: {"status": "pending", "reviewer": "", "reviewed_at": "", "notes": ""})

    @property
    def is_accepted(self) -> bool:
        """Whether this row belongs in the evaluation set.

        :returns: True when a human accepted or edited it.
        """
        return self.review.get("status") in ("accepted", "edited")

    def relevance_map(self) -> Dict[tuple, int]:
        """Ground truth in the form the retrieval metrics consume.

        :returns: ``(arxiv_id, chunk_index)`` to grade.
        """
        return {chunk.key: chunk.grade for chunk in self.relevant_chunks}

    def validate(self) -> List[str]:
        """Check the row for defects that would corrupt scoring.

        :returns: Problem descriptions; empty when the row is sound.
        """
        problems: List[str] = []

        if not self.qid:
            problems.append("missing qid")
        if not self.question.strip():
            problems.append(f"{self.qid}: empty question")
        if self.question_type not in QUESTION_TYPES:
            problems.append(f"{self.qid}: unknown question_type {self.question_type!r}")
        if self.review.get("status") not in REVIEW_STATUSES:
            problems.append(f"{self.qid}: unknown review status {self.review.get('status')!r}")

        if self.answerable and not self.relevant_chunks:
            problems.append(f"{self.qid}: answerable but has no relevant_chunks")
        if not self.answerable and self.relevant_chunks:
            problems.append(f"{self.qid}: unanswerable but lists relevant_chunks")

        for chunk in self.relevant_chunks:
            if chunk.chunk_index < 0:
                problems.append(f"{self.qid}: negative chunk_index {chunk.chunk_index}")
            if chunk.grade not in (GRADE_SUPPORTING, GRADE_ANSWER):
                problems.append(f"{self.qid}: grade must be 1 or 2, got {chunk.grade}")
            if not chunk.chunk_text.strip():
                problems.append(f"{self.qid}: chunk {chunk.key} has no stored chunk_text")

        # relevant_papers is derived, but stored; a mismatch means one of the
        # two was edited by hand and they no longer agree.
        derived = {chunk.arxiv_id for chunk in self.relevant_chunks}
        if derived != set(self.relevant_papers):
            problems.append(f"{self.qid}: relevant_papers {sorted(self.relevant_papers)} != chunks' papers {sorted(derived)}")

        if self.question_type == "multi_hop" and len(derived) < 2:
            problems.append(f"{self.qid}: multi_hop must span >= 2 papers, spans {len(derived)}")

        return problems

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-ready dictionary.

        :returns: The row as plain data.
        """
        payload = asdict(self)
        payload["relevant_chunks"] = [asdict(chunk) for chunk in self.relevant_chunks]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GoldenQuestion":
        """Rebuild a row from parsed JSON.

        :param payload: Parsed row.
        :returns: The question.
        """
        chunks = [RelevantChunk(**entry) for entry in payload.get("relevant_chunks", [])]
        fields = {key: value for key, value in payload.items() if key != "relevant_chunks"}
        return cls(relevant_chunks=chunks, **fields)


def write_jsonl(questions: Iterable[GoldenQuestion], path: Path) -> int:
    """Write questions as JSON Lines.

    :param questions: Rows to write.
    :param path: Destination. Parent directories are created.
    :returns: Number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with path.open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question.to_dict(), ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Wrote {count} questions to {path}")
    return count


def read_jsonl(path: Path, accepted_only: bool = False) -> List[GoldenQuestion]:
    """Read questions from JSON Lines.

    :param path: Source file.
    :param accepted_only: Drop rows a human has not accepted or edited. Rejected
        rows stay in the file for provenance, so the loader filters rather than
        the writer deleting.
    :returns: The rows.
    :raises FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"No dataset at {path}. Run `python -m src.evaluation.dataset.generate` first.")

    questions: List[GoldenQuestion] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(GoldenQuestion.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number} is not a valid question row: {exc}") from exc

    if accepted_only:
        questions = [question for question in questions if question.is_accepted]

    return questions


def validate_dataset(questions: List[GoldenQuestion]) -> List[str]:
    """Validate a whole dataset, including cross-row constraints.

    :param questions: Rows to check.
    :returns: Problem descriptions; empty when the dataset is sound.
    """
    problems: List[str] = []

    seen: Dict[str, int] = {}
    for question in questions:
        problems.extend(question.validate())
        seen[question.qid] = seen.get(question.qid, 0) + 1

    problems.extend(f"duplicate qid: {qid}" for qid, count in seen.items() if count > 1)

    known = {question.qid for question in questions}
    for question in questions:
        if question.paraphrase_of and question.paraphrase_of not in known:
            problems.append(f"{question.qid}: paraphrase_of {question.paraphrase_of!r} does not exist")

    return problems
