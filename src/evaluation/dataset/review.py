"""Human review of the generated question set.

This step is not optional. An LLM-generated golden set that nobody checked
measures the question generator, not the retriever: if the generator writes a
vague question, the retriever "fails" on a question no system could answer, and
the resulting number is noise wearing a metric's clothes.

The loop shows each question beside the chunk it came from and takes one
keystroke. Rejected rows are kept in the file with their status rather than
deleted, so the set's history stays auditable; the loader filters them out.

Usage::

    python -m src.evaluation.dataset.review
    python -m src.evaluation.dataset.review --status pending   # resume
    python -m src.evaluation.dataset.review --type single_hop
"""

import argparse
import getpass
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import src.evaluation._bootstrap  # noqa: F401
from src.evaluation import config as eval_config
from src.evaluation.dataset.schema import GoldenQuestion, read_jsonl, validate_dataset, write_jsonl

logger = logging.getLogger(__name__)

WIDTH = 78
RULE = "=" * WIDTH

HELP = """
  a  accept        the question is good as written
  e  edit          fix the wording, then accept
  r  reject        unanswerable, vague, leaky, or refers to "the paper"
  s  skip          decide later (stays pending)
  b  back          revisit the previous question
  q  quit          save and exit
  ?  help
"""


def wrap(text: str, indent: str = "  ") -> str:
    """Wrap text to the review column width.

    :param text: Text to wrap.
    :param indent: Prefix for each line.
    :returns: Wrapped text.
    """
    import textwrap

    return "\n".join(
        textwrap.fill(line, WIDTH - len(indent), initial_indent=indent, subsequent_indent=indent)
        for line in text.splitlines() or [""]
    )


def render(question: GoldenQuestion, position: int, total: int) -> str:
    """Render one question and its source for review.

    Shows the chunk text inline — stored on the row precisely so this needs no
    OpenSearch round-trip and stays readable if the index is rebuilt.

    :param question: The question under review.
    :param position: 1-based position in the queue.
    :param total: Queue length.
    :returns: The rendered block.
    """
    lines = [
        "",
        RULE,
        f"[{position}/{total}]  {question.qid}  ({question.question_type}){'  UNANSWERABLE' if not question.answerable else ''}",
        RULE,
        "",
        "QUESTION:",
        wrap(question.question),
        "",
    ]

    if question.reference_answer:
        lines += ["REFERENCE ANSWER:", wrap(question.reference_answer), ""]
    if question.must_mention:
        lines += [f"MUST MENTION: {', '.join(question.must_mention)}", ""]

    if question.review.get("status") == "rejected":
        lines += [f"!! already rejected by {question.review.get('reviewer')}: {question.review.get('notes')}", ""]

    if question.paraphrase_of:
        lines += [f"PARAPHRASE OF: {question.paraphrase_of}", ""]

    if not question.relevant_chunks:
        lines += ["SOURCE: none — this is a negative control.", ""]
        why = question.generated_from.get("why")
        if why:
            lines += [f"  claimed reason: {why}", ""]
    for chunk in question.relevant_chunks:
        grade = "ANSWER" if chunk.grade == 2 else "SUPPORTING"
        lines += [
            f"SOURCE [{grade}] {chunk.arxiv_id} chunk {chunk.chunk_index}"
            f"{f'  ({chunk.section_title})' if chunk.section_title else ''}:",
        ]
        if chunk.answer_snippet:
            lines += [wrap(f">> {chunk.answer_snippet}"), ""]
        body = chunk.chunk_text
        lines += [wrap(body[:900] + ("..." if len(body) > 900 else "")), ""]

    return "\n".join(lines)


def prompt_choice() -> str:
    """Read one review decision from the terminal.

    :returns: The chosen action key.
    """
    while True:
        choice = input("  [a]ccept [e]dit [r]eject [s]kip [b]ack [q]uit ? ").strip().lower()
        if choice in ("a", "e", "r", "s", "b", "q"):
            return choice
        if choice == "?":
            print(HELP)
        else:
            print("  Unrecognised. Press ? for help.")


def stamp(status: str, reviewer: str, notes: str = "") -> dict:
    """Build a review record.

    :param status: One of the review statuses.
    :param reviewer: Who reviewed it.
    :param notes: Optional free text.
    :returns: The record.
    """
    return {
        "status": status,
        "reviewer": reviewer,
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    }


def run_review(questions: List[GoldenQuestion], queue: List[int], reviewer: str, output: Path) -> None:
    """Drive the interactive review loop.

    Saves after every decision, so an interrupted session loses nothing.

    :param questions: Full dataset, mutated in place.
    :param queue: Indices into ``questions`` to review, in order.
    :param reviewer: Name recorded on each decision.
    :param output: Where to save after each decision.
    """
    position = 0

    while position < len(queue):
        question = questions[queue[position]]
        print(render(question, position + 1, len(queue)))

        choice = prompt_choice()

        if choice == "q":
            break
        if choice == "b":
            position = max(0, position - 1)
            continue
        if choice == "s":
            position += 1
            continue

        if choice == "a":
            question.review = stamp("accepted", reviewer)
        elif choice == "r":
            reason = input("  reason (optional): ").strip()
            question.review = stamp("rejected", reviewer, reason)
        elif choice == "e":
            print(f"  current: {question.question}")
            edited = input("  new question (blank to keep): ").strip()
            if edited:
                question.question = edited
            answer = input("  new reference answer (blank to keep): ").strip()
            if answer:
                question.reference_answer = answer
            question.review = stamp("edited", reviewer, "manually edited during review")

        write_jsonl(questions, output)
        position += 1


def summarise(questions: Sequence[GoldenQuestion]) -> None:
    """Print counts by status and type, and paper coverage.

    :param questions: The dataset.
    """
    by_status: dict = {}
    by_type: dict = {}
    covered: set = set()

    for question in questions:
        status = question.review.get("status", "pending")
        by_status[status] = by_status.get(status, 0) + 1
        if question.is_accepted:
            by_type[question.question_type] = by_type.get(question.question_type, 0) + 1
            covered.update(question.relevant_papers)

    print("\n" + RULE)
    print("REVIEW SUMMARY")
    print(RULE)
    for status in ("accepted", "edited", "rejected", "pending"):
        if status in by_status:
            print(f"  {status:<10} {by_status[status]}")
    print(f"\n  accepted by type: {by_type}")
    print(f"  papers covered by accepted questions: {len(covered)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.dataset.review",
        description="Review generated questions and promote them to the golden set.",
    )
    parser.add_argument("--input", type=Path, default=eval_config.GOLDEN_DRAFT, help="Draft JSONL to review")
    parser.add_argument("--output", type=Path, default=None, help="Where to save (defaults to --input, in place)")
    parser.add_argument("--reviewer", default=None, help="Name recorded on each decision")
    parser.add_argument(
        "--status",
        default="pending",
        choices=["pending", "rejected", "accepted", "edited", "all"],
        help="Which rows to queue. Default 'pending' resumes an interrupted session.",
    )
    parser.add_argument("--type", dest="qtype", default=None, help="Only review this question type")
    parser.add_argument("--promote", action="store_true", help="Write accepted rows to the golden set and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    eval_config.ensure_dirs()

    try:
        questions = read_jsonl(args.input)
    except FileNotFoundError as exc:
        print(str(exc))
        return 2

    if args.promote:
        accepted = [question for question in questions if question.is_accepted]
        if not accepted:
            print("No accepted questions to promote. Review the draft first.")
            return 1

        problems = validate_dataset(accepted)
        for problem in problems[:20]:
            print(f"  schema problem: {problem}")
        if problems:
            print(f"\n{len(problems)} schema problems — not promoting.")
            return 1

        write_jsonl(accepted, eval_config.GOLDEN_SET)
        print(f"Promoted {len(accepted)} questions to {eval_config.GOLDEN_SET}")
        summarise(questions)
        return 0

    output = args.output or args.input
    reviewer = args.reviewer or getpass.getuser()

    queue = [
        index
        for index, question in enumerate(questions)
        if (args.status == "all" or question.review.get("status") == args.status)
        and (args.qtype is None or question.question_type == args.qtype)
    ]

    if not queue:
        print(f"Nothing to review with status={args.status!r}" + (f", type={args.qtype!r}" if args.qtype else ""))
        summarise(questions)
        return 0

    print(f"Reviewing {len(queue)} question(s) as {reviewer!r}. Press ? for help, q to save and quit.")
    try:
        run_review(questions, queue, reviewer, output)
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted — decisions so far are saved.")

    write_jsonl(questions, output)
    summarise(questions)
    print(f"\nWhen finished: python -m src.evaluation.dataset.review --promote --input {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
