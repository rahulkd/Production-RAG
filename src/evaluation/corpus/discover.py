"""Discover the evaluation corpus and write the manifest.

Runs one arXiv query per topic slice, selects a deduplicated and balanced set
of papers, and writes ``corpus_manifest.json``. Downloads nothing and touches
neither Postgres nor OpenSearch — that is :mod:`src.evaluation.corpus.build`.

Usage::

    python -m src.evaluation.corpus.discover
    python -m src.evaluation.corpus.discover --dry-run      # show queries only
    python -m src.evaluation.corpus.discover --from-date 20240101 --to-date 20260809

arXiv asks for ~3s between requests and ``ArxivClient`` enforces that
internally, so 20 topic queries take roughly a minute.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import src.evaluation._bootstrap  # noqa: F401  (must precede src.services imports)
from src.config import get_settings
from src.evaluation import config as eval_config
from src.evaluation.corpus.manifest import CorpusManifest, ManifestPaper
from src.evaluation.corpus.query import build_topic_query, today_stamp
from src.evaluation.corpus.selection import Candidate, SelectionReport, select_corpus
from src.evaluation.corpus.topics import CS_AI_TOPICS, CS_LG_TOPICS, Topic, validate_topics
from src.evaluation.logging_setup import setup_logging
from src.schemas.arxiv.paper import ArxivPaper
from src.services.arxiv.factory import make_arxiv_client

logger = logging.getLogger(__name__)

# Over-fetch per topic so there is slack for papers dropped as duplicates,
# out-of-window, or cross-listed into the wrong category, and so the backfill
# pass has leftovers to draw on.
FETCH_MULTIPLIER = 6
MIN_FETCH_PER_TOPIC = 15


def to_candidate(paper: ArxivPaper) -> Candidate:
    """Normalise an ``ArxivPaper`` into a selection :class:`Candidate`.

    :param paper: Paper as parsed from the arXiv API.
    :returns: The normalised candidate.
    """
    return Candidate(
        arxiv_id=paper.arxiv_id,
        title=(paper.title or "").strip(),
        abstract=(paper.abstract or "").strip(),
        authors=list(paper.authors or []),
        categories=list(paper.categories or []),
        published_date=str(paper.published_date or ""),
        pdf_url=paper.pdf_url or "",
    )


async def fetch_topic_candidates(
    arxiv_client,
    topics: Sequence[Topic],
    from_date: str,
    to_date: str,
    sort_by: str,
) -> Dict[str, List[Candidate]]:
    """Run one arXiv query per topic and collect the results.

    A failing topic is logged and skipped rather than aborting the run — the
    backfill pass can usually absorb one missing slice, and losing 19 good
    queries to one bad one is a poor trade.

    :param arxiv_client: Configured ``ArxivClient``.
    :param topics: Topic slices to query for.
    :param from_date: Inclusive lower bound, ``YYYYMMDD``.
    :param to_date: Inclusive upper bound, ``YYYYMMDD``.
    :param sort_by: arXiv sort key (``relevance`` or ``submittedDate``).
    :returns: Topic key to the candidates its query returned.
    """
    results: Dict[str, List[Candidate]] = {}

    for index, topic in enumerate(topics, start=1):
        query = build_topic_query(topic, from_date, to_date)
        fetch_size = max(topic.quota * FETCH_MULTIPLIER, MIN_FETCH_PER_TOPIC)

        logger.info(f"[{index}/{len(topics)}] {topic.category} :: {topic.key} (quota {topic.quota}, fetching {fetch_size})")
        logger.debug(f"  query: {query}")

        try:
            papers = await arxiv_client.fetch_papers_with_query(
                search_query=query,
                max_results=fetch_size,
                sort_by=sort_by,
                sort_order="descending",
            )
        except Exception as exc:
            logger.error(f"  topic {topic.key!r} query failed, skipping: {exc}")
            results[topic.key] = []
            continue

        results[topic.key] = [to_candidate(paper) for paper in papers]
        logger.info(f"  -> {len(results[topic.key])} candidates")

    return results


def log_selection(category: str, report: SelectionReport, topics: Sequence[Topic]) -> None:
    """Log a per-topic breakdown of one category's selection.

    :param category: The category selected for.
    :param report: Outcome of the selection pass.
    :param topics: The topics that were offered.
    """
    logger.info(f"--- {category}: selected {report.total} papers ---")
    for topic in topics:
        count = report.per_topic.get(topic.key, 0)
        marker = "  " if count >= topic.quota else " !"
        logger.info(f"{marker} {topic.key:<26} {count}/{topic.quota}  {topic.label}")

    if report.rejected:
        summary = ", ".join(f"{reason}={count}" for reason, count in sorted(report.rejected.items()))
        logger.info(f"   rejected during selection: {summary}")

    backfilled = sum(1 for item in report.selected if item.via_backfill)
    if backfilled:
        logger.info(f"   {backfilled} paper(s) came from backfill")


def build_manifest(
    reports: Sequence[SelectionReport],
    from_date: str,
    to_date: str,
) -> CorpusManifest:
    """Assemble a discovered (not yet built) manifest from selection reports.

    :param reports: One report per category, in the order to record papers.
    :param from_date: Discovery lower bound, ``YYYYMMDD``.
    :param to_date: Discovery upper bound, ``YYYYMMDD``.
    :returns: A manifest whose papers have no text hashes yet.
    """
    settings = get_settings()

    papers: List[ManifestPaper] = []
    for report in reports:
        for item in report.selected:
            candidate = item.candidate
            papers.append(
                ManifestPaper(
                    arxiv_id=candidate.arxiv_id,
                    base_id=candidate.base_id,
                    title=candidate.title,
                    primary_category=candidate.primary_category,
                    categories=list(candidate.categories),
                    published_date=candidate.published_date,
                    topic_key=item.topic_key,
                    via_backfill=item.via_backfill,
                )
            )

    return CorpusManifest(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        date_window={"from": from_date, "to": to_date},
        index_name=eval_config.EVAL_INDEX_NAME,
        embedding_model="jina-embeddings-v3",
        embedding_dim=settings.opensearch.vector_dimension,
        chunking={
            "chunk_size": settings.chunking.chunk_size,
            "overlap_size": settings.chunking.overlap_size,
            "min_chunk_size": settings.chunking.min_chunk_size,
        },
        papers=papers,
    )


async def discover(
    from_date: str,
    to_date: str,
    sort_by: str,
    output: Path,
    dry_run: bool,
) -> int:
    """Run discovery end to end.

    :param from_date: Inclusive lower bound, ``YYYYMMDD``.
    :param to_date: Inclusive upper bound, ``YYYYMMDD``.
    :param sort_by: arXiv sort key.
    :param output: Manifest destination.
    :param dry_run: Print the queries and exit without calling arXiv.
    :returns: Process exit code.
    """
    problems = validate_topics()
    if problems:
        for problem in problems:
            logger.error(f"topic definition problem: {problem}")
        return 2

    all_topics = list(CS_AI_TOPICS) + list(CS_LG_TOPICS)

    if dry_run:
        logger.info("Dry run — queries that would be issued:\n")
        for topic in all_topics:
            print(f"# {topic.category} :: {topic.key} (quota {topic.quota})")
            print(build_topic_query(topic, from_date, to_date))
            print()
        logger.info(f"{len(all_topics)} queries, target {eval_config.TARGET_TOTAL} papers. Nothing fetched.")
        return 0

    arxiv_client = make_arxiv_client()

    logger.info(f"Discovering corpus: window {from_date}..{to_date}, sort_by={sort_by}")
    logger.info(f"Targets: cs.AI={eval_config.TARGET_CS_AI}, cs.LG={eval_config.TARGET_CS_LG}")

    # cs.AI is selected first so that a paper cross-listed into both categories
    # is attributed to the Gen-AI slice it was actually chosen for. `claimed` is
    # shared across both passes to keep the two categories disjoint.
    claimed: set = set()

    ai_candidates = await fetch_topic_candidates(arxiv_client, CS_AI_TOPICS, from_date, to_date, sort_by)
    ai_report = select_corpus(
        topics=CS_AI_TOPICS,
        candidates_by_topic=ai_candidates,
        from_date=from_date,
        to_date=to_date,
        claimed=claimed,
        target_total=eval_config.TARGET_CS_AI,
    )
    log_selection("cs.AI", ai_report, CS_AI_TOPICS)

    lg_candidates = await fetch_topic_candidates(arxiv_client, CS_LG_TOPICS, from_date, to_date, sort_by)
    lg_report = select_corpus(
        topics=CS_LG_TOPICS,
        candidates_by_topic=lg_candidates,
        from_date=from_date,
        to_date=to_date,
        claimed=claimed,
        target_total=eval_config.TARGET_CS_LG,
    )
    log_selection("cs.LG", lg_report, CS_LG_TOPICS)

    manifest = build_manifest([ai_report, lg_report], from_date, to_date)

    total = len(manifest.papers)
    logger.info(f"=== Corpus: {total} papers ===")
    logger.info(f"  primary category: {manifest.category_counts()}")
    logger.info(f"  distinct topics:  {len(manifest.topic_counts())}")

    if total < eval_config.TARGET_TOTAL:
        logger.warning(
            f"Selected {total} of {eval_config.TARGET_TOTAL} target papers. "
            "Widen --from-date/--to-date or add terms to the short topics above."
        )

    manifest.save(output)
    logger.info(f"Next: python -m src.evaluation.corpus.build --manifest {output}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.corpus.discover",
        description="Select a topically diverse cs.AI + cs.LG corpus and write the manifest.",
    )
    parser.add_argument("--from-date", default=eval_config.CORPUS_FROM_DATE, help="Inclusive lower bound, YYYYMMDD")
    parser.add_argument("--to-date", default=None, help="Inclusive upper bound, YYYYMMDD (default: today)")
    parser.add_argument(
        "--sort-by",
        default="relevance",
        choices=["relevance", "submittedDate", "lastUpdatedDate"],
        help="arXiv sort key. 'relevance' keeps each slice on-topic; the date window already enforces recency.",
    )
    parser.add_argument("--output", type=Path, default=eval_config.CORPUS_MANIFEST, help="Manifest destination")
    parser.add_argument("--dry-run", action="store_true", help="Print the queries and exit without calling arXiv")
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG output on the console")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    :returns: Process exit code.
    """
    args = parse_args(argv)
    eval_config.ensure_dirs()

    run_id = eval_config.new_run_id()
    setup_logging(eval_config.log_path(run_id, "corpus_discover"), verbose=args.verbose)

    to_date = args.to_date or today_stamp()

    try:
        return asyncio.run(
            discover(
                from_date=args.from_date,
                to_date=to_date,
                sort_by=args.sort_by,
                output=args.output,
                dry_run=args.dry_run,
            )
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted before the manifest was written")
        return 130
    except Exception as exc:
        logger.exception(f"Discovery failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
